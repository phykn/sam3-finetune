import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def test_script_files_do_not_import_other_script_files() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    offenders: list[str] = []
    for path in scripts_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "scripts":
                offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "scripts."
            ):
                offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "scripts" or alias.name.startswith("scripts."):
                        offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == []


def test_ground_script_refines_logits_in_one_batch(monkeypatch) -> None:
    import scripts.ground as ground

    class FakeSingle:
        instances = []

        def __init__(self):
            self.encode_calls = 0
            self.predict_embed_calls = []
            self.refine_calls = 0
            FakeSingle.instances.append(self)

        @classmethod
        def from_path(cls, _path, _config):
            return cls()

        def encode(self, image):
            assert not torch.is_grad_enabled()
            self.encode_calls += 1
            return {"orig_hw": (image.height, image.width)}

        def predict_embed(self, embed, mask=None, multimask=True):
            assert not torch.is_grad_enabled()
            self.predict_embed_calls.append((embed, np.asarray(mask), multimask))
            count = len(mask)
            masks = np.zeros((count, 1, 4, 5), dtype=bool)
            masks[:, :, 1:3, 2:4] = True
            scores = np.arange(count, dtype=np.float32)[:, None] + 0.25
            return {"masks": masks, "scores": scores}

        def refine(self, *_args, **_kwargs):
            self.refine_calls += 1
            raise AssertionError("refine should not be called per logit")

    monkeypatch.setattr(ground, "SinglePredictor", FakeSingle)
    out = {
        "frog": {
            "logits": np.ones((3, 2, 2), dtype=np.float32),
        }
    }

    ground.refine(Image.new("RGB", (5, 4)), out, "cpu")

    fake = FakeSingle.instances[0]
    assert fake.encode_calls == 1
    assert len(fake.predict_embed_calls) == 1
    _embed, mask, multimask = fake.predict_embed_calls[0]
    assert mask.shape == (3, 2, 2)
    assert multimask is False
    assert fake.refine_calls == 0
    assert out["frog"]["refined_masks"].shape == (3, 4, 5)
    np.testing.assert_allclose(
        out["frog"]["refined_scores"],
        np.array([0.25, 1.25, 2.25], dtype=np.float32),
    )


def test_ground_script_segments_reference_points_in_one_batch(monkeypatch) -> None:
    import scripts.ground as ground

    concepts = [
        {
            "name": "flower",
            "points": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "color": (255, 180, 0),
        }
    ]

    class FakeSingle:
        instances = []

        def __init__(self):
            self.encode_calls = 0
            self.predict_embed_calls = []
            self.predict_calls = 0
            FakeSingle.instances.append(self)

        @classmethod
        def from_path(cls, _path, _config):
            return cls()

        def encode(self, image):
            assert not torch.is_grad_enabled()
            self.encode_calls += 1
            return {"orig_hw": (image.height, image.width)}

        def predict_embed(self, embed, point_coords=None, point_labels=None, **kwargs):
            assert not torch.is_grad_enabled()
            self.predict_embed_calls.append(
                (embed, np.asarray(point_coords), np.asarray(point_labels), kwargs)
            )
            masks = np.zeros((2, 3, 4, 5), dtype=bool)
            masks[0, 1, 1:3, 1:3] = True
            masks[1, 2, 2:4, 2:4] = True
            scores = np.array([[0.1, 0.9, 0.2], [0.3, 0.4, 0.8]], dtype=np.float32)
            return {"masks": masks, "scores": scores}

        def predict(self, *_args, **_kwargs):
            self.predict_calls += 1
            raise AssertionError("predict should not be called per point")

    monkeypatch.setattr(ground, "CONCEPTS", concepts)
    monkeypatch.setattr(ground, "SinglePredictor", FakeSingle)

    refs = ground.segment_refs(Image.new("RGB", (5, 4)), "cpu")

    fake = FakeSingle.instances[0]
    assert fake.encode_calls == 1
    assert len(fake.predict_embed_calls) == 1
    _embed, point_coords, point_labels, kwargs = fake.predict_embed_calls[0]
    assert point_coords.shape == (2, 1, 2)
    assert point_labels.shape == (2, 1)
    assert kwargs["multimask"] is True
    assert fake.predict_calls == 0
    assert refs[0]["masks"].shape == (2, 4, 5)
    np.testing.assert_allclose(refs[0]["scores"], np.array([0.9, 0.8]))


def test_grid_script_uses_small_gpu_batch(monkeypatch, tmp_path) -> None:
    import scripts.grid as grid

    calls = []

    class FakePredictor:
        before = []

        @classmethod
        def from_path(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return cls()

        def predict(self, _image):
            return []

        def iter_points(self, _size):
            return iter(())

    class FakeSheet:
        def save(self, path):
            Path(path).touch()
            return None

    monkeypatch.setattr(grid, "GridPredictor", FakePredictor)
    monkeypatch.setattr(grid.Image, "open", lambda _path: Image.new("RGB", (5, 4)))
    monkeypatch.setattr(grid, "make_sheet", lambda *_args: FakeSheet())
    monkeypatch.setattr(grid, "OUT", tmp_path)

    grid.main()

    assert calls
    assert calls[0][1]["batch_size"] == 4
    assert (tmp_path / "frog_grid.json").exists()
    assert '\n  "schema_version"' in (tmp_path / "frog_grid.json").read_text(
        encoding="utf-8"
    )
    data = json.loads((tmp_path / "frog_grid.json").read_text(encoding="utf-8"))
    assert set(data) == {"schema_version", "image", "objects"}


def test_finetune_grid_script_writes_json(monkeypatch, tmp_path) -> None:
    import scripts.finetune_grid as grid

    calls = []

    class FakePredictor:
        before = []

        def predict(self, _image):
            calls.append("predict")
            return [
                {
                    "score": 0.75,
                    "point": (2.0, 2.0),
                    "class_scores": np.array([0.8, 0.2], dtype=np.float32),
                }
            ]

        @staticmethod
        def expand_mask(_item, image_size):
            mask = np.zeros((image_size[1], image_size[0]), dtype=bool)
            mask[1:3, 2:4] = True
            return mask

    class FakeSheet:
        def save(self, path):
            Path(path).touch()
            return None

    monkeypatch.setattr(grid, "make_predictor", lambda _device: FakePredictor())
    monkeypatch.setattr(grid.Image, "open", lambda _path: Image.new("RGB", (5, 4)))
    monkeypatch.setattr(grid, "make_sheet", lambda *_args: FakeSheet())
    monkeypatch.setattr(grid, "OUT", tmp_path)

    grid.main()

    assert calls == ["predict"]
    assert (tmp_path / "frog_grid.json").exists()
    assert (tmp_path / "frog_grid.png").exists()
    data = json.loads((tmp_path / "frog_grid.json").read_text(encoding="utf-8"))
    assert set(data) == {"schema_version", "image", "objects"}
    assert data["objects"][0]["metrics"]["class_scores"] == pytest.approx([0.8, 0.2])


def test_single_script_writes_json_and_draws_from_json(monkeypatch, tmp_path) -> None:
    import scripts.single as single

    class FakePredictor:
        @classmethod
        def from_path(cls, *_args, **_kwargs):
            return cls()

        def predict(self, *_args, **_kwargs):
            mask = np.zeros((4, 5), dtype=bool)
            mask[1:3, 2:4] = True
            return {
                "masks": np.array([mask]),
                "scores": np.array([0.75], dtype=np.float32),
            }

    monkeypatch.setattr(single, "SinglePredictor", FakePredictor)
    monkeypatch.setattr(single.Image, "open", lambda _path: Image.new("RGB", (5, 4)))
    monkeypatch.setattr(single, "OUT", tmp_path)

    single.main()

    data = single.load_result(tmp_path / "frog_single.json")

    assert (tmp_path / "frog_single.json").exists()
    assert (tmp_path / "frog_single.png").exists()
    assert '\n  "schema_version"' in (tmp_path / "frog_single.json").read_text(
        encoding="utf-8"
    )
    assert data.objects[0].metrics["score"] == 0.75
    assert data.objects[0].mask(data.image.shape).sum() == 4


def test_finetune_single_script_writes_json_and_draws_from_json(
    monkeypatch,
    tmp_path,
) -> None:
    import scripts.finetune_single as single

    class FakePredictor:
        def predict(self, *_args, **_kwargs):
            mask = np.zeros((4, 5), dtype=bool)
            mask[1:3, 2:4] = True
            return {
                "masks": np.array([mask]),
                "scores": np.array([0.75], dtype=np.float32),
                "class_scores": np.array([[0.8, 0.2]], dtype=np.float32),
            }

    monkeypatch.setattr(single, "make_predictor", lambda _device: FakePredictor())
    monkeypatch.setattr(single.Image, "open", lambda _path: Image.new("RGB", (5, 4)))
    monkeypatch.setattr(single, "OUT", tmp_path)

    single.main()

    data = single.load_result(tmp_path / "frog_single.json")

    assert (tmp_path / "frog_single.json").exists()
    assert (tmp_path / "frog_single.png").exists()
    assert data.objects[0].metrics["score"] == 0.75
    assert data.objects[0].metrics["class_scores"] == pytest.approx([0.8, 0.2])
    assert data.objects[0].mask(data.image.shape).sum() == 4


def test_ground_script_round_trips_json_for_drawing(tmp_path) -> None:
    import scripts.ground as ground

    ref = Image.new("RGB", (5, 4))
    target = Image.new("RGB", (5, 4))
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True
    refs = [
        {
            "name": "frog",
            "points": np.array([[2.0, 1.0]], dtype=np.float32),
            "color": (255, 60, 60),
            "masks": np.array([mask]),
            "scores": np.array([0.9], dtype=np.float32),
        }
    ]
    out = {
        "frog": {
            "scores": np.array([0.8], dtype=np.float32),
            "similarities": np.array([0.7], dtype=np.float32),
            "boxes": np.array([[2.0, 1.0, 4.0, 3.0]], dtype=np.float32),
            "masks": np.array([mask]),
            "refined_masks": np.array([mask]),
            "refined_scores": np.array([0.85], dtype=np.float32),
        }
    }

    ground.save_result(
        ground.make_result(ref, refs, target, out), tmp_path / "out.json"
    )
    data = ground.load_result(tmp_path / "out.json")
    sheet = ground.make_sheet(data)
    raw = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))

    assert sheet.size == (10, 4)
    assert set(raw) == {"schema_version", "image", "objects"}
    assert '\n  "schema_version"' in (tmp_path / "out.json").read_text(encoding="utf-8")
    assert data.objects[0].meta["name"] == "frog"
    assert data.objects[0].mask(data.image.shape).sum() == 4
    np.testing.assert_allclose(data.objects[0].metrics["score"], 0.8)
    np.testing.assert_allclose(data.objects[0].metrics["refined_score"], 0.85)


def test_video_script_round_trips_json_for_drawing(tmp_path) -> None:
    import scripts.video as video

    frames = [Image.new("RGB", (5, 4)) for _index in range(3)]
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True
    outputs = [
        {"masks": np.array([mask]), "scores": np.array([0.6], dtype=np.float32)},
        {"masks": np.array([mask]), "scores": np.array([0.7], dtype=np.float32)},
    ]

    video.save_result(
        video.make_result(frames, mask, 0.9, outputs),
        tmp_path / "out.json",
    )
    data = video.load_result(tmp_path / "out.json")
    sheet = video.make_sheet(data)
    raw = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))

    assert sheet.size == (10, 4)
    assert set(raw) == {"schema_version", "image", "objects"}
    assert '\n  "schema_version"' in (tmp_path / "out.json").read_text(encoding="utf-8")
    assert data.objects[0].mask(data.image.shape).sum() == 4
    np.testing.assert_allclose(data.objects[0].metrics["score"], 0.7)


def test_runtime_scripts_use_predict_api_only() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    offenders: list[str] = []
    runtime_scripts = (
        "single.py",
        "grid.py",
        "finetune_single.py",
        "finetune_grid.py",
        "ground.py",
        "video.py",
    )

    for name in runtime_scripts:
        path = scripts_dir / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module.startswith("src.ml"):
                offenders.append(f"{name}:{node.lineno}:{node.module}")
            if node.module.startswith("src.predict.") and "_ops" in node.module:
                offenders.append(f"{name}:{node.lineno}:{node.module}")

    assert offenders == []
