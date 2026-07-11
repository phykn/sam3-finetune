import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def mock_path_image(monkeypatch, module, size=(5, 4)):
    real_open = Image.open

    def open_image(value):
        if isinstance(value, (str, Path)):
            return Image.new("RGB", size)
        return real_open(value)

    monkeypatch.setattr(module.Image, "open", open_image)


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


def test_ground_benchmark_import_has_no_runtime_side_effects() -> None:
    import scripts.bench_ground as benchmark

    assert callable(benchmark.main)


def test_ground_script_refines_logits_in_one_batch(monkeypatch) -> None:
    import scripts.ground as ground

    class FakeSingle:
        instances = []

        def __init__(self):
            self.encode_calls = 0
            self.refine_embed_calls = []
            self.refine_calls = 0
            FakeSingle.instances.append(self)

        @classmethod
        def from_path(cls, _path, device="cuda"):
            return cls()

        def encode(self, image):
            assert not torch.is_grad_enabled()
            self.encode_calls += 1
            return {"orig_hw": (image.height, image.width)}

        def refine_embed(self, embed, logit):
            assert not torch.is_grad_enabled()
            self.refine_embed_calls.append((embed, np.asarray(logit)))
            return [
                {
                    "box": (2, 1, 4, 3),
                    "roi": np.ones((2, 2), dtype=bool),
                    "metrics": {"score": float(index) + 0.25},
                }
                for index in range(len(logit))
            ]

        def refine(self, *_args, **_kwargs):
            self.refine_calls += 1
            raise AssertionError("refine should not be called per logit")

    monkeypatch.setattr(ground, "SinglePredictor", FakeSingle)
    objects = [
        {
            "object_id": index + 1,
            "class_id": index,
            "box": (0, 0, 2, 2),
            "roi": np.ones((2, 2), dtype=bool),
            "logit": np.ones((2, 2), dtype=np.float32),
            "metrics": {"score": 0.5, "similarity": 0.6},
        }
        for index in range(3)
    ]

    ground.refine(Image.new("RGB", (5, 4)), objects, "cpu")

    fake = FakeSingle.instances[0]
    assert fake.encode_calls == 1
    assert len(fake.refine_embed_calls) == 1
    _embed, logit = fake.refine_embed_calls[0]
    assert logit.shape == (3, 2, 2)
    assert fake.refine_calls == 0
    assert all(item["roi"].shape == (2, 2) for item in objects)
    assert all("mask" not in item for item in objects)
    assert [item["metrics"]["refined_score"] for item in objects] == pytest.approx(
        [0.25, 1.25, 2.25]
    )


def test_ground_script_reads_reference_boxes_and_classes() -> None:
    import scripts.ground as ground
    from src.data.sample import Image as DataImage
    from src.data.sample import Object, Sample

    sample = Sample(
        image=DataImage(array=np.zeros((4, 5, 3), dtype=np.uint8)),
        objects=[
            Object(1, 3, (0, 0, 2, 2), np.ones((2, 2), dtype=np.uint8)),
            Object(2, 3, (2, 1, 5, 4), np.ones((3, 3), dtype=np.uint8)),
            Object(3, 7, (1, 1, 3, 3), np.ones((2, 2), dtype=np.uint8)),
        ],
    )

    boxes, class_ids = ground.reference_arrays(sample)

    np.testing.assert_array_equal(
        boxes,
        [[0, 0, 2, 2], [2, 1, 5, 4], [1, 1, 3, 3]],
    )
    assert class_ids.tolist() == [3, 3, 7]


def test_grid_script_uses_small_gpu_batch(monkeypatch, tmp_path) -> None:
    import scripts.grid as grid

    calls = []

    class FakePredictor:
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
    mock_path_image(monkeypatch, grid)
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
        def predict(self, _image):
            calls.append("predict")
            return [
                {
                    "object_id": 1,
                    "class_id": None,
                    "box": (2, 1, 4, 3),
                    "roi": np.ones((2, 2), dtype=bool),
                    "points": [[2.0, 2.0, 1]],
                    "metrics": {"score": 0.75, "class_scores": [0.8, 0.2]},
                }
            ]

    class FakeSheet:
        def save(self, path):
            Path(path).touch()
            return None

    monkeypatch.setattr(grid, "make_predictor", lambda _device: FakePredictor())
    mock_path_image(monkeypatch, grid)
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
            return [
                {
                    "object_id": 1,
                    "class_id": None,
                    "box": (2, 1, 4, 3),
                    "roi": np.ones((2, 2), dtype=bool),
                    "prompt_index": 0,
                    "candidate_index": 0,
                    "metrics": {"score": 0.75},
                }
            ]

    monkeypatch.setattr(single, "SinglePredictor", FakePredictor)
    mock_path_image(monkeypatch, single)
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
            return [
                {
                    "object_id": 1,
                    "class_id": None,
                    "box": (2, 1, 4, 3),
                    "roi": np.ones((2, 2), dtype=bool),
                    "prompt_index": 0,
                    "candidate_index": 0,
                    "metrics": {"score": 0.75, "class_scores": [0.8, 0.2]},
                }
            ]

    monkeypatch.setattr(single, "make_predictor", lambda _device: FakePredictor())
    mock_path_image(monkeypatch, single)
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

    target = Image.new("RGB", (5, 4))
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True
    objects = [
        {
            "object_id": 1,
            "class_id": 3,
            "box": (2, 1, 4, 3),
            "roi": mask[1:3, 2:4],
            "logit": np.ones((2, 2), dtype=np.float32),
            "metrics": {
                "score": 0.8,
                "similarity": 0.7,
                "refined_score": 0.85,
            },
        }
    ]

    ground.save_result(ground.make_result(target, objects), tmp_path / "out.json")
    data = ground.load_result(tmp_path / "out.json")
    sheet = ground.make_sheet(data)
    raw = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))

    assert sheet.size == (10, 4)
    assert set(raw) == {"schema_version", "image", "objects"}
    assert '\n  "schema_version"' in (tmp_path / "out.json").read_text(encoding="utf-8")
    assert data.objects[0].class_id == 3
    assert data.objects[0].mask(data.image.shape).sum() == 4
    np.testing.assert_allclose(data.objects[0].metrics["score"], 0.8)
    np.testing.assert_allclose(data.objects[0].metrics["similarity"], 0.7)
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
        "finetune.py",
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
