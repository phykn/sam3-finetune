import numpy as np
from PIL import Image
from scripts.video_memory_reference import (
    build_reference_mask,
    predict_sam_mask_from_box,
    resolve_box,
)
from src.predict.refine import MaskRefiner, select_best_mask


def test_select_best_mask_uses_highest_score_candidate() -> None:
    masks = np.zeros((1, 3, 2, 3), dtype=bool)
    masks[0, 0, 0, 0] = True
    masks[0, 1, 0:2, 1] = True
    masks[0, 2, 1, 2] = True
    scores = np.array([[0.2, 0.9, 0.4]], dtype=np.float32)

    selected, score, index = select_best_mask(masks, scores)

    assert index == 1
    assert score == np.float32(0.9)
    np.testing.assert_array_equal(selected, masks[0, 1])


def test_select_best_mask_rejects_score_count_mismatch() -> None:
    masks = np.zeros((2, 3, 4), dtype=bool)
    scores = np.array([0.1], dtype=np.float32)

    try:
        select_best_mask(masks, scores)
    except ValueError as exc:
        assert "score" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_build_reference_mask_box_source_clips_to_image_bounds() -> None:
    image = Image.new("RGB", (6, 4), color=(0, 0, 0))

    result = build_reference_mask(
        image=image,
        box=[-2, 1, 4, 8],
        source="box",
        checkpoint=None,
        device="cpu",
    )

    assert result.source == "box"
    assert result.score is None
    assert result.selected_index is None
    assert result.mask.shape == (4, 6)
    assert int(result.mask.sum()) == 12
    assert result.mask[:, 4:].sum() == 0


def test_resolve_box_defaults_to_center_region() -> None:
    image = Image.new("RGB", (100, 80), color=(0, 0, 0))

    assert resolve_box(image, None) == [25, 16, 75, 68]


def test_predict_sam_mask_from_box_keeps_box_during_refinement(monkeypatch) -> None:
    import src.predict.prompted as prompted

    class FakePredictor:
        instances = []

        def __init__(self) -> None:
            self.calls = []
            FakePredictor.instances.append(self)

        @classmethod
        def from_checkpoint(cls, checkpoint, device):
            return cls()

        def set_image(self, image):
            self.image = image

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                masks = np.zeros((1, 3, 4, 6), dtype=bool)
                masks[0, 1, 1:3, 2:5] = True
                scores = np.array([[0.1, 0.9, 0.2]], dtype=np.float32)
                low_res = np.zeros((1, 3, 2, 2), dtype=np.float32)
                low_res[0, 1] = 1.0
                return masks, scores, low_res
            masks = np.zeros((1, 1, 4, 6), dtype=bool)
            masks[0, 0, 1:3, 2:5] = True
            scores = np.array([[0.8]], dtype=np.float32)
            low_res = np.zeros((1, 1, 2, 2), dtype=np.float32)
            return masks, scores, low_res

    monkeypatch.setattr(
        prompted.Sam3Predictor,
        "from_checkpoint",
        FakePredictor.from_checkpoint,
    )
    image = Image.new("RGB", (6, 4), color=(0, 0, 0))

    result = predict_sam_mask_from_box(
        image=image,
        box=[1, 1, 5, 3],
        checkpoint="checkpoint.pt",
        device="cpu",
    )

    fake = FakePredictor.instances[-1]
    np.testing.assert_array_equal(fake.calls[0]["box"], np.array([1, 1, 5, 3]))
    np.testing.assert_array_equal(fake.calls[1]["box"], np.array([1, 1, 5, 3]))
    assert fake.calls[1]["mask_input"].shape == (2, 2)
    assert result.source == "sam"


def test_mask_refiner_preserves_prompts_during_second_pass() -> None:
    class FakePredictor:
        def __init__(self) -> None:
            self.calls = []

        def set_image(self, image):
            self.image = image

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            masks = np.zeros((1, 1, 4, 6), dtype=bool)
            masks[0, 0, 1:3, 2:5] = True
            scores = np.array([[0.8]], dtype=np.float32)
            low_res = np.zeros((1, 1, 2, 2), dtype=np.float32)
            return masks, scores, low_res

    fake = FakePredictor()
    refiner = MaskRefiner(fake)
    image = Image.new("RGB", (6, 4), color=(0, 0, 0))
    low_res = np.ones((2, 2), dtype=np.float32)

    result = refiner.refine(
        image=image,
        point_coords=np.array([[3, 2]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int64),
        box=np.array([1, 1, 5, 3], dtype=np.float32),
        mask_input=low_res,
    )

    np.testing.assert_array_equal(
        fake.calls[0]["point_coords"],
        np.array([[3, 2]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        fake.calls[0]["point_labels"],
        np.array([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        fake.calls[0]["box"],
        np.array([1, 1, 5, 3], dtype=np.float32),
    )
    assert fake.calls[0]["mask_input"].shape == (2, 2)
    assert fake.calls[0]["multimask_output"] is False
    assert result.mask.shape == (4, 6)
    assert result.score == np.float32(0.8)
