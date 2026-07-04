from pathlib import Path

import numpy as np
from src.types import GroundingPrediction


def _prediction() -> GroundingPrediction:
    masks = np.zeros((4, 8, 8), dtype=bool)
    masks[0, 0:4, 0:4] = True
    masks[1, 1:5, 1:5] = True
    masks[2, 4:7, 4:7] = True
    masks[3, 6:8, 0:2] = True
    mask_logits = masks.astype(np.float32)
    boxes = np.array(
        [
            [0, 0, 4, 4],
            [1, 1, 5, 5],
            [4, 4, 7, 7],
            [0, 6, 2, 8],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7, 0.4], dtype=np.float32)
    return GroundingPrediction(
        masks=masks,
        mask_logits=mask_logits,
        boxes_xyxy=boxes,
        scores=scores,
    )


def test_filter_grounding_prediction_uses_model_score_and_mask_nms() -> None:
    from src.predict.grounding.postprocess import filter_grounding_prediction

    filtered = filter_grounding_prediction(
        _prediction(),
        score_threshold=0.5,
        mask_nms_thresh=0.25,
        max_masks=3,
    )

    np.testing.assert_allclose(filtered.scores, np.array([0.9, 0.7], dtype=np.float32))
    np.testing.assert_allclose(
        filtered.boxes_xyxy,
        np.array([[0, 0, 4, 4], [4, 4, 7, 7]], dtype=np.float32),
    )


def test_filter_grounding_prediction_allows_disabling_nms() -> None:
    from src.predict.grounding.postprocess import filter_grounding_prediction

    filtered = filter_grounding_prediction(
        _prediction(),
        score_threshold=0.5,
        mask_nms_thresh=None,
        max_masks=2,
    )

    np.testing.assert_allclose(filtered.scores, np.array([0.9, 0.8], dtype=np.float32))


def test_grounding_postprocess_lives_under_grounding_package() -> None:
    from src.predict.grounding.postprocess import filter_grounding_prediction

    assert filter_grounding_prediction.__module__ == "src.predict.grounding.postprocess"
    assert not Path("src/grounding_postprocess.py").exists()


def test_grounding_postprocess_does_not_expose_labeling_wrapper() -> None:
    import src.predict.grounding.postprocess as postprocess

    assert not hasattr(postprocess, "LabeledGroundingPrediction")
    assert not hasattr(postprocess, "label_grounding_prediction")
