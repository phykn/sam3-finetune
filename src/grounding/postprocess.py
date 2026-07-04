import numpy as np

from .types import GroundingPrediction
from ..metrics.mask import mask_iou


def filter_grounding_prediction(
    prediction: GroundingPrediction,
    *,
    score_threshold: float = 0.0,
    mask_nms_thresh: float | None = 0.7,
    max_masks: int | None = None,
) -> GroundingPrediction:
    if max_masks is not None and max_masks <= 0:
        raise ValueError("max_masks must be positive or None")
    if mask_nms_thresh is not None and not 0.0 <= mask_nms_thresh <= 1.0:
        raise ValueError("mask_nms_thresh must be between 0 and 1 or None")

    order = np.argsort(-prediction.scores)
    selected: list[int] = []
    for index in order.tolist():
        if float(prediction.scores[index]) < float(score_threshold):
            continue
        if mask_nms_thresh is not None and any(
            mask_iou(prediction.masks[index], prediction.masks[kept]) > mask_nms_thresh
            for kept in selected
        ):
            continue
        selected.append(index)
        if max_masks is not None and len(selected) >= max_masks:
            break

    indices = np.asarray(selected, dtype=np.int64)
    return GroundingPrediction(
        masks=prediction.masks[indices],
        mask_logits=prediction.mask_logits[indices],
        boxes_xyxy=prediction.boxes_xyxy[indices],
        scores=prediction.scores[indices],
    )
