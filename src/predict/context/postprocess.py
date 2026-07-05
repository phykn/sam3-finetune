from collections.abc import Sequence

import numpy as np

from ...ops.box import calc_iou
from ...types import ContextPrediction


def nms_context_predictions(
    predictions: Sequence[ContextPrediction],
    iou_threshold: float,
    max_masks: int | None,
) -> list[ContextPrediction]:
    kept: list[ContextPrediction] = []
    for prediction in predictions:
        same_concept = [
            existing
            for existing in kept
            if existing.concept_id == prediction.concept_id
        ]
        if all(
            calc_iou(prediction.bbox, existing.bbox) <= iou_threshold
            for existing in same_concept
        ):
            if max_masks is not None and len(same_concept) >= max_masks:
                continue
            kept.append(prediction)
    return kept


def context_prediction_to_full_mask(prediction: ContextPrediction) -> np.ndarray:
    x0, y0, x1, y1 = prediction.bbox
    expected_shape = (y1 - y0, x1 - x0)
    if prediction.segmentation.shape != expected_shape:
        raise ValueError(
            "segmentation shape must match bbox size: "
            f"got {prediction.segmentation.shape}, expected {expected_shape}"
        )
    width, height = prediction.image_size
    full_mask = np.zeros((height, width), dtype=bool)
    full_mask[y0:y1, x0:x1] = prediction.segmentation.astype(bool)
    return full_mask
