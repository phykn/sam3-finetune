from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundingPrediction:
    masks: np.ndarray
    mask_logits: np.ndarray
    boxes_xyxy: np.ndarray
    scores: np.ndarray
