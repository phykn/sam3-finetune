import numpy as np


def calc_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_mask = np.asarray(first, dtype=bool)
    second_mask = np.asarray(second, dtype=bool)
    intersection = np.logical_and(first_mask, second_mask).sum()
    union = np.logical_or(first_mask, second_mask).sum()
    if int(union) == 0:
        return 0.0
    return float(intersection / union)
