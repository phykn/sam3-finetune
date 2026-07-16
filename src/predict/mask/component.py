import cv2
import numpy as np


def largest(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim < 2:
        raise ValueError("mask must have at least two dimensions")

    out = np.zeros_like(mask)
    if mask.shape[-2] == 0 or mask.shape[-1] == 0:
        return out

    for index in np.ndindex(mask.shape[:-2]):
        value = np.ascontiguousarray(mask[index], dtype=np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            value,
            connectivity=8,
        )
        if count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            out[index] = labels == int(areas.argmax()) + 1
    return out
