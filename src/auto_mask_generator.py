from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np


def build_point_grid(points_per_side: int) -> np.ndarray:
    if points_per_side <= 0:
        raise ValueError("points_per_side must be a positive integer")
    offset = 1.0 / (2 * points_per_side)
    points = np.linspace(offset, 1.0 - offset, points_per_side, dtype=np.float32)
    xv, yv = np.meshgrid(points, points)
    return np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.float32)


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def calculate_stability_score(
    logits: np.ndarray,
    mask_threshold: float = 0.0,
    offset: float = 1.0,
) -> float:
    high = logits > (mask_threshold + offset)
    low = logits > (mask_threshold - offset)
    union = int(low.sum())
    if union == 0:
        return 0.0
    return float(high.sum() / union)


def box_area(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = box
    return max(float(x1) - float(x0), 0.0) * max(float(y1) - float(y0), 0.0)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(float(ax0), float(bx0))
    iy0 = max(float(ay0), float(by0))
    ix1 = min(float(ax1), float(bx1))
    iy1 = min(float(ay1), float(by1))
    intersection = box_area((ix0, iy0, ix1, iy1))
    union = box_area(box_a) + box_area(box_b) - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores, kind="mergesort")
    keep: list[int] = []
    for index in order:
        candidate = boxes[index]
        if all(box_iou(candidate, boxes[kept]) <= iou_threshold for kept in keep):
            keep.append(int(index))
    return keep


def batched(items: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
