from collections.abc import Sequence

import numpy as np
import torch


def convert_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = box.unbind(-1)
    return torch.stack(
        [
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ],
        dim=-1,
    )


def calc_area(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = box
    return max(float(x1) - float(x0), 0.0) * max(float(y1) - float(y0), 0.0)


def calc_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    ix0 = max(float(ax0), float(bx0))
    iy0 = max(float(ay0), float(by0))
    ix1 = min(float(ax1), float(bx1))
    iy1 = min(float(ay1), float(by1))

    intersection = calc_area((ix0, iy0, ix1, iy1))
    union = calc_area(box_a) + calc_area(box_b) - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def filter_boxes(
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
        if all(calc_iou(candidate, boxes[kept]) <= iou_threshold for kept in keep):
            keep.append(int(index))

    return keep
