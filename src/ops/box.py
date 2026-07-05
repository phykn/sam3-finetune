from collections.abc import Sequence

import numpy as np
import torch
from torchvision.ops import nms


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
    boxes: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor,
    iou_threshold: float,
) -> list[int]:
    boxes_t = _as_float_tensor(boxes)
    scores_t = _as_float_tensor(scores)
    if boxes_t.numel() == 0:
        return []
    if boxes_t.ndim != 2 or boxes_t.shape[1] != 4:
        raise ValueError("boxes must have shape Nx4")
    if scores_t.ndim != 1 or scores_t.shape[0] != boxes_t.shape[0]:
        raise ValueError("scores must have shape N")

    keep = nms(boxes_t, scores_t, float(iou_threshold))
    return [int(index) for index in keep.detach().cpu().tolist()]


def _as_float_tensor(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)
