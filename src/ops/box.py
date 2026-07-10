import numpy as np
import torch
from torchvision.ops import nms


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        [
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ],
        dim=-1,
    )


def nms_indices(
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
