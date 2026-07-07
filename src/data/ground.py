import numpy as np
import torch


def build_points(
    coords: object | None,
    labels: object | None,
    orig_hw: tuple[int, int],
    device: str | torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if coords is None:
        return None, None

    points = torch.as_tensor(coords, dtype=torch.float32, device=device).reshape(-1, 2)
    height, width = orig_hw
    points = points / points.new_tensor([width, height])
    labels = torch.as_tensor(labels, dtype=torch.long, device=device).reshape(-1)
    return points[:, None, :], labels[:, None]


def build_boxes(
    boxes: object | None,
    orig_hw: tuple[int, int],
    device: str | torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if boxes is None:
        return None, None

    boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4)
    height, width = orig_hw
    scale = boxes.new_tensor([width, height, width, height])
    boxes = boxes / scale
    x0, y0, x1, y1 = boxes.unbind(-1)
    boxes = torch.stack(
        [
            (x0 + x1) * 0.5,
            (y0 + y1) * 0.5,
            x1 - x0,
            y1 - y0,
        ],
        dim=-1,
    )
    labels = torch.ones(boxes.shape[0], dtype=torch.long, device=device)
    return boxes[:, None, :], labels[:, None]


def build_masks(
    masks: object | None,
    device: str | torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if masks is None:
        return None, None

    masks = torch.as_tensor(np.asarray(masks), dtype=torch.float32, device=device)
    if masks.dim() == 2:
        masks = masks[None, None, None]
    elif masks.dim() == 3:
        masks = masks[:, None, None]
    labels = torch.ones(masks.shape[0], 1, dtype=torch.long, device=device)
    return masks, labels
