import numpy as np
import torch
import torch.nn.functional as F


def _scale(value, orig_hw: tuple[int, int], size: int, device) -> torch.Tensor:
    coords = torch.as_tensor(value, dtype=torch.float32, device=device)
    height, width = orig_hw
    return coords * coords.new_tensor([size / width, size / height])


def build_points(coords, labels, orig_hw: tuple[int, int], size: int, device):
    if coords is None:
        return None

    coords = _scale(coords, orig_hw, size, device)
    if labels is None:
        labels = torch.ones(coords.shape[:-1], dtype=torch.int, device=device)
    else:
        labels = torch.as_tensor(labels, dtype=torch.int, device=device)

    if coords.ndim == 2:
        coords = coords[None]
        labels = labels[None]
    return coords, labels


def build_box(value, orig_hw: tuple[int, int], size: int, device) -> torch.Tensor:
    if value is None:
        return None

    coords = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(
        -1, 2, 2
    )
    coords = _scale(coords, orig_hw, size, device)
    labels = torch.tensor((2, 3), dtype=torch.int, device=device)
    return coords, labels.expand(coords.shape[0], 2)


def build_mask(value, size: tuple[int, int], device):
    if value is None:
        return None

    out = torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
    if out.ndim == 2:
        out = out[None, None, :, :]
    elif out.ndim == 3:
        out = out[:, None, :, :]

    if out.shape[-2:] != size:
        out = F.interpolate(
            out,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return out
