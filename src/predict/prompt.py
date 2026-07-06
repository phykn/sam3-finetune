import numpy as np
import torch
import torch.nn.functional as F

from . import transform
from .result import ImageEmbed


def sam_prompt(
    *,
    prompt_encoder,
    embed: ImageEmbed,
    image_size: int,
    device: torch.device,
    point_coords=None,
    point_labels=None,
    box=None,
    mask=None,
):
    point_prompt = points(embed, image_size, device, point_coords, point_labels)
    box_prompt = boxes(embed, image_size, device, box)
    point_prompt = merge(box_prompt, point_prompt)
    mask_prompt = masks(prompt_encoder, device, mask)

    if point_prompt is None and mask_prompt is None:
        raise ValueError("prompt is required")
    if point_prompt is None:
        point_prompt = dummy(mask_prompt.shape[0], device)
    return point_prompt, mask_prompt


def points(embed, image_size, device, coords, labels):
    if coords is None:
        return None

    coords = transform.points(coords, embed.orig_hw, image_size, device)
    if labels is None:
        labels = torch.ones(coords.shape[:-1], dtype=torch.int, device=device)
    else:
        labels = torch.as_tensor(labels, dtype=torch.int, device=device)

    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
        labels = labels.unsqueeze(0)
    if coords.ndim != 3 or labels.ndim != 2:
        raise ValueError("point prompt must have BxNx2 coordinates and BxN labels")
    if coords.shape[:2] != labels.shape:
        raise ValueError("point coordinates and labels must align")
    return coords, labels


def boxes(embed, image_size, device, value):
    if value is None:
        return None

    coords = transform.box(value, embed.orig_hw, image_size, device)
    labels = torch.tensor([2, 3], dtype=torch.int, device=device)
    labels = labels.expand(coords.shape[0], 2)
    return coords, labels


def masks(prompt_encoder, device, value):
    if value is None:
        return None

    out = torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
    if out.ndim == 2:
        out = out[None, None, :, :]
    elif out.ndim == 3:
        out = out[:, None, :, :]
    elif out.ndim != 4:
        raise ValueError("mask prompt must have 2, 3, or 4 dimensions")

    if out.shape[-2:] != prompt_encoder.mask_input_size:
        out = F.interpolate(
            out,
            size=prompt_encoder.mask_input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return out


def merge(first, second):
    if first is None:
        return second
    if second is None:
        return first
    if first[0].shape[0] != second[0].shape[0]:
        raise ValueError("box and point prompt batches must align")
    return torch.cat([first[0], second[0]], dim=1), torch.cat(
        [first[1], second[1]],
        dim=1,
    )


def dummy(batch_size: int, device: torch.device):
    return (
        torch.zeros(batch_size, 1, 2, device=device),
        -torch.ones(batch_size, 1, dtype=torch.int, device=device),
    )
