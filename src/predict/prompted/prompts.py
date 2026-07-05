import numpy as np
import torch
import torch.nn.functional as F

from ...types import Sam3ImageEmbedding


def prepare_prompt_tensors(
    *,
    transforms,
    prompt_encoder,
    device: torch.device | str,
    embedding: Sam3ImageEmbedding,
    point_coords: np.ndarray | torch.Tensor | None = None,
    point_labels: np.ndarray | torch.Tensor | None = None,
    box: np.ndarray | torch.Tensor | None = None,
    mask_input: np.ndarray | torch.Tensor | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
    device = torch.device(device)
    concat_points = prepare_point_prompt(
        transforms=transforms,
        device=device,
        embedding=embedding,
        point_coords=point_coords,
        point_labels=point_labels,
    )

    box_points = prepare_box_prompt(
        transforms=transforms,
        device=device,
        embedding=embedding,
        box=box,
    )
    concat_points = merge_point_prompts(box_points, concat_points)

    mask_prompt = prepare_mask_prompt(
        prompt_encoder=prompt_encoder,
        device=device,
        mask_input=mask_input,
    )

    if concat_points is None and mask_prompt is None:
        raise ValueError("Provide at least one point, box, or mask prompt.")
    if concat_points is None and mask_prompt is not None:
        concat_points = dummy_negative_point(mask_prompt.shape[0], device=device)
    return concat_points, mask_prompt


def prepare_point_prompt(
    *,
    transforms,
    device: torch.device,
    embedding: Sam3ImageEmbedding,
    point_coords: np.ndarray | torch.Tensor | None,
    point_labels: np.ndarray | torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if point_coords is None:
        return None
    if point_labels is None:
        raise ValueError("point_labels must be supplied with point_coords")

    coords = transforms.transform_coords(point_coords, embedding.orig_hw).to(device)
    labels = torch.as_tensor(point_labels, dtype=torch.int, device=device)
    if coords.ndim == 2:
        coords = coords[None, ...]
        labels = labels[None, ...]
    if coords.ndim != 3 or labels.ndim != 2:
        raise ValueError("point prompts must have shape BxNx2 and BxN")
    if coords.shape[:2] != labels.shape:
        raise ValueError("point coordinates and labels must align")
    return coords, labels


def prepare_box_prompt(
    *,
    transforms,
    device: torch.device,
    embedding: Sam3ImageEmbedding,
    box: np.ndarray | torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if box is None:
        return None

    box_coords = transforms.transform_box(box, embedding.orig_hw).to(device)
    box_labels = torch.tensor([2, 3], dtype=torch.int, device=device)
    box_labels = box_labels.expand(box_coords.shape[0], 2)
    return box_coords, box_labels


def prepare_mask_prompt(
    *,
    prompt_encoder,
    device: torch.device,
    mask_input: np.ndarray | torch.Tensor | None,
) -> torch.Tensor | None:
    if mask_input is None:
        return None

    mask_prompt = torch.as_tensor(mask_input, dtype=torch.float32, device=device)
    if mask_prompt.ndim == 2:
        mask_prompt = mask_prompt[None, None, :, :]
    elif mask_prompt.ndim == 3:
        mask_prompt = mask_prompt[:, None, :, :]
    elif mask_prompt.ndim != 4:
        raise ValueError("mask_input must have 2, 3, or 4 dimensions")
    if mask_prompt.shape[-2:] != prompt_encoder.mask_input_size:
        mask_prompt = F.interpolate(
            mask_prompt,
            size=prompt_encoder.mask_input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return mask_prompt


def merge_point_prompts(
    first: tuple[torch.Tensor, torch.Tensor] | None,
    second: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if first is None:
        return second
    if second is None:
        return first
    if first[0].shape[0] != second[0].shape[0]:
        raise ValueError("point and box prompt batches must align")
    return (
        torch.cat([first[0], second[0]], dim=1),
        torch.cat([first[1], second[1]], dim=1),
    )


def dummy_negative_point(
    batch_size: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(batch_size, 1, 2, device=device),
        -torch.ones(batch_size, 1, dtype=torch.int, device=device),
    )
