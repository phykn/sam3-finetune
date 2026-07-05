from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from ...ops.box import filter_boxes
from ...types import MaskProposal
from .geometry import calculate_stability_score, mask_to_box, touches_internal_crop_edge


@dataclass(frozen=True)
class ProposalFilterConfig:
    pred_iou_thresh: float
    stability_score_thresh: float
    stability_score_offset: float
    min_mask_region_area: int
    filter_crop_edge_masks: bool


def count_proposals_by_crop_grid(
    proposals: Sequence[MaskProposal],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for proposal in proposals:
        counts[proposal.crop_grid] = counts.get(proposal.crop_grid, 0) + 1
    return dict(sorted(counts.items()))


def proposals_from_batch(
    points: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    low_res_masks: np.ndarray,
    crop_box: tuple[int, int, int, int],
    crop_grid: int,
    crop_index: int,
    full_size: tuple[int, int],
    *,
    config: ProposalFilterConfig,
) -> list[MaskProposal]:
    proposals: list[MaskProposal] = []
    for point_index, point in enumerate(points):
        for mask_index in range(masks.shape[1]):
            predicted_iou = float(scores[point_index, mask_index])
            if predicted_iou < config.pred_iou_thresh:
                continue
            mask = masks[point_index, mask_index].astype(bool)
            area = int(mask.sum())
            if area < config.min_mask_region_area:
                continue
            stability = calculate_stability_score(
                low_res_masks[point_index, mask_index],
                offset=config.stability_score_offset,
            )
            if stability < config.stability_score_thresh:
                continue
            proposal = proposal_from_mask(
                point,
                mask,
                predicted_iou=predicted_iou,
                stability=stability,
                crop_box=crop_box,
                crop_grid=crop_grid,
                crop_index=crop_index,
                full_size=full_size,
                filter_crop_edge_masks=config.filter_crop_edge_masks,
            )
            if proposal is not None:
                proposals.append(proposal)
    return proposals


def proposals_from_low_res_batch(
    points: np.ndarray,
    scores: np.ndarray | torch.Tensor,
    low_res_masks: np.ndarray | torch.Tensor,
    crop_hw: tuple[int, int],
    crop_box: tuple[int, int, int, int],
    crop_grid: int,
    crop_index: int,
    full_size: tuple[int, int],
    *,
    config: ProposalFilterConfig,
    postprocess_low_res_masks,
) -> list[MaskProposal]:
    scores_np = _to_numpy(scores)
    low_res_np = _to_numpy(low_res_masks)
    if scores_np.ndim == 1:
        scores_np = scores_np[None]
    if low_res_np.ndim == 3:
        low_res_np = low_res_np[None]

    selected: list[tuple[int, int, float, float]] = []
    for point_index in range(scores_np.shape[0]):
        for mask_index in range(scores_np.shape[1]):
            predicted_iou = float(scores_np[point_index, mask_index])
            if predicted_iou < config.pred_iou_thresh:
                continue
            stability = calculate_stability_score(
                low_res_np[point_index, mask_index],
                offset=config.stability_score_offset,
            )
            if stability < config.stability_score_thresh:
                continue
            selected.append((point_index, mask_index, predicted_iou, stability))
    if not selected:
        return []

    low_res_tensor = torch.as_tensor(low_res_masks)
    survivor_low_res = torch.stack(
        [
            low_res_tensor[point_index, mask_index]
            for point_index, mask_index, _predicted_iou, _stability in selected
        ],
        dim=0,
    )[:, None]
    masks = postprocess_low_res_masks(survivor_low_res, crop_hw)
    if masks.ndim == 3:
        masks = masks[:, None]

    proposals: list[MaskProposal] = []
    for survivor_index, (
        point_index,
        _mask_index,
        predicted_iou,
        stability,
    ) in enumerate(selected):
        mask = masks[survivor_index, 0].astype(bool)
        area = int(mask.sum())
        if area < config.min_mask_region_area:
            continue
        proposal = proposal_from_mask(
            points[point_index],
            mask,
            predicted_iou=predicted_iou,
            stability=stability,
            crop_box=crop_box,
            crop_grid=crop_grid,
            crop_index=crop_index,
            full_size=full_size,
            filter_crop_edge_masks=config.filter_crop_edge_masks,
        )
        if proposal is not None:
            proposals.append(proposal)
    return proposals


def proposal_from_mask(
    point: np.ndarray,
    mask: np.ndarray,
    *,
    predicted_iou: float,
    stability: float,
    crop_box: tuple[int, int, int, int],
    crop_grid: int,
    crop_index: int,
    full_size: tuple[int, int],
    filter_crop_edge_masks: bool,
) -> MaskProposal | None:
    crop_x0, crop_y0, _crop_x1, _crop_y1 = crop_box
    full_width, full_height = full_size
    local_bbox = mask_to_box(mask)
    if local_bbox is None:
        return None
    if filter_crop_edge_masks and touches_internal_crop_edge(
        local_bbox,
        crop_box,
        full_size,
    ):
        return None
    bbox = (
        local_bbox[0] + crop_x0,
        local_bbox[1] + crop_y0,
        local_bbox[2] + crop_x0,
        local_bbox[3] + crop_y0,
    )
    lx0, ly0, lx1, ly1 = local_bbox
    return MaskProposal(
        segmentation=mask[ly0:ly1, lx0:lx1].copy(),
        bbox=bbox,
        area=int(mask.sum()),
        predicted_iou=predicted_iou,
        stability_score=stability,
        point_coords=(
            float(point[0] + crop_x0),
            float(point[1] + crop_y0),
        ),
        crop_box=crop_box,
        crop_grid=crop_grid,
        crop_index=crop_index,
        image_size=(full_width, full_height),
    )


def remove_duplicate_proposals(
    proposals: list[MaskProposal],
    iou_threshold: float,
) -> list[MaskProposal]:
    if not proposals:
        return []
    scores = np.array(
        [
            proposal.predicted_iou + proposal.stability_score * 1e-3
            for proposal in proposals
        ],
        dtype=np.float32,
    )
    boxes = np.array([proposal.bbox for proposal in proposals], dtype=np.float32)
    keep = filter_boxes(boxes, scores, iou_threshold)
    return [proposals[index] for index in keep]


def proposal_to_full_mask(proposal: MaskProposal) -> np.ndarray:
    if proposal.image_size is None:
        raise ValueError("proposal.image_size is required")
    x0, y0, x1, y1 = _validate_roi_geometry(proposal)
    width, height = proposal.image_size
    full_mask = np.zeros((height, width), dtype=bool)
    full_mask[y0:y1, x0:x1] = proposal.segmentation.astype(bool)
    return full_mask


def proposal_mask_image(proposal: MaskProposal, alpha: int = 255) -> Image.Image:
    _validate_roi_geometry(proposal)
    mask = proposal.segmentation.astype(np.uint8) * int(alpha)
    return Image.fromarray(mask, mode="L")


def save_proposal_overlay(
    image: Image.Image,
    proposals: Sequence[MaskProposal],
    path: str | Path,
    max_masks: int = 50,
) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for index, proposal in enumerate(proposals[:max_masks]):
        overlay = _paste_proposal_overlay(
            overlay,
            proposal,
            _proposal_color(index),
            alpha=110,
        )
    Image.alpha_composite(base, overlay).save(path)


def save_proposal_grid(
    image: Image.Image,
    proposals: Sequence[MaskProposal],
    path: str | Path,
    max_masks: int = 24,
    columns: int = 6,
) -> None:
    if columns <= 0:
        raise ValueError("columns must be a positive integer")
    selected = list(proposals[:max_masks])
    if not selected:
        Image.new("RGB", (1, 1), (0, 0, 0)).save(path)
        return

    thumb_width = 160
    thumb_height = int(round(thumb_width * image.height / image.width))
    rows = int(np.ceil(len(selected) / columns))
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows * thumb_height),
        (20, 20, 20),
    )

    for index, proposal in enumerate(selected):
        tile = image.convert("RGBA")
        color = _proposal_color(index)
        tile = _paste_proposal_overlay(tile, proposal, color, alpha=130)
        draw = ImageDraw.Draw(tile)
        draw.rectangle(proposal.bbox, outline=color[:3], width=3)
        tile = tile.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_width
        y = (index // columns) * thumb_height
        sheet.paste(tile.convert("RGB"), (x, y))

    sheet.save(path)


def _validate_roi_geometry(proposal: MaskProposal) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = proposal.bbox
    expected_shape = (y1 - y0, x1 - x0)
    if proposal.segmentation.shape != expected_shape:
        raise ValueError(
            "segmentation shape must match bbox size: "
            f"got {proposal.segmentation.shape}, expected {expected_shape}"
        )
    return x0, y0, x1, y1


def _paste_proposal_overlay(
    overlay: Image.Image,
    proposal: MaskProposal,
    color: tuple[int, int, int, int],
    alpha: int,
) -> Image.Image:
    x0, y0, x1, y1 = _validate_roi_geometry(proposal)
    mask = proposal_mask_image(proposal, alpha=alpha)
    layer = Image.new("RGBA", (x1 - x0, y1 - y0), color)
    layer.putalpha(mask)
    overlay.alpha_composite(layer, dest=(x0, y0))
    return overlay


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _proposal_color(index: int) -> tuple[int, int, int, int]:
    palette = [
        (230, 57, 70, 255),
        (42, 157, 143, 255),
        (69, 123, 157, 255),
        (244, 162, 97, 255),
        (131, 56, 236, 255),
        (255, 190, 11, 255),
        (6, 214, 160, 255),
        (239, 71, 111, 255),
    ]
    return palette[index % len(palette)]
