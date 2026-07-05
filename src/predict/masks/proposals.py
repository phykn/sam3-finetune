from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .types import MaskProposal


def count_proposals_by_crop_grid(
    proposals: Sequence[MaskProposal],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for proposal in proposals:
        counts[proposal.crop_grid] = counts.get(proposal.crop_grid, 0) + 1
    return dict(sorted(counts.items()))


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
