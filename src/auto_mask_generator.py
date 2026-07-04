from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from collections.abc import Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .predictor import Sam3Predictor


@dataclass(frozen=True)
class MaskProposal:
    segmentation: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    predicted_iou: float
    stability_score: float
    point_coords: tuple[float, float]
    crop_box: tuple[int, int, int, int]
    crop_grid: int = 1
    crop_index: int = 0


def build_point_grid(points_per_side: int) -> np.ndarray:
    if points_per_side <= 0:
        raise ValueError("points_per_side must be a positive integer")
    offset = 1.0 / (2 * points_per_side)
    points = np.linspace(offset, 1.0 - offset, points_per_side, dtype=np.float32)
    xv, yv = np.meshgrid(points, points)
    return np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.float32)


def generate_crop_boxes(
    width: int,
    height: int,
    grid_size: int,
    overlap_ratio: float,
) -> list[tuple[int, int, int, int]]:
    if grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    if overlap_ratio < 0.0 or overlap_ratio >= 0.5:
        raise ValueError("overlap_ratio must be in [0.0, 0.5)")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if grid_size == 1:
        return [(0, 0, width, height)]

    overlap_w = int(round((width / grid_size) * overlap_ratio))
    overlap_h = int(round((height / grid_size) * overlap_ratio))
    crop_w = int(ceil((width + overlap_w * (grid_size - 1)) / grid_size))
    crop_h = int(ceil((height + overlap_h * (grid_size - 1)) / grid_size))
    stride_w = crop_w - overlap_w
    stride_h = crop_h - overlap_h

    boxes: list[tuple[int, int, int, int]] = []
    for iy in range(grid_size):
        y0 = min(iy * stride_h, height - crop_h)
        y1 = min(y0 + crop_h, height)
        for ix in range(grid_size):
            x0 = min(ix * stride_w, width - crop_w)
            x1 = min(x0 + crop_w, width)
            boxes.append((int(x0), int(y0), int(x1), int(y1)))
    return boxes


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


class Sam3AutomaticMaskGenerator:
    def __init__(
        self,
        predictor,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.0,
        stability_score_thresh: float = 0.75,
        stability_score_offset: float = 1.0,
        min_mask_region_area: int = 0,
        box_nms_thresh: float = 0.7,
        max_masks: int | None = None,
        crop_grids: Sequence[int] | None = None,
        crop_points_per_side: Sequence[int] | None = None,
        crop_overlap_ratio: float = 0.25,
        crop_nms_thresh: float | None = None,
        max_masks_per_crop: int | None = None,
        filter_crop_edge_masks: bool = True,
    ) -> None:
        if points_per_side <= 0:
            raise ValueError("points_per_side must be a positive integer")
        if points_per_batch <= 0:
            raise ValueError("points_per_batch must be a positive integer")
        if crop_grids is None and crop_points_per_side is not None:
            raise ValueError("crop_points_per_side requires crop_grids")
        if crop_grids is not None:
            if crop_points_per_side is None or len(crop_grids) != len(
                crop_points_per_side
            ):
                raise ValueError(
                    "crop_grids and crop_points_per_side must have the same length"
                )
            if any(grid <= 0 for grid in crop_grids):
                raise ValueError("crop_grids entries must be positive integers")
            if any(points <= 0 for points in crop_points_per_side):
                raise ValueError(
                    "crop_points_per_side entries must be positive integers"
                )
        if crop_overlap_ratio < 0.0 or crop_overlap_ratio >= 0.5:
            raise ValueError("crop_overlap_ratio must be in [0.0, 0.5)")
        self.predictor = predictor
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.min_mask_region_area = min_mask_region_area
        self.box_nms_thresh = box_nms_thresh
        self.max_masks = max_masks
        self.crop_grids = tuple(crop_grids) if crop_grids is not None else None
        self.crop_points_per_side = (
            tuple(crop_points_per_side) if crop_points_per_side is not None else None
        )
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_nms_thresh = (
            box_nms_thresh if crop_nms_thresh is None else crop_nms_thresh
        )
        self.max_masks_per_crop = max_masks_per_crop
        self.filter_crop_edge_masks = filter_crop_edge_masks

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cuda",
        **kwargs,
    ) -> "Sam3AutomaticMaskGenerator":
        predictor = Sam3Predictor.from_checkpoint(checkpoint_path, device=device)
        return cls(predictor, **kwargs)

    def generate(self, image: Image.Image | np.ndarray) -> list[MaskProposal]:
        width, height = _image_size(image)
        point_grid_cache: dict[int, np.ndarray] = {}
        proposals: list[MaskProposal] = []
        for crop_grid, points_per_side in self._crop_grid_config():
            normalized_grid = point_grid_cache.setdefault(
                points_per_side,
                build_point_grid(points_per_side),
            )
            crop_boxes = generate_crop_boxes(
                width,
                height,
                crop_grid,
                self.crop_overlap_ratio,
            )
            for crop_index, crop_box in enumerate(crop_boxes):
                crop_image = _crop_image(image, crop_box)
                proposals.extend(
                    self._generate_for_crop(
                        crop_image,
                        crop_box,
                        crop_grid,
                        crop_index,
                        normalized_grid,
                        (width, height),
                    )
                )

        proposals = self._remove_duplicates(proposals)
        proposals.sort(
            key=lambda proposal: (
                proposal.predicted_iou,
                proposal.stability_score,
                proposal.area,
            ),
            reverse=True,
        )
        if self.max_masks is not None:
            proposals = proposals[: self.max_masks]
        return proposals

    def _crop_grid_config(self) -> list[tuple[int, int]]:
        if self.crop_grids is None:
            return [(1, self.points_per_side)]
        assert self.crop_points_per_side is not None
        return list(zip(self.crop_grids, self.crop_points_per_side))

    def _generate_for_crop(
        self,
        crop_image: Image.Image | np.ndarray,
        crop_box: tuple[int, int, int, int],
        crop_grid: int,
        crop_index: int,
        normalized_grid: np.ndarray,
        full_size: tuple[int, int],
    ) -> list[MaskProposal]:
        crop_width, crop_height = _image_size(crop_image)
        self.predictor.set_image(crop_image)
        pixel_grid = normalized_grid.copy()
        pixel_grid[:, 0] *= float(crop_width)
        pixel_grid[:, 1] *= float(crop_height)

        proposals: list[MaskProposal] = []
        for point_batch in batched(pixel_grid, self.points_per_batch):
            point_labels = np.ones((len(point_batch), 1), dtype=np.int64)
            masks, scores, low_res_masks = self.predictor.predict(
                point_coords=point_batch[:, None, :].astype(np.float32),
                point_labels=point_labels,
                multimask_output=True,
            )
            proposals.extend(
                self._proposals_from_batch(
                    point_batch,
                    masks,
                    scores,
                    low_res_masks,
                    crop_box,
                    crop_grid,
                    crop_index,
                    full_size,
                )
            )
        proposals = self._remove_duplicates(proposals, self.crop_nms_thresh)
        if self.max_masks_per_crop is not None:
            proposals = proposals[: self.max_masks_per_crop]
        return proposals

    def _proposals_from_batch(
        self,
        points: np.ndarray,
        masks: np.ndarray,
        scores: np.ndarray,
        low_res_masks: np.ndarray,
        crop_box: tuple[int, int, int, int],
        crop_grid: int,
        crop_index: int,
        full_size: tuple[int, int],
    ) -> list[MaskProposal]:
        proposals: list[MaskProposal] = []
        crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
        full_width, full_height = full_size
        for point_index, point in enumerate(points):
            for mask_index in range(masks.shape[1]):
                predicted_iou = float(scores[point_index, mask_index])
                if predicted_iou < self.pred_iou_thresh:
                    continue
                mask = masks[point_index, mask_index].astype(bool)
                area = int(mask.sum())
                if area < self.min_mask_region_area:
                    continue
                local_bbox = mask_to_box(mask)
                if local_bbox is None:
                    continue
                if self.filter_crop_edge_masks and _touches_internal_crop_edge(
                    local_bbox,
                    crop_box,
                    full_size,
                ):
                    continue
                stability = calculate_stability_score(
                    low_res_masks[point_index, mask_index],
                    offset=self.stability_score_offset,
                )
                if stability < self.stability_score_thresh:
                    continue
                bbox = (
                    local_bbox[0] + crop_x0,
                    local_bbox[1] + crop_y0,
                    local_bbox[2] + crop_x0,
                    local_bbox[3] + crop_y0,
                )
                full_mask = np.zeros((full_height, full_width), dtype=bool)
                full_mask[crop_y0:crop_y1, crop_x0:crop_x1] = mask
                proposals.append(
                    MaskProposal(
                        segmentation=full_mask,
                        bbox=bbox,
                        area=area,
                        predicted_iou=predicted_iou,
                        stability_score=stability,
                        point_coords=(
                            float(point[0] + crop_x0),
                            float(point[1] + crop_y0),
                        ),
                        crop_box=crop_box,
                        crop_grid=crop_grid,
                        crop_index=crop_index,
                    )
                )
        return proposals

    def _remove_duplicates(
        self,
        proposals: list[MaskProposal],
        iou_threshold: float | None = None,
    ) -> list[MaskProposal]:
        if not proposals:
            return []
        threshold = self.box_nms_thresh if iou_threshold is None else iou_threshold
        scores = np.array(
            [
                proposal.predicted_iou + proposal.stability_score * 1e-3
                for proposal in proposals
            ],
            dtype=np.float32,
        )
        boxes = np.array([proposal.bbox for proposal in proposals], dtype=np.float32)
        keep = nms_boxes(boxes, scores, threshold)
        return [proposals[index] for index in keep]


def _image_size(image: Image.Image | np.ndarray) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return image.size
    if isinstance(image, np.ndarray):
        if image.ndim != 3:
            raise ValueError("NumPy images must have shape HxWxC")
        height, width = image.shape[:2]
        return width, height
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _crop_image(
    image: Image.Image | np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> Image.Image | np.ndarray:
    x0, y0, x1, y1 = crop_box
    if isinstance(image, Image.Image):
        return image.crop(crop_box)
    return image[y0:y1, x0:x1, :]


def _touches_internal_crop_edge(
    local_bbox: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    full_size: tuple[int, int],
) -> bool:
    x0, y0, x1, y1 = local_bbox
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
    full_width, full_height = full_size
    touches_left = x0 <= 0 and crop_x0 > 0
    touches_top = y0 <= 0 and crop_y0 > 0
    touches_right = x1 >= crop_x1 - crop_x0 and crop_x1 < full_width
    touches_bottom = y1 >= crop_y1 - crop_y0 and crop_y1 < full_height
    return touches_left or touches_top or touches_right or touches_bottom


def count_proposals_by_crop_grid(
    proposals: Sequence[MaskProposal],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for proposal in proposals:
        counts[proposal.crop_grid] = counts.get(proposal.crop_grid, 0) + 1
    return dict(sorted(counts.items()))


def save_proposal_overlay(
    image: Image.Image,
    proposals: Sequence[MaskProposal],
    path: str | Path,
    max_masks: int = 50,
) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for index, proposal in enumerate(proposals[:max_masks]):
        color = _proposal_color(index)
        mask = Image.fromarray((proposal.segmentation.astype(np.uint8) * 110), mode="L")
        layer = Image.new("RGBA", base.size, color)
        layer.putalpha(mask)
        overlay = Image.alpha_composite(overlay, layer)
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
        mask = Image.fromarray((proposal.segmentation.astype(np.uint8) * 130), mode="L")
        layer = Image.new("RGBA", tile.size, color)
        layer.putalpha(mask)
        tile = Image.alpha_composite(tile, layer)
        draw = ImageDraw.Draw(tile)
        draw.rectangle(proposal.bbox, outline=color[:3], width=3)
        tile = tile.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_width
        y = (index // columns) * thumb_height
        sheet.paste(tile.convert("RGB"), (x, y))

    sheet.save(path)


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
