from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator, Sequence

import numpy as np
from PIL import Image

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


def build_point_grid(points_per_side: int) -> np.ndarray:
    if points_per_side <= 0:
        raise ValueError("points_per_side must be a positive integer")
    offset = 1.0 / (2 * points_per_side)
    points = np.linspace(offset, 1.0 - offset, points_per_side, dtype=np.float32)
    xv, yv = np.meshgrid(points, points)
    return np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.float32)


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
    ) -> None:
        if points_per_side <= 0:
            raise ValueError("points_per_side must be a positive integer")
        if points_per_batch <= 0:
            raise ValueError("points_per_batch must be a positive integer")
        self.predictor = predictor
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.min_mask_region_area = min_mask_region_area
        self.box_nms_thresh = box_nms_thresh
        self.max_masks = max_masks

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
        crop_box = (0, 0, width, height)
        self.predictor.set_image(image)

        normalized_grid = build_point_grid(self.points_per_side)
        pixel_grid = normalized_grid.copy()
        pixel_grid[:, 0] *= float(width)
        pixel_grid[:, 1] *= float(height)

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

    def _proposals_from_batch(
        self,
        points: np.ndarray,
        masks: np.ndarray,
        scores: np.ndarray,
        low_res_masks: np.ndarray,
        crop_box: tuple[int, int, int, int],
    ) -> list[MaskProposal]:
        proposals: list[MaskProposal] = []
        for point_index, point in enumerate(points):
            for mask_index in range(masks.shape[1]):
                predicted_iou = float(scores[point_index, mask_index])
                if predicted_iou < self.pred_iou_thresh:
                    continue
                mask = masks[point_index, mask_index].astype(bool)
                area = int(mask.sum())
                if area < self.min_mask_region_area:
                    continue
                bbox = mask_to_box(mask)
                if bbox is None:
                    continue
                stability = calculate_stability_score(
                    low_res_masks[point_index, mask_index],
                    offset=self.stability_score_offset,
                )
                if stability < self.stability_score_thresh:
                    continue
                proposals.append(
                    MaskProposal(
                        segmentation=mask,
                        bbox=bbox,
                        area=area,
                        predicted_iou=predicted_iou,
                        stability_score=stability,
                        point_coords=(float(point[0]), float(point[1])),
                        crop_box=crop_box,
                    )
                )
        return proposals

    def _remove_duplicates(self, proposals: list[MaskProposal]) -> list[MaskProposal]:
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
        keep = nms_boxes(boxes, scores, self.box_nms_thresh)
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
