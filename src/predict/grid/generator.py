from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ...ops.box import filter_boxes
from ...types import MaskInstance, MaskProposal, Sam3PromptBatch
from ..prompted import Sam3Predictor
from .geometry import (
    batched,
    build_point_grid,
    calculate_stability_score,
    crop_image,
    generate_crop_boxes,
    image_size,
    mask_to_box,
    touches_internal_crop_edge,
)
from .instances import mask_instances_from_proposals


class AutomaticMaskGenerator:
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
        crop_encode_batch_size: int = 1,
        prompt_decode_batch_size: int = 1,
        image_batch_size: int | None = None,
        prompt_batch_size: int | None = None,
        allow_cross_crop_prompt_decode: bool = False,
    ) -> None:
        if image_batch_size is not None:
            crop_encode_batch_size = image_batch_size
        if prompt_batch_size is not None:
            prompt_decode_batch_size = prompt_batch_size
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
        if crop_encode_batch_size <= 0:
            raise ValueError("crop_encode_batch_size must be a positive integer")
        if prompt_decode_batch_size <= 0:
            raise ValueError("prompt_decode_batch_size must be a positive integer")
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
        self.image_batch_size = crop_encode_batch_size
        self.prompt_batch_size = prompt_decode_batch_size
        self.crop_encode_batch_size = self.image_batch_size
        self.prompt_decode_batch_size = self.prompt_batch_size
        self.allow_cross_crop_prompt_decode = allow_cross_crop_prompt_decode

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cuda",
        **kwargs,
    ) -> "AutomaticMaskGenerator":
        predictor = Sam3Predictor.from_checkpoint(path, device=device)
        return cls(predictor, **kwargs)

    def generate(self, image: Image.Image | np.ndarray) -> list[MaskProposal]:
        width, height = image_size(image)
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
            crop_jobs = [
                (crop_index, crop_box, crop_image(image, crop_box))
                for crop_index, crop_box in enumerate(crop_boxes)
            ]
            proposals.extend(
                self._generate_for_crop_jobs(
                    crop_jobs,
                    crop_grid,
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

    def generate_instances(
        self,
        image: Image.Image | np.ndarray,
        *,
        concept_id: int | None = None,
        object_id_start: int | None = None,
        source: str = "auto",
    ) -> list[MaskInstance]:
        proposals = self.generate(image)
        return mask_instances_from_proposals(
            proposals,
            concept_id=concept_id,
            object_id_start=object_id_start,
            source=source,
        )

    def _crop_grid_config(self) -> list[tuple[int, int]]:
        if self.crop_grids is None:
            return [(1, self.points_per_side)]
        assert self.crop_points_per_side is not None
        return list(zip(self.crop_grids, self.crop_points_per_side))

    def _generate_for_crop_jobs(
        self,
        crop_jobs: list[
            tuple[int, tuple[int, int, int, int], Image.Image | np.ndarray]
        ],
        crop_grid: int,
        normalized_grid: np.ndarray,
        full_size: tuple[int, int],
    ) -> list[MaskProposal]:
        can_batch_encode = (
            self.crop_encode_batch_size > 1
            and hasattr(self.predictor, "encode_image_batch")
            and hasattr(self.predictor, "predict_from_embedding")
        )
        if not can_batch_encode:
            return self._generate_for_crop_jobs_single(
                crop_jobs,
                crop_grid,
                normalized_grid,
                full_size,
            )

        proposals: list[MaskProposal] = []
        for start in range(0, len(crop_jobs), self.crop_encode_batch_size):
            crop_batch = crop_jobs[start : start + self.crop_encode_batch_size]
            crop_images = [job[2] for job in crop_batch]
            try:
                embeddings = self.predictor.encode_image_batch(crop_images)
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                proposals.extend(
                    self._generate_for_crop_jobs_single(
                        crop_batch,
                        crop_grid,
                        normalized_grid,
                        full_size,
                    )
                )
                continue

            can_batch_decode = self.prompt_decode_batch_size > 1 and hasattr(
                self.predictor, "predict_from_embedding_batches"
            )
            if not can_batch_decode:
                for (crop_index, crop_box, crop_input), embedding in zip(
                    crop_batch,
                    embeddings,
                ):
                    proposals.extend(
                        self._generate_for_crop_embedding(
                            embedding,
                            crop_input,
                            crop_box,
                            crop_grid,
                            crop_index,
                            normalized_grid,
                            full_size,
                        )
                    )
                continue

            try:
                proposals.extend(
                    self._generate_for_crop_embeddings_batched(
                        crop_batch,
                        embeddings,
                        crop_grid,
                        normalized_grid,
                        full_size,
                    )
                )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for (crop_index, crop_box, crop_input), embedding in zip(
                    crop_batch,
                    embeddings,
                ):
                    proposals.extend(
                        self._generate_for_crop_embedding(
                            embedding,
                            crop_input,
                            crop_box,
                            crop_grid,
                            crop_index,
                            normalized_grid,
                            full_size,
                        )
                    )
        return proposals

    def _generate_for_crop_jobs_single(
        self,
        crop_jobs: list[
            tuple[int, tuple[int, int, int, int], Image.Image | np.ndarray]
        ],
        crop_grid: int,
        normalized_grid: np.ndarray,
        full_size: tuple[int, int],
    ) -> list[MaskProposal]:
        proposals: list[MaskProposal] = []
        for crop_index, crop_box, crop_input in crop_jobs:
            proposals.extend(
                self._generate_for_crop(
                    crop_input,
                    crop_box,
                    crop_grid,
                    crop_index,
                    normalized_grid,
                    full_size,
                )
            )
        return proposals

    def _generate_for_crop(
        self,
        crop_input: Image.Image | np.ndarray,
        crop_box: tuple[int, int, int, int],
        crop_grid: int,
        crop_index: int,
        normalized_grid: np.ndarray,
        full_size: tuple[int, int],
    ) -> list[MaskProposal]:
        crop_width, crop_height = image_size(crop_input)
        self.predictor.set_image(crop_input)
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

    def _generate_for_crop_embedding(
        self,
        embedding,
        crop_input: Image.Image | np.ndarray,
        crop_box: tuple[int, int, int, int],
        crop_grid: int,
        crop_index: int,
        normalized_grid: np.ndarray,
        full_size: tuple[int, int],
    ) -> list[MaskProposal]:
        crop_width, crop_height = image_size(crop_input)
        pixel_grid = normalized_grid.copy()
        pixel_grid[:, 0] *= float(crop_width)
        pixel_grid[:, 1] *= float(crop_height)

        proposals: list[MaskProposal] = []
        for point_batch in batched(pixel_grid, self.points_per_batch):
            point_labels = np.ones((len(point_batch), 1), dtype=np.int64)
            masks, scores, low_res_masks = self.predictor.predict_from_embedding(
                embedding,
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

    def _generate_for_crop_embeddings_batched(
        self,
        crop_batch: list[
            tuple[int, tuple[int, int, int, int], Image.Image | np.ndarray]
        ],
        embeddings,
        crop_grid: int,
        normalized_grid: np.ndarray,
        full_size: tuple[int, int],
    ) -> list[MaskProposal]:
        crop_proposals: list[list[MaskProposal]] = [[] for _ in crop_batch]
        all_decode_jobs = []
        for crop_slot, ((crop_index, crop_box, crop_input), embedding) in enumerate(
            zip(crop_batch, embeddings)
        ):
            crop_width, crop_height = image_size(crop_input)
            pixel_grid = normalized_grid.copy()
            pixel_grid[:, 0] *= float(crop_width)
            pixel_grid[:, 1] *= float(crop_height)
            decode_jobs = []
            for point_batch in batched(pixel_grid, self.points_per_batch):
                point_labels = np.ones((len(point_batch), 1), dtype=np.int64)
                decode_jobs.append(
                    (
                        crop_slot,
                        crop_index,
                        crop_box,
                        point_batch,
                        Sam3PromptBatch(
                            embedding=embedding,
                            point_coords=point_batch[:, None, :].astype(np.float32),
                            point_labels=point_labels,
                        ),
                    )
                )

            if self.allow_cross_crop_prompt_decode:
                all_decode_jobs.extend(decode_jobs)
                continue

            self._decode_prompt_jobs(
                decode_jobs,
                crop_proposals,
                crop_grid,
                full_size,
            )

        if self.allow_cross_crop_prompt_decode:
            self._decode_prompt_jobs(
                all_decode_jobs,
                crop_proposals,
                crop_grid,
                full_size,
            )

        proposals: list[MaskProposal] = []
        for crop_items in crop_proposals:
            crop_items = self._remove_duplicates(crop_items, self.crop_nms_thresh)
            if self.max_masks_per_crop is not None:
                crop_items = crop_items[: self.max_masks_per_crop]
            proposals.extend(crop_items)
        return proposals

    def _decode_prompt_jobs(
        self,
        decode_jobs,
        crop_proposals: list[list[MaskProposal]],
        crop_grid: int,
        full_size: tuple[int, int],
    ) -> None:
        if not decode_jobs:
            return
        for start in range(0, len(decode_jobs), self.prompt_decode_batch_size):
            job_batch = decode_jobs[start : start + self.prompt_decode_batch_size]
            prompt_batches = [job[-1] for job in job_batch]
            results = self.predictor.predict_from_embedding_batches(
                prompt_batches,
                multimask_output=True,
            )
            for (
                crop_slot,
                crop_index,
                crop_box,
                point_batch,
                _prompt_batch,
            ), (
                masks,
                scores,
                low_res_masks,
            ) in zip(job_batch, results):
                crop_proposals[crop_slot].extend(
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
        crop_x0, crop_y0, _crop_x1, _crop_y1 = crop_box
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
                if self.filter_crop_edge_masks and touches_internal_crop_edge(
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
                lx0, ly0, lx1, ly1 = local_bbox
                roi_mask = mask[ly0:ly1, lx0:lx1].copy()
                proposals.append(
                    MaskProposal(
                        segmentation=roi_mask,
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
                        image_size=(full_width, full_height),
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
        keep = filter_boxes(boxes, scores, threshold)
        return [proposals[index] for index in keep]
