from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ...types import MaskInstance, MaskProposal, Sam3PromptBatch
from ..prompted import Sam3Predictor
from .batching import decode_prompt_jobs
from .geometry import (
    batched,
    build_point_grid,
    crop_image,
    generate_crop_boxes,
    image_size,
)
from .instances import mask_instances_from_proposals
from .proposals import (
    ProposalFilterConfig,
    proposals_from_batch,
    proposals_from_low_res_batch,
    remove_duplicate_proposals,
)


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
        self._point_label_cache: dict[tuple[str, int], np.ndarray | torch.Tensor] = {}

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

        proposals = remove_duplicate_proposals(proposals, self.box_nms_thresh)
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

    def _point_coords(self, point_batch: np.ndarray) -> np.ndarray | torch.Tensor:
        coords = point_batch[:, None, :].astype(np.float32, copy=False)
        device = getattr(self.predictor, "device", None)
        if device is None:
            return coords
        return torch.as_tensor(coords, dtype=torch.float32, device=torch.device(device))

    def _point_labels(self, count: int) -> np.ndarray | torch.Tensor:
        device = getattr(self.predictor, "device", None)
        key = ("cpu", count) if device is None else (str(torch.device(device)), count)
        cached = self._point_label_cache.get(key)
        if cached is not None:
            return cached
        if device is None:
            labels = np.ones((count, 1), dtype=np.int64)
        else:
            labels = torch.ones(
                (count, 1),
                dtype=torch.int64,
                device=torch.device(device),
            )
        self._point_label_cache[key] = labels
        return labels

    def _proposal_config(self) -> ProposalFilterConfig:
        return ProposalFilterConfig(
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            stability_score_offset=self.stability_score_offset,
            min_mask_region_area=self.min_mask_region_area,
            filter_crop_edge_masks=self.filter_crop_edge_masks,
        )

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
        embedding = self.predictor.encode_image(crop_input)
        return self._generate_for_crop_embedding(
            embedding,
            crop_input,
            crop_box,
            crop_grid,
            crop_index,
            normalized_grid,
            full_size,
        )

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
        config = self._proposal_config()
        for point_batch in batched(pixel_grid, self.points_per_batch):
            point_coords = self._point_coords(point_batch)
            point_labels = self._point_labels(len(point_batch))
            if hasattr(self.predictor, "decode_low_res_from_embedding"):
                low_res_masks, scores = self.predictor.decode_low_res_from_embedding(
                    embedding,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
                proposals.extend(
                    proposals_from_low_res_batch(
                        point_batch,
                        scores,
                        low_res_masks,
                        (crop_height, crop_width),
                        crop_box,
                        crop_grid,
                        crop_index,
                        full_size,
                        config=config,
                        postprocess_low_res_masks=self._postprocess_low_res_masks,
                    )
                )
                continue

            masks, scores, low_res_masks = self.predictor.predict_from_embedding(
                embedding,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            proposals.extend(
                proposals_from_batch(
                    point_batch,
                    masks,
                    scores,
                    low_res_masks,
                    crop_box,
                    crop_grid,
                    crop_index,
                    full_size,
                    config=config,
                )
            )
        proposals = remove_duplicate_proposals(proposals, self.crop_nms_thresh)
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
        config = self._proposal_config()
        for crop_slot, ((crop_index, crop_box, crop_input), embedding) in enumerate(
            zip(crop_batch, embeddings)
        ):
            crop_width, crop_height = image_size(crop_input)
            pixel_grid = normalized_grid.copy()
            pixel_grid[:, 0] *= float(crop_width)
            pixel_grid[:, 1] *= float(crop_height)
            decode_jobs = []
            for point_batch in batched(pixel_grid, self.points_per_batch):
                decode_jobs.append(
                    (
                        crop_slot,
                        crop_index,
                        crop_box,
                        (crop_height, crop_width),
                        point_batch,
                        Sam3PromptBatch(
                            embedding=embedding,
                            point_coords=self._point_coords(point_batch),
                            point_labels=self._point_labels(len(point_batch)),
                        ),
                    )
                )

            if self.allow_cross_crop_prompt_decode:
                all_decode_jobs.extend(decode_jobs)
                continue

            decode_prompt_jobs(
                predictor=self.predictor,
                decode_jobs=decode_jobs,
                crop_proposals=crop_proposals,
                crop_grid=crop_grid,
                full_size=full_size,
                prompt_decode_batch_size=self.prompt_decode_batch_size,
                config=config,
                postprocess_low_res_masks=self._postprocess_low_res_masks,
            )

        if self.allow_cross_crop_prompt_decode:
            decode_prompt_jobs(
                predictor=self.predictor,
                decode_jobs=all_decode_jobs,
                crop_proposals=crop_proposals,
                crop_grid=crop_grid,
                full_size=full_size,
                prompt_decode_batch_size=self.prompt_decode_batch_size,
                config=config,
                postprocess_low_res_masks=self._postprocess_low_res_masks,
            )

        proposals: list[MaskProposal] = []
        for crop_items in crop_proposals:
            crop_items = remove_duplicate_proposals(crop_items, self.crop_nms_thresh)
            if self.max_masks_per_crop is not None:
                crop_items = crop_items[: self.max_masks_per_crop]
            proposals.extend(crop_items)
        return proposals

    def _postprocess_low_res_masks(
        self,
        low_res_masks: torch.Tensor,
        crop_hw: tuple[int, int],
    ) -> np.ndarray:
        if hasattr(self.predictor, "postprocess_low_res_masks"):
            return self.predictor.postprocess_low_res_masks(
                low_res_masks,
                crop_hw,
                return_logits=False,
            )
        return low_res_masks.detach().cpu().numpy() > 0
