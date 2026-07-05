from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ...types import MaskInstance
from ..grid import AutomaticMaskGenerator
from ..grid.geometry import batched, calculate_stability_score, mask_to_box
from ..prompted import Sam3Predictor
from .masks import select_best_mask


@dataclass(frozen=True)
class GridRefineResult:
    base_instances: list[MaskInstance]
    refined_instances: list[MaskInstance]


class GridMaskRefiner:
    def __init__(
        self,
        *,
        predictor,
        base_generator,
        batch_size: int = 8,
        multimask_output: bool = False,
        mask_foreground: float = 4.0,
        mask_background: float = -4.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.predictor = predictor
        self.base_generator = base_generator
        self.batch_size = int(batch_size)
        self.multimask_output = bool(multimask_output)
        self.mask_foreground = float(mask_foreground)
        self.mask_background = float(mask_background)

    @classmethod
    def from_predictor(
        cls,
        predictor,
        *,
        grid_kwargs: dict | None = None,
        batch_size: int = 8,
        multimask_output: bool = False,
        mask_foreground: float = 4.0,
        mask_background: float = -4.0,
    ) -> "GridMaskRefiner":
        base_generator = AutomaticMaskGenerator(
            predictor,
            **(grid_kwargs or {}),
        )
        return cls(
            predictor=predictor,
            base_generator=base_generator,
            batch_size=batch_size,
            multimask_output=multimask_output,
            mask_foreground=mask_foreground,
            mask_background=mask_background,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cuda",
        *,
        grid_kwargs: dict | None = None,
        batch_size: int = 8,
        multimask_output: bool = False,
        mask_foreground: float = 4.0,
        mask_background: float = -4.0,
    ) -> "GridMaskRefiner":
        predictor = Sam3Predictor.from_checkpoint(path, device=device)
        return cls.from_predictor(
            predictor,
            grid_kwargs=grid_kwargs,
            batch_size=batch_size,
            multimask_output=multimask_output,
            mask_foreground=mask_foreground,
            mask_background=mask_background,
        )

    @torch.inference_mode()
    def refine(self, image) -> GridRefineResult:
        base_instances = self.base_generator.generate_instances(image)
        if not base_instances:
            return GridRefineResult(base_instances=[], refined_instances=[])

        embedding = self.predictor.encode_image(image)
        refined_instances: list[MaskInstance] = []
        for instance_batch in batched(
            np.asarray(base_instances, dtype=object), self.batch_size
        ):
            batch = list(instance_batch)
            mask_input = np.stack(
                [
                    np.where(
                        instance.to_full_mask(),
                        self.mask_foreground,
                        self.mask_background,
                    ).astype(np.float32)
                    for instance in batch
                ],
                axis=0,
            )
            masks, scores, low_res_masks = self.predictor.predict_from_embedding(
                embedding,
                mask_input=mask_input,
                multimask_output=self.multimask_output,
            )
            masks, scores, low_res_masks = _ensure_batch_shapes(
                masks,
                scores,
                low_res_masks,
                batch_size=len(batch),
            )
            refined_instances.extend(
                _instances_from_refined_batch(
                    batch,
                    masks,
                    scores,
                    low_res_masks,
                )
            )

        return GridRefineResult(
            base_instances=base_instances,
            refined_instances=refined_instances,
        )


def _ensure_batch_shapes(
    masks: np.ndarray,
    scores: np.ndarray,
    low_res_masks: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = np.asarray(masks)
    scores = np.asarray(scores)
    low_res_masks = np.asarray(low_res_masks)
    if batch_size == 1:
        if masks.ndim == 3:
            masks = masks[None]
        if scores.ndim == 1:
            scores = scores[None]
        if low_res_masks.ndim == 3:
            low_res_masks = low_res_masks[None]
    if masks.ndim != 4:
        raise ValueError("refined masks must have shape BxMxHxW")
    if scores.ndim != 2:
        raise ValueError("refined scores must have shape BxM")
    if low_res_masks.ndim != 4:
        raise ValueError("refined low_res_masks must have shape BxMxHxW")
    return masks, scores, low_res_masks


def _instances_from_refined_batch(
    base_instances: list[MaskInstance],
    masks: np.ndarray,
    scores: np.ndarray,
    low_res_masks: np.ndarray,
) -> list[MaskInstance]:
    refined_instances = []
    for batch_index, base_instance in enumerate(base_instances):
        mask, score, selected_index = select_best_mask(
            masks[batch_index],
            scores[batch_index],
        )
        bbox = mask_to_box(mask)
        if bbox is None:
            continue
        flat_low_res = np.asarray(low_res_masks[batch_index]).reshape(
            -1,
            *np.asarray(low_res_masks[batch_index]).shape[-2:],
        )
        stability_score = calculate_stability_score(flat_low_res[selected_index])
        x0, y0, x1, y1 = bbox
        roi_mask = mask[y0:y1, x0:x1].copy()
        refined_instances.append(
            MaskInstance(
                segmentation=roi_mask,
                bbox=bbox,
                area=int(mask.sum()),
                score=score,
                source="grid_refined",
                concept_id=base_instance.concept_id,
                object_id=base_instance.object_id,
                base_score=base_instance.score,
                predicted_iou=score,
                stability_score=stability_score,
                point_coords=base_instance.point_coords,
                crop_box=base_instance.crop_box,
                crop_grid=base_instance.crop_grid,
                crop_index=base_instance.crop_index,
                image_size=base_instance.image_size,
            )
        )
    return refined_instances
