from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...types import ContextPrediction, ContextReference, Sam3ImageEmbedding
from ..grid.geometry import calculate_stability_score, mask_to_box
from ..prompted import Sam3Predictor
from .postprocess import nms_context_predictions
from .prototype import (
    build_context_prototype,
    ContextPrototype,
    mean_feature_score_over_mask,
    mean_score_over_mask,
    resize_similarity_map,
    select_feature,
    similarity_map,
)
from .scoring import area_ratio_score


@dataclass(frozen=True)
class ReferenceShapePrior:
    roi_mask: np.ndarray
    width_ratio: float
    height_ratio: float


@dataclass(frozen=True)
class PreparedContextReferences:
    references: tuple[ContextReference, ...]
    embeddings: tuple[Sam3ImageEmbedding, ...]
    prototype: ContextPrototype
    shape_prior: ReferenceShapePrior | None


class ContextMatcher:
    def __init__(
        self,
        predictor: Sam3Predictor,
        *,
        feature_layer: str | int = "image_embed",
        candidate_count: int = 64,
        decode_batch_size: int = 16,
        max_masks: int | None = None,
        min_cell_distance: float = 2.0,
        mask_nms_thresh: float = 0.7,
        candidate_score_mode: str = "point",
        context_score_weight: float = 1.0,
        predicted_iou_weight: float = 0.1,
        stability_score_weight: float = 0.05,
        area_score_weight: float = 0.0,
        negative_context_mode: str = "local",
        negative_context_weight: float = 0.75,
        negative_context_scale: float = 2.0,
        use_reference_mask_prior: bool = False,
        mask_prior_scale: float = 1.0,
        mask_prior_foreground: float = 4.0,
        mask_prior_background: float = -4.0,
        min_context_score: float | None = None,
        min_mask_area: int = 1,
        score_resolution: str = "full",
    ) -> None:
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if decode_batch_size <= 0:
            raise ValueError("decode_batch_size must be positive")
        if max_masks is not None and max_masks <= 0:
            raise ValueError("max_masks must be positive")
        if min_cell_distance < 0:
            raise ValueError("min_cell_distance must be non-negative")
        if mask_nms_thresh < 0:
            raise ValueError("mask_nms_thresh must be non-negative")
        if candidate_score_mode not in {"point", "shape"}:
            raise ValueError("candidate_score_mode must be 'point' or 'shape'")
        if negative_context_mode not in {"none", "image", "local"}:
            raise ValueError(
                "negative_context_mode must be 'none', 'image', or 'local'"
            )
        if negative_context_weight < 0:
            raise ValueError("negative_context_weight must be non-negative")
        if negative_context_scale <= 1.0:
            raise ValueError("negative_context_scale must be greater than 1.0")
        if score_resolution not in {"full", "feature"}:
            raise ValueError("score_resolution must be 'full' or 'feature'")
        self.predictor = predictor
        self.feature_layer = feature_layer
        self.candidate_count = int(candidate_count)
        self.decode_batch_size = int(decode_batch_size)
        self.max_masks = None if max_masks is None else int(max_masks)
        self.min_cell_distance = float(min_cell_distance)
        self.mask_nms_thresh = float(mask_nms_thresh)
        self.candidate_score_mode = candidate_score_mode
        self.context_score_weight = float(context_score_weight)
        self.predicted_iou_weight = float(predicted_iou_weight)
        self.stability_score_weight = float(stability_score_weight)
        self.area_score_weight = float(area_score_weight)
        self.negative_context_mode = negative_context_mode
        self.negative_context_weight = float(negative_context_weight)
        self.negative_context_scale = float(negative_context_scale)
        self.use_reference_mask_prior = bool(use_reference_mask_prior)
        self.mask_prior_scale = float(mask_prior_scale)
        self.mask_prior_foreground = float(mask_prior_foreground)
        self.mask_prior_background = float(mask_prior_background)
        self.min_context_score = min_context_score
        self.min_mask_area = int(min_mask_area)
        self.score_resolution = score_resolution

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cuda",
        **kwargs,
    ) -> "ContextMatcher":
        return cls(
            predictor=Sam3Predictor.from_checkpoint(path, device=device),
            **kwargs,
        )

    @torch.inference_mode()
    def prepare_references(
        self,
        references: Sequence[ContextReference],
    ) -> PreparedContextReferences:
        if not references:
            raise ValueError("references must be non-empty")
        reference_embeddings = tuple(
            self.predictor.encode_image_batch(
                [reference.image for reference in references]
            )
        )
        return self._prepare_from_embeddings(references, reference_embeddings)

    def _prepare_from_embeddings(
        self,
        references: Sequence[ContextReference],
        reference_embeddings: Sequence[Sam3ImageEmbedding],
    ) -> PreparedContextReferences:
        prototype = build_context_prototype(
            references,
            reference_embeddings,
            feature_layer=self.feature_layer,
            negative_context_mode=self.negative_context_mode,
            negative_context_scale=self.negative_context_scale,
        )
        shape_prior = (
            _reference_shape_prior(references, reference_embeddings)
            if self.use_reference_mask_prior or self.candidate_score_mode == "shape"
            else None
        )
        return PreparedContextReferences(
            references=tuple(references),
            embeddings=tuple(reference_embeddings),
            prototype=prototype,
            shape_prior=shape_prior,
        )

    @torch.inference_mode()
    def predict(
        self,
        target_image: Image.Image | np.ndarray,
        references: Sequence[ContextReference] | PreparedContextReferences,
        *,
        max_masks: int | None = None,
        target_point_coords: np.ndarray | torch.Tensor | None = None,
    ) -> list[ContextPrediction]:
        if not references:
            raise ValueError("references must be non-empty")
        if max_masks is not None and max_masks <= 0:
            raise ValueError("max_masks must be positive")

        if isinstance(references, PreparedContextReferences):
            prepared = references
            target_embedding = self.predictor.encode_image_batch([target_image])[0]
        else:
            image_batch = [reference.image for reference in references] + [target_image]
            embeddings = self.predictor.encode_image_batch(image_batch)
            prepared = self._prepare_from_embeddings(references, embeddings[:-1])
            target_embedding = embeddings[-1]

        target_features = select_feature(target_embedding, self.feature_layer)
        similarity = similarity_map(
            target_features,
            prepared.prototype,
            negative_context_weight=self.negative_context_weight,
        )
        if target_point_coords is None:
            candidate_score_map = self._candidate_score_map(
                similarity,
                shape_prior=prepared.shape_prior,
            )
            point_coords = self._candidate_points(
                candidate_score_map,
                target_embedding.orig_hw,
            )
        else:
            point_coords = _target_points_tensor(
                target_point_coords,
                target_embedding.orig_hw,
                device=similarity.device,
            )
        if point_coords.shape[0] == 0:
            return []

        decode_shape_prior = (
            None if target_point_coords is not None else prepared.shape_prior
        )
        predictions = self._decode_candidates(
            target_embedding,
            point_coords,
            similarity,
            reference_area_ratio=prepared.prototype.reference_area_ratio,
            shape_prior=decode_shape_prior,
            max_masks=self.max_masks if max_masks is None else int(max_masks),
        )
        return predictions

    def _candidate_score_map(
        self,
        similarity_map: torch.Tensor,
        *,
        shape_prior: ReferenceShapePrior | None,
    ) -> torch.Tensor:
        if self.candidate_score_mode == "point":
            return similarity_map
        if shape_prior is None:
            return similarity_map
        return _shape_anchor_score_map(
            similarity_map,
            shape_prior,
        )

    def _candidate_points(
        self,
        similarity_map: torch.Tensor,
        orig_hw: tuple[int, int],
    ) -> torch.Tensor:
        height, width = similarity_map.shape
        flat = similarity_map.flatten()
        topk = min(flat.numel(), self.candidate_count * 8)
        values, indices = torch.topk(flat, k=topk)
        if self.min_context_score is not None:
            keep = values >= float(self.min_context_score)
            indices = indices[keep]

        selected: list[torch.Tensor] = []
        min_distance_sq = self.min_cell_distance * self.min_cell_distance
        for index in indices:
            y = torch.div(index, width, rounding_mode="floor")
            x = index % width
            cell = torch.stack([x, y]).to(dtype=torch.float32)
            if selected:
                selected_cells = torch.stack(selected, dim=0)
                distances = ((selected_cells - cell) ** 2).sum(dim=1)
                keep = bool(torch.all(distances >= min_distance_sq).detach().cpu())
            else:
                keep = True
            if keep:
                selected.append(cell)
            if len(selected) >= self.candidate_count:
                break
        if not selected:
            return torch.empty(
                (0, 2),
                dtype=torch.float32,
                device=similarity_map.device,
            )

        orig_h, orig_w = orig_hw
        cells = torch.stack(selected, dim=0).to(device=similarity_map.device)
        points = torch.empty_like(cells)
        points[:, 0] = (cells[:, 0] + 0.5) * float(orig_w) / float(width)
        points[:, 1] = (cells[:, 1] + 0.5) * float(orig_h) / float(height)
        return points

    def _decode_candidates(
        self,
        embedding: Sam3ImageEmbedding,
        point_coords: torch.Tensor,
        similarity_map: torch.Tensor,
        *,
        reference_area_ratio: float,
        shape_prior: ReferenceShapePrior | None,
        max_masks: int | None,
    ) -> list[ContextPrediction]:
        labels = torch.ones(
            (point_coords.shape[0], 1),
            dtype=torch.int64,
            device=point_coords.device,
        )
        if self.score_resolution == "full":
            similarity_scores = resize_similarity_map(similarity_map, embedding.orig_hw)
        else:
            similarity_scores = similarity_map
        candidates: list[ContextPrediction] = []
        for start in range(0, point_coords.shape[0], self.decode_batch_size):
            end = min(start + self.decode_batch_size, point_coords.shape[0])
            point_batch = point_coords[start:end]
            mask_input = (
                _make_mask_prior_batch(
                    shape_prior,
                    point_batch,
                    target_hw=embedding.orig_hw,
                    scale=self.mask_prior_scale,
                    foreground=self.mask_prior_foreground,
                    background=self.mask_prior_background,
                )
                if shape_prior is not None
                else None
            )
            masks, scores, low_res_masks = self.predictor.predict_from_embedding(
                embedding,
                point_coords=point_batch[:, None, :],
                point_labels=labels[start:end],
                mask_input=mask_input,
                multimask_output=True,
                return_logits=True,
            )
            masks, scores, low_res_masks = _ensure_decode_batch_shapes(
                masks,
                scores,
                low_res_masks,
                batch_size=len(point_batch),
            )
            candidates.extend(
                self._predictions_from_decode_batch(
                    point_batch,
                    masks,
                    scores,
                    low_res_masks,
                    similarity_scores,
                    reference_area_ratio=reference_area_ratio,
                    image_size=(embedding.orig_hw[1], embedding.orig_hw[0]),
                    orig_hw=embedding.orig_hw,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return nms_context_predictions(candidates, self.mask_nms_thresh, max_masks)

    def _predictions_from_decode_batch(
        self,
        point_batch: torch.Tensor,
        masks: np.ndarray,
        scores: np.ndarray,
        low_res_masks: np.ndarray,
        similarity_scores: np.ndarray | torch.Tensor,
        reference_area_ratio: float,
        image_size: tuple[int, int],
        orig_hw: tuple[int, int],
    ) -> list[ContextPrediction]:
        predictions: list[ContextPrediction] = []
        for point_index, point in enumerate(point_batch):
            for mask_index in range(masks.shape[1]):
                mask = masks[point_index, mask_index] > 0
                area = int(mask.sum())
                if area < self.min_mask_area:
                    continue
                bbox = mask_to_box(mask)
                if bbox is None:
                    continue
                if isinstance(similarity_scores, torch.Tensor):
                    context_score = mean_feature_score_over_mask(
                        similarity_scores,
                        mask,
                        orig_hw,
                    )
                else:
                    context_score = mean_score_over_mask(similarity_scores, mask)
                if self.min_context_score is not None and context_score < float(
                    self.min_context_score
                ):
                    continue
                stability = calculate_stability_score(
                    low_res_masks[point_index, mask_index]
                )
                predicted_iou = float(scores[point_index, mask_index])
                area_score = area_ratio_score(
                    candidate_ratio=area / float(image_size[0] * image_size[1]),
                    reference_ratio=reference_area_ratio,
                )
                combined_score = (
                    context_score * self.context_score_weight
                    + predicted_iou * self.predicted_iou_weight
                    + stability * self.stability_score_weight
                    + area_score * self.area_score_weight
                )
                x0, y0, x1, y1 = bbox
                predictions.append(
                    ContextPrediction(
                        segmentation=mask[y0:y1, x0:x1].copy(),
                        bbox=bbox,
                        area=area,
                        point_coords=(
                            float(point[0].detach().cpu()),
                            float(point[1].detach().cpu()),
                        ),
                        context_score=float(context_score),
                        predicted_iou=predicted_iou,
                        stability_score=float(stability),
                        score=float(combined_score),
                        image_size=image_size,
                        area_score=float(area_score),
                    )
                )
        return predictions


def _target_points_tensor(
    point_coords: np.ndarray | torch.Tensor,
    orig_hw: tuple[int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    points = torch.as_tensor(point_coords, dtype=torch.float32, device=device).clone()
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("target_point_coords must have shape Nx2")
    height, width = orig_hw
    if points.numel() == 0:
        return points.reshape(0, 2)
    if (
        torch.any(points[:, 0] < 0)
        or torch.any(points[:, 0] >= width)
        or torch.any(points[:, 1] < 0)
        or torch.any(points[:, 1] >= height)
    ):
        raise ValueError("target_point_coords must be within the target image")
    return points


def _shape_anchor_score_map(
    similarity_map: torch.Tensor,
    shape_prior: ReferenceShapePrior,
) -> torch.Tensor:
    if similarity_map.ndim != 2:
        raise ValueError("similarity_map must have shape HxW")
    feature_h, feature_w = similarity_map.shape
    kernel_w = max(1, int(round(shape_prior.width_ratio * feature_w)))
    kernel_h = max(1, int(round(shape_prior.height_ratio * feature_h)))
    kernel = Image.fromarray(shape_prior.roi_mask.astype(np.uint8) * 255).resize(
        (kernel_w, kernel_h),
        resample=Image.Resampling.BILINEAR,
    )
    weights = torch.as_tensor(
        np.asarray(kernel, dtype=np.float32) / 255.0,
        device=similarity_map.device,
    )
    weight_sum = weights.sum()
    if float(weight_sum.detach().cpu()) <= 0.0:
        return similarity_map
    weights = weights / weight_sum
    padding_y = kernel_h // 2
    padding_x = kernel_w // 2
    score = F.conv2d(
        similarity_map[None, None].float(),
        weights[None, None].float(),
        padding=(padding_y, padding_x),
    )[0, 0]
    start_y = max(0, (score.shape[0] - feature_h) // 2)
    start_x = max(0, (score.shape[1] - feature_w) // 2)
    return score[start_y : start_y + feature_h, start_x : start_x + feature_w]


def _ensure_decode_batch_shapes(
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
        raise ValueError("decoded masks must have shape BxMxHxW")
    if scores.ndim != 2:
        raise ValueError("decoded scores must have shape BxM")
    if low_res_masks.ndim != 4:
        raise ValueError("decoded low_res_masks must have shape BxMxHxW")
    return masks, scores, low_res_masks


def _reference_shape_prior(
    references: Sequence[ContextReference],
    reference_embeddings: Sequence[Sam3ImageEmbedding],
) -> ReferenceShapePrior:
    reference_index = max(
        range(len(references)),
        key=lambda index: references[index].weight,
    )
    reference = references[reference_index]
    embedding = reference_embeddings[reference_index]
    mask_array = np.asarray(
        reference.mask.detach().cpu()
        if isinstance(reference.mask, torch.Tensor)
        else reference.mask
    ).astype(bool)
    if mask_array.ndim != 2:
        raise ValueError("reference mask must have shape HxW")
    if tuple(mask_array.shape) != tuple(embedding.orig_hw):
        raise ValueError("reference mask size must match reference image size")
    bbox = mask_to_box(mask_array)
    if bbox is None:
        raise ValueError("reference mask must contain at least one foreground pixel")
    x0, y0, x1, y1 = bbox
    orig_h, orig_w = embedding.orig_hw
    return ReferenceShapePrior(
        roi_mask=mask_array[y0:y1, x0:x1].copy(),
        width_ratio=(x1 - x0) / float(orig_w),
        height_ratio=(y1 - y0) / float(orig_h),
    )


def _make_mask_prior_batch(
    shape_prior: ReferenceShapePrior,
    point_batch: np.ndarray | torch.Tensor,
    *,
    target_hw: tuple[int, int],
    scale: float,
    foreground: float,
    background: float,
) -> np.ndarray:
    points = _to_numpy(point_batch)
    target_h, target_w = target_hw
    batch = np.full(
        (len(points), target_h, target_w),
        float(background),
        dtype=np.float32,
    )
    box_w = max(1, int(round(shape_prior.width_ratio * target_w * scale)))
    box_h = max(1, int(round(shape_prior.height_ratio * target_h * scale)))
    for index, point in enumerate(points):
        cx, cy = float(point[0]), float(point[1])
        x0 = int(round(cx - box_w / 2.0))
        y0 = int(round(cy - box_h / 2.0))
        x1 = x0 + box_w
        y1 = y0 + box_h
        clipped_x0 = max(0, x0)
        clipped_y0 = max(0, y0)
        clipped_x1 = min(target_w, x1)
        clipped_y1 = min(target_h, y1)
        if clipped_x0 >= clipped_x1 or clipped_y0 >= clipped_y1:
            continue
        resized = Image.fromarray(shape_prior.roi_mask.astype(np.uint8) * 255).resize(
            (box_w, box_h),
            resample=Image.Resampling.BILINEAR,
        )
        prior = np.asarray(resized, dtype=np.float32) / 255.0
        src_x0 = clipped_x0 - x0
        src_y0 = clipped_y0 - y0
        src_x1 = src_x0 + (clipped_x1 - clipped_x0)
        src_y1 = src_y0 + (clipped_y1 - clipped_y0)
        prior_crop = prior[src_y0:src_y1, src_x0:src_x1]
        batch[index, clipped_y0:clipped_y1, clipped_x0:clipped_x1] = np.where(
            prior_crop >= 0.5,
            float(foreground),
            float(background),
        )
    return batch


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
