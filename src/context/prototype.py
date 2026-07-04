from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..image.types import Sam3ImageEmbedding
from ..masks.geometry import mask_to_box
from .types import ContextReference


@dataclass(frozen=True)
class ContextPrototype:
    positive: torch.Tensor
    negative: torch.Tensor | None
    reference_area_ratio: float


def build_context_prototype(
    references: Sequence[ContextReference],
    reference_embeddings: Sequence[Sam3ImageEmbedding],
    *,
    feature_layer: str | int,
    negative_context_mode: str,
    negative_context_scale: float,
) -> ContextPrototype:
    positive_sum: torch.Tensor | None = None
    negative_sum: torch.Tensor | None = None
    total_weight = 0.0
    negative_total_weight = 0.0
    weighted_area_ratio = 0.0
    for reference, embedding in zip(references, reference_embeddings):
        if reference.weight <= 0:
            raise ValueError("reference weight must be positive")
        features = select_feature(embedding, feature_layer)
        positive = _masked_feature_mean(
            features,
            reference.mask,
            embedding.orig_hw,
        )
        positive_sum = (
            positive * float(reference.weight)
            if positive_sum is None
            else positive_sum + positive * float(reference.weight)
        )
        negative = _negative_feature_mean(
            features,
            reference.mask,
            embedding.orig_hw,
            mode=negative_context_mode,
            scale=negative_context_scale,
        )
        if negative is not None:
            negative_sum = (
                negative * float(reference.weight)
                if negative_sum is None
                else negative_sum + negative * float(reference.weight)
            )
            negative_total_weight += float(reference.weight)
        total_weight += float(reference.weight)
        weighted_area_ratio += _mask_area_ratio(
            reference.mask,
            embedding.orig_hw,
        ) * float(reference.weight)
    assert positive_sum is not None
    negative_prototype = (
        F.normalize(negative_sum / negative_total_weight, dim=0)
        if negative_sum is not None and negative_total_weight > 0.0
        else None
    )
    return ContextPrototype(
        positive=F.normalize(positive_sum / total_weight, dim=0),
        negative=negative_prototype,
        reference_area_ratio=weighted_area_ratio / total_weight,
    )


def select_feature(
    embedding: Sam3ImageEmbedding,
    feature_layer: str | int,
) -> torch.Tensor:
    if feature_layer == "image_embed":
        features = embedding.image_embed
    elif isinstance(feature_layer, int):
        features = embedding.high_res_features[feature_layer]
    else:
        raise ValueError("feature_layer must be 'image_embed' or a feature index")
    if features.ndim != 4 or features.shape[0] != 1:
        raise ValueError("selected embedding feature must have shape 1xCxHxW")
    return features[0].float()


def similarity_map(
    target_features: torch.Tensor,
    prototype: ContextPrototype,
    *,
    negative_context_weight: float,
) -> torch.Tensor:
    normalized_features = F.normalize(target_features, dim=0)
    positive = torch.einsum(
        "c,chw->hw",
        prototype.positive.to(target_features.device),
        normalized_features,
    )
    if prototype.negative is None or negative_context_weight == 0.0:
        return positive
    negative = torch.einsum(
        "c,chw->hw",
        prototype.negative.to(target_features.device),
        normalized_features,
    )
    return positive - negative * negative_context_weight


def resize_similarity_map(
    score_map: torch.Tensor,
    orig_hw: tuple[int, int],
) -> np.ndarray:
    resized = F.interpolate(
        score_map[None, None].float(),
        size=orig_hw,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.detach().cpu().numpy()


def mean_score_over_mask(scores: np.ndarray, mask: np.ndarray) -> float:
    values = scores[mask]
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _masked_feature_mean(
    features: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    orig_hw: tuple[int, int],
) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("features must have shape CxHxW")
    mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=features.device)
    if mask_tensor.ndim != 2:
        raise ValueError("reference mask must have shape HxW")
    if tuple(mask_tensor.shape) != tuple(orig_hw):
        raise ValueError("reference mask size must match reference image size")
    mask_weights = F.interpolate(
        mask_tensor[None, None],
        size=features.shape[-2:],
        mode="area",
    )[0, 0]
    weight_sum = mask_weights.sum()
    if float(weight_sum.detach().cpu()) <= 0.0:
        raise ValueError("reference mask must contain at least one foreground pixel")
    prototype = (features * mask_weights[None]).sum(dim=(1, 2)) / weight_sum
    return F.normalize(prototype.float(), dim=0)


def _negative_feature_mean(
    features: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    orig_hw: tuple[int, int],
    *,
    mode: str,
    scale: float,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if features.ndim != 3:
        raise ValueError("features must have shape CxHxW")
    mask_array = np.asarray(
        mask.detach().cpu() if isinstance(mask, torch.Tensor) else mask
    )
    if mask_array.ndim != 2:
        raise ValueError("reference mask must have shape HxW")
    if tuple(mask_array.shape) != tuple(orig_hw):
        raise ValueError("reference mask size must match reference image size")
    mask_bool = mask_array.astype(bool)
    bbox = mask_to_box(mask_bool)
    if bbox is None:
        raise ValueError("reference mask must contain at least one foreground pixel")

    if mode == "image":
        negative_mask = ~mask_bool
    elif mode == "local":
        negative_mask = _expanded_bbox_background_mask(mask_bool, bbox, scale=scale)
        if not bool(negative_mask.any()):
            negative_mask = ~mask_bool
    else:
        raise ValueError("negative context mode must be 'none', 'image', or 'local'")

    if not bool(negative_mask.any()):
        return None
    mask_weights = F.interpolate(
        torch.as_tensor(negative_mask, dtype=torch.float32, device=features.device)[
            None, None
        ],
        size=features.shape[-2:],
        mode="area",
    )[0, 0]
    weight_sum = mask_weights.sum()
    if float(weight_sum.detach().cpu()) <= 0.0:
        return None
    prototype = (features * mask_weights[None]).sum(dim=(1, 2)) / weight_sum
    return F.normalize(prototype.float(), dim=0)


def _expanded_bbox_background_mask(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    scale: float,
) -> np.ndarray:
    height, width = mask.shape
    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    expanded_w = box_w * scale
    expanded_h = box_h * scale
    expanded_x0 = max(0, int(round(center_x - expanded_w / 2.0)))
    expanded_y0 = max(0, int(round(center_y - expanded_h / 2.0)))
    expanded_x1 = min(width, int(round(center_x + expanded_w / 2.0)))
    expanded_y1 = min(height, int(round(center_y + expanded_h / 2.0)))
    local = np.zeros_like(mask, dtype=bool)
    local[expanded_y0:expanded_y1, expanded_x0:expanded_x1] = True
    return local & ~mask.astype(bool)


def _mask_area_ratio(
    mask: np.ndarray | torch.Tensor,
    orig_hw: tuple[int, int],
) -> float:
    mask_array = np.asarray(
        mask.detach().cpu() if isinstance(mask, torch.Tensor) else mask
    )
    if mask_array.ndim != 2:
        raise ValueError("reference mask must have shape HxW")
    if tuple(mask_array.shape) != tuple(orig_hw):
        raise ValueError("reference mask size must match reference image size")
    area = float(mask_array.astype(bool).sum())
    if area <= 0.0:
        raise ValueError("reference mask must contain at least one foreground pixel")
    return area / float(orig_hw[0] * orig_hw[1])
