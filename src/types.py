from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class Sam3ImageEmbedding:
    image_embed: torch.Tensor
    high_res_features: tuple[torch.Tensor, ...]
    orig_hw: tuple[int, int]


@dataclass(frozen=True)
class Sam3PromptBatch:
    embedding: Sam3ImageEmbedding
    point_coords: np.ndarray | torch.Tensor | None = None
    point_labels: np.ndarray | torch.Tensor | None = None
    box: np.ndarray | torch.Tensor | None = None
    mask_input: np.ndarray | torch.Tensor | None = None


@dataclass(frozen=True)
class MemoryReference:
    image: Image.Image | np.ndarray
    mask: np.ndarray | torch.Tensor
    obj_id: int


@dataclass(frozen=True)
class MemoryPrediction:
    frame_index: int
    obj_ids: list[int]
    masks: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class GroundingPrediction:
    masks: np.ndarray
    mask_logits: np.ndarray
    boxes_xyxy: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class VisualLanguageCache:
    language_features: torch.Tensor
    language_mask: torch.Tensor
    language_embeds: torch.Tensor | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "VisualLanguageCache":
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if "language_features" not in cache or "language_mask" not in cache:
            raise ValueError(
                "visual cache must contain language_features and language_mask"
            )
        return cls(
            language_features=cache["language_features"],
            language_mask=cache["language_mask"],
            language_embeds=cache.get("language_embeds"),
        )

    def to_backbone_out(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        output = {
            "language_features": self.language_features.to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            ),
            "language_mask": self.language_mask.to(
                device=device,
                dtype=torch.bool,
                non_blocking=True,
            ),
        }
        if self.language_embeds is not None:
            output["language_embeds"] = self.language_embeds.to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )
        return output


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
    image_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class MaskInstance:
    segmentation: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    score: float
    source: str = "unknown"
    concept_id: int | None = None
    object_id: int | None = None
    context_score: float | None = None
    base_score: float | None = None
    predicted_iou: float | None = None
    stability_score: float | None = None
    point_coords: tuple[float, float] | None = None
    crop_box: tuple[int, int, int, int] | None = None
    crop_grid: int | None = None
    crop_index: int | None = None
    image_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        segmentation = np.asarray(self.segmentation).astype(bool, copy=False)
        if segmentation.ndim != 2:
            raise ValueError("segmentation must be a 2D ROI mask")
        bbox = _xyxy_tuple(self.bbox, "bbox")
        expected_shape = (bbox[3] - bbox[1], bbox[2] - bbox[0])
        if segmentation.shape != expected_shape:
            raise ValueError(
                "segmentation shape must match bbox size: "
                f"got {segmentation.shape}, expected {expected_shape}"
            )
        if self.area < 0:
            raise ValueError("area must be non-negative")
        if int(segmentation.sum()) != int(self.area):
            raise ValueError("area must match segmentation foreground pixels")
        _validate_optional_id(self.concept_id, "concept_id")
        _validate_optional_id(self.object_id, "object_id")

        object.__setattr__(self, "segmentation", segmentation)
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "area", int(self.area))
        object.__setattr__(self, "score", float(self.score))
        if self.context_score is not None:
            object.__setattr__(self, "context_score", float(self.context_score))
        if self.base_score is not None:
            object.__setattr__(self, "base_score", float(self.base_score))
        if self.predicted_iou is not None:
            object.__setattr__(self, "predicted_iou", float(self.predicted_iou))
        if self.stability_score is not None:
            object.__setattr__(self, "stability_score", float(self.stability_score))
        if self.point_coords is not None:
            object.__setattr__(
                self,
                "point_coords",
                (float(self.point_coords[0]), float(self.point_coords[1])),
            )
        if self.crop_box is not None:
            object.__setattr__(self, "crop_box", _xyxy_tuple(self.crop_box, "crop_box"))
        if self.image_size is not None:
            object.__setattr__(self, "image_size", _image_size_tuple(self.image_size))

    def to_full_mask(self) -> np.ndarray:
        if self.image_size is None:
            raise ValueError("image_size is required")
        width, height = self.image_size
        x0, y0, x1, y1 = self.bbox
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise ValueError("bbox must be inside image_size")
        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[y0:y1, x0:x1] = self.segmentation
        return full_mask


@dataclass(frozen=True)
class ReferenceExample:
    concept_id: int
    mask: np.ndarray
    image: Image.Image | np.ndarray | None = None
    box_xyxy: tuple[int, int, int, int] | None = None
    object_id: int | None = None
    weight: float = 1.0
    source: str = "reference"

    def __post_init__(self) -> None:
        if self.image is not None:
            _validate_image(self.image, "image")
        _validate_optional_id(self.concept_id, "concept_id")
        _validate_optional_id(self.object_id, "object_id")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        mask = np.asarray(self.mask).astype(bool, copy=False)
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D array")
        if self.image is not None and mask.shape != _image_hw(self.image):
            raise ValueError("mask shape must match image size")
        if not mask.any():
            raise ValueError("mask must contain foreground pixels")
        box = (
            _mask_to_box(mask)
            if self.box_xyxy is None
            else _xyxy_tuple(self.box_xyxy, "box_xyxy")
        )

        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "box_xyxy", box)
        object.__setattr__(self, "concept_id", int(self.concept_id))
        if self.object_id is not None:
            object.__setattr__(self, "object_id", int(self.object_id))
        object.__setattr__(self, "weight", float(self.weight))

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass(frozen=True)
class ContextReference:
    image: Image.Image | np.ndarray
    mask: np.ndarray | torch.Tensor
    weight: float = 1.0
    concept_id: int = 0

    def __post_init__(self) -> None:
        _validate_image(self.image, "image")
        _validate_optional_id(self.concept_id, "concept_id")
        object.__setattr__(self, "concept_id", int(self.concept_id))


@dataclass(frozen=True)
class ContextPrediction:
    segmentation: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    point_coords: tuple[float, float]
    context_score: float
    predicted_iou: float
    stability_score: float
    score: float
    image_size: tuple[int, int]
    area_score: float = 1.0
    concept_id: int = 0


def _validate_optional_id(value: int | None, name: str) -> None:
    if value is not None and int(value) < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_image(value: Image.Image | np.ndarray, name: str) -> None:
    if isinstance(value, Image.Image):
        return
    if isinstance(value, np.ndarray):
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(f"{name} must have shape HxWx3")
        return
    raise TypeError(f"{name} must be a PIL image or NumPy array")


def _image_hw(value: Image.Image | np.ndarray) -> tuple[int, int]:
    if isinstance(value, Image.Image):
        width, height = value.size
        return height, width
    return int(value.shape[0]), int(value.shape[1])


def _xyxy_tuple(
    value: Sequence[int | float],
    name: str,
) -> tuple[int, int, int, int]:
    if len(value) != 4:
        raise ValueError(f"{name} must have four entries")
    x0, y0, x1, y1 = (int(v) for v in value)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{name} must be an inclusive-exclusive xyxy box")
    return x0, y0, x1, y1


def _image_size_tuple(value: Sequence[int | float]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("image_size must have width and height")
    width, height = (int(v) for v in value)
    if width <= 0 or height <= 0:
        raise ValueError("image_size entries must be positive")
    return width, height


def _mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
