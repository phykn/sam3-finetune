from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class ReferenceExample:
    concept_id: int
    mask: np.ndarray
    image: object | None = None
    box_xyxy: tuple[int, int, int, int] | None = None
    object_id: int | None = None
    weight: float = 1.0
    source: str = "reference"

    def __post_init__(self) -> None:
        _validate_optional_id(self.concept_id, "concept_id")
        _validate_optional_id(self.object_id, "object_id")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        mask = np.asarray(self.mask).astype(bool, copy=False)
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D array")
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
    image: Image.Image | np.ndarray | torch.Tensor
    mask: np.ndarray | torch.Tensor
    weight: float = 1.0


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


@dataclass(frozen=True)
class ContextPrototype:
    positive: torch.Tensor
    negative: torch.Tensor | None
    reference_area_ratio: float


@dataclass(frozen=True)
class ReferenceShapePrior:
    roi_mask: np.ndarray
    width_ratio: float
    height_ratio: float


def _validate_optional_id(value: int | None, name: str) -> None:
    if value is not None and int(value) < 0:
        raise ValueError(f"{name} must be non-negative")


def _xyxy_tuple(
    value: tuple[int, int, int, int], name: str
) -> tuple[int, int, int, int]:
    if len(value) != 4:
        raise ValueError(f"{name} must have four entries")
    x0, y0, x1, y1 = (int(v) for v in value)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{name} must be an inclusive-exclusive xyxy box")
    return x0, y0, x1, y1


def _mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
