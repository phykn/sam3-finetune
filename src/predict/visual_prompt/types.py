from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class VisualExemplar:
    image: Image.Image | np.ndarray
    mask: np.ndarray | torch.Tensor
    concept_id: int = 0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.image, Image.Image):
            image_hw = (self.image.height, self.image.width)
        elif isinstance(self.image, np.ndarray):
            if self.image.ndim != 3 or self.image.shape[2] != 3:
                raise ValueError("image must have shape HxWx3")
            image_hw = (int(self.image.shape[0]), int(self.image.shape[1]))
        else:
            raise TypeError("image must be a PIL image or NumPy array")

        if int(self.concept_id) < 0:
            raise ValueError("concept_id must be non-negative")
        if self.weight <= 0:
            raise ValueError("weight must be positive")

        mask = (
            self.mask.detach().cpu().numpy()
            if isinstance(self.mask, torch.Tensor)
            else self.mask
        )
        mask = np.asarray(mask).astype(bool, copy=False)
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D array")
        if tuple(mask.shape) != image_hw:
            raise ValueError("mask shape must match image size")
        if not bool(mask.any()):
            raise ValueError("mask must contain foreground pixels")

        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "concept_id", int(self.concept_id))
        object.__setattr__(self, "weight", float(self.weight))


@dataclass(frozen=True)
class PreparedVisualConcept:
    concept_id: int
    exemplars: tuple[VisualExemplar, ...]
    visual_prompt_embed: torch.Tensor
    visual_prompt_mask: torch.Tensor


@dataclass(frozen=True)
class PreparedVisualPrompts:
    concepts: tuple[PreparedVisualConcept, ...]


@dataclass(frozen=True)
class VisualPromptPrediction:
    concept_id: int
    masks: np.ndarray
    boxes_xyxy: np.ndarray
    scores: np.ndarray
