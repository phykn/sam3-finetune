from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


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
