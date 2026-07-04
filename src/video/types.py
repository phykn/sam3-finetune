from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class MemoryReference:
    image: Image.Image | np.ndarray
    mask: np.ndarray | torch.Tensor
    obj_id: int


@dataclass(frozen=True)
class PreparedReference:
    reference: MemoryReference
    frame_index: int


@dataclass(frozen=True)
class MemoryPrediction:
    frame_index: int
    obj_ids: list[int]
    masks: np.ndarray
    scores: np.ndarray
