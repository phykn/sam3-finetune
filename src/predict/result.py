from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ImageEmbed:
    image_embed: torch.Tensor
    high_res: tuple[torch.Tensor, ...]
    orig_hw: tuple[int, int]


@dataclass(frozen=True)
class SingleResult:
    masks: np.ndarray
    scores: np.ndarray
    logits: np.ndarray
