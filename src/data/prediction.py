from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Sam3ImageEmbedding:
    image_embed: torch.Tensor
    high_res_features: tuple[torch.Tensor, ...]
    orig_hw: tuple[int, int]


@dataclass(frozen=True)
class Sam3PromptBatch:
    embedding: Sam3ImageEmbedding
    point_coords: np.ndarray | None = None
    point_labels: np.ndarray | None = None
    box: np.ndarray | None = None
    mask_input: np.ndarray | torch.Tensor | None = None
