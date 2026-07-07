import numpy as np
import torch
from src.data import image


def test_make_tensor_normalizes_and_returns_original_size():
    tensor, orig_hw = image.make_tensor(
        np.zeros((10, 20, 3), dtype=np.uint8),
        8,
        torch.device("cpu"),
    )

    assert orig_hw == (10, 20)
    assert tensor.shape == (1, 3, 8, 8)
    assert torch.allclose(tensor, torch.full_like(tensor, -1.0))
