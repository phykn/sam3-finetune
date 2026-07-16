import numpy as np
import torch
from src.data import image


def test_to_tensor_normalizes_uint8_hwc_image():
    tensor = image.to_tensor(np.full((2, 3, 3), 255, dtype=np.uint8))

    assert tensor.shape == (3, 2, 3)
    assert torch.allclose(tensor, torch.ones_like(tensor))


def test_make_tensor_normalizes_and_returns_original_size():
    tensor, orig_hw = image.make_tensor(
        np.zeros((10, 20, 3), dtype=np.uint8),
        8,
        torch.device("cpu"),
    )

    assert orig_hw == (10, 20)
    assert tensor.shape == (1, 3, 8, 8)
    assert torch.allclose(tensor, torch.full_like(tensor, -1.0))


def test_make_tensor_uses_the_shared_image_resize():
    array = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)

    tensor, _orig_hw = image.make_tensor(array, 6, torch.device("cpu"))
    expected = image.to_tensor(image.resize_image(array, 6)).unsqueeze(0)

    assert torch.equal(tensor, expected)


def test_resize_mask_stretches_all_channels_to_the_target_square():
    mask = np.ones((4, 8, 3), dtype=np.uint8)

    out = image.resize_mask(mask, 8)

    assert out.shape == (8, 8, 3)
    assert (out == 1).all()
