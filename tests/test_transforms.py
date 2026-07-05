import numpy as np
import torch
from PIL import Image
from src.predict.image_transform import ImageTransforms


def test_transform_coords_scales_pixel_points_to_model_resolution():
    transforms = ImageTransforms(resolution=1008)
    coords = np.array([[50.0, 25.0]], dtype=np.float32)

    out = transforms.transform_coords(coords, orig_hw=(100, 200))

    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[252.0, 252.0]]))


def test_transform_box_scales_xyxy_to_two_corner_points():
    transforms = ImageTransforms(resolution=1008)
    box = np.array([20.0, 10.0, 180.0, 90.0], dtype=np.float32)

    out = transforms.transform_box(box, orig_hw=(100, 200))

    assert out.shape == (1, 2, 2)
    assert torch.allclose(out[0, 0], torch.tensor([100.8, 100.8]))
    assert torch.allclose(out[0, 1], torch.tensor([907.2, 907.2]))


def test_transform_box_accepts_batched_xyxy_boxes():
    transforms = ImageTransforms(resolution=1008)
    boxes = np.array(
        [
            [20.0, 10.0, 180.0, 90.0],
            [0.0, 0.0, 200.0, 100.0],
        ],
        dtype=np.float32,
    )

    out = transforms.transform_box(boxes, orig_hw=(100, 200))

    assert out.shape == (2, 2, 2)
    assert torch.allclose(out[0, 0], torch.tensor([100.8, 100.8]))
    assert torch.allclose(out[1, 1], torch.tensor([1008.0, 1008.0]))


def test_preprocess_image_returns_batch_tensor_and_original_hw():
    transforms = ImageTransforms(resolution=1008)
    image = Image.new("RGB", (20, 10), color=(255, 0, 0))

    tensor, orig_hw = transforms.preprocess_image(image, device=torch.device("cpu"))

    assert orig_hw == (10, 20)
    assert tensor.shape == (1, 3, 1008, 1008)
    assert tensor.dtype == torch.float32
