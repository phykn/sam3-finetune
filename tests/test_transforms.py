import numpy as np
import torch
from PIL import Image
from src.predict.prompted.transforms import (
    ImageTransforms,
    preprocess_rgb_images,
    scale_coords,
    to_uint8_rgb_array,
)


def test_transform_coords_scales_pixel_points_to_model_resolution():
    transforms = ImageTransforms(resolution=1008)
    coords = np.array([[50.0, 25.0]], dtype=np.float32)

    out = transforms.transform_coords(coords, orig_hw=(100, 200))

    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[252.0, 252.0]]))


def test_scale_coords_scales_points_to_square_resolution():
    coords = torch.tensor([[[50.0, 25.0], [100.0, 50.0]]])

    out = scale_coords(coords, orig_hw=(100, 200), resolution=1008)

    assert out.shape == (1, 2, 2)
    assert torch.allclose(
        out,
        torch.tensor([[[252.0, 252.0], [504.0, 504.0]]]),
    )


def test_scale_coords_keeps_tensor_device():
    coords = torch.empty((1, 1, 2), device="meta")

    out = scale_coords(coords, orig_hw=(100, 200), resolution=1008)

    assert out.device.type == "meta"


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


def test_preprocess_float_numpy_image_preserves_unit_range_values():
    transforms = ImageTransforms(resolution=4)
    image = np.full((2, 2, 3), 0.5, dtype=np.float32)

    tensor, orig_hw = transforms.preprocess_image(image, device=torch.device("cpu"))

    assert orig_hw == (2, 2)
    assert tensor.shape == (1, 3, 4, 4)
    assert float(tensor.mean()) > -0.01


def test_to_uint8_rgb_array_clips_integer_values_before_cast():
    image = np.array([[[300, -10, 128]]], dtype=np.int32)

    out = to_uint8_rgb_array(image)

    np.testing.assert_array_equal(out, np.array([[[255, 0, 128]]], dtype=np.uint8))


def test_preprocess_rgb_images_returns_batch_and_frame_sizes():
    images = [
        Image.fromarray(np.zeros((3, 5, 3), dtype=np.uint8)),
        np.full((4, 6, 3), 0.5, dtype=np.float32),
    ]

    batch, orig_hw, frame_hws = preprocess_rgb_images(
        images,
        resolution=8,
        output_image_index=1,
        dtype=torch.float16,
    )

    assert tuple(batch.shape) == (2, 3, 8, 8)
    assert batch.dtype == torch.float16
    assert orig_hw == (4, 6)
    assert frame_hws == [(3, 5), (4, 6)]
    assert float(batch[1].mean()) > -0.01
