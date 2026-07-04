from collections.abc import Iterator
from math import ceil

import numpy as np
from PIL import Image


def build_point_grid(points_per_side: int) -> np.ndarray:
    if points_per_side <= 0:
        raise ValueError("points_per_side must be a positive integer")
    offset = 1.0 / (2 * points_per_side)
    points = np.linspace(offset, 1.0 - offset, points_per_side, dtype=np.float32)
    xv, yv = np.meshgrid(points, points)
    return np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.float32)


def generate_crop_boxes(
    width: int,
    height: int,
    grid_size: int,
    overlap_ratio: float,
) -> list[tuple[int, int, int, int]]:
    if grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    if overlap_ratio < 0.0 or overlap_ratio >= 0.5:
        raise ValueError("overlap_ratio must be in [0.0, 0.5)")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if grid_size == 1:
        return [(0, 0, width, height)]

    overlap_w = int(round((width / grid_size) * overlap_ratio))
    overlap_h = int(round((height / grid_size) * overlap_ratio))
    crop_w = int(ceil((width + overlap_w * (grid_size - 1)) / grid_size))
    crop_h = int(ceil((height + overlap_h * (grid_size - 1)) / grid_size))
    stride_w = crop_w - overlap_w
    stride_h = crop_h - overlap_h

    boxes: list[tuple[int, int, int, int]] = []
    for iy in range(grid_size):
        y0 = min(iy * stride_h, height - crop_h)
        y1 = min(y0 + crop_h, height)
        for ix in range(grid_size):
            x0 = min(ix * stride_w, width - crop_w)
            x1 = min(x0 + crop_w, width)
            boxes.append((int(x0), int(y0), int(x1), int(y1)))
    return boxes


def image_size(image: Image.Image | np.ndarray) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return image.size
    if isinstance(image, np.ndarray):
        if image.ndim != 3:
            raise ValueError("NumPy images must have shape HxWxC")
        height, width = image.shape[:2]
        return width, height
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def crop_image(
    image: Image.Image | np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> Image.Image | np.ndarray:
    x0, y0, x1, y1 = crop_box
    if isinstance(image, Image.Image):
        return image.crop(crop_box)
    return image[y0:y1, x0:x1, :]


def touches_internal_crop_edge(
    local_bbox: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    full_size: tuple[int, int],
) -> bool:
    x0, y0, x1, y1 = local_bbox
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
    full_width, full_height = full_size
    touches_left = x0 <= 0 and crop_x0 > 0
    touches_top = y0 <= 0 and crop_y0 > 0
    touches_right = x1 >= crop_x1 - crop_x0 and crop_x1 < full_width
    touches_bottom = y1 >= crop_y1 - crop_y0 and crop_y1 < full_height
    return touches_left or touches_top or touches_right or touches_bottom


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def calculate_stability_score(
    logits: np.ndarray,
    mask_threshold: float = 0.0,
    offset: float = 1.0,
) -> float:
    high = logits > (mask_threshold + offset)
    low = logits > (mask_threshold - offset)
    union = int(low.sum())
    if union == 0:
        return 0.0
    return float(high.sum() / union)


def batched(items: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
