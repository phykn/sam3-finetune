import numpy as np

from .resize import resize


def random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    scale: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    _check_scale(scale)

    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)
    height, width = image.shape[:2]
    out_side = max(height, width)
    side = max(1, int(round(min(height, width) * scale)))

    box = _find_box(mask)
    if box is None:
        y0 = _rand_start(height, side)
        x0 = _rand_start(width, side)
    else:
        x0, y0, x1, y1 = box
        side = _fit_object_crop(min(height, width), side, max(y1 - y0, x1 - x0))
        x0 = _rand_object_start(width, side, x0, x1)
        y0 = _rand_object_start(height, side, y0, y1)

    crop_image = image[y0 : y0 + side, x0 : x0 + side]
    crop_mask = mask[y0 : y0 + side, x0 : x0 + side]
    return resize(crop_image, crop_mask, size=(out_side, out_side))


def _check_scale(scale: float) -> None:
    if not 0.0 <= scale <= 1.0:
        raise ValueError("scale must be between 0 and 1")


def _find_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    source = mask[..., 0] if mask.ndim == 3 else mask
    ys, xs = np.where(source > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _rand_start(size: int, crop_size: int) -> int:
    if size <= crop_size:
        return 0
    return int(np.random.randint(size - crop_size + 1))


def _rand_object_start(size: int, crop_size: int, box0: int, box1: int) -> int:
    low = max(0, box1 - crop_size)
    high = min(box0, size - crop_size)
    if high <= low:
        center = int(round((box0 + box1 - crop_size) / 2))
        return int(np.clip(center, 0, max(0, size - crop_size)))
    return int(np.random.randint(low, high + 1))


def _fit_object_crop(size: int, crop_size: int, object_size: int) -> int:
    if crop_size > object_size:
        return crop_size
    margin = max(1, int(round(object_size * 0.25)))
    return min(size, object_size + margin)
