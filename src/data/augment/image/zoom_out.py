import numpy as np

from .resize import resize


def random_zoom_out(
    image: np.ndarray,
    mask: np.ndarray,
    scale: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    _check_scale(scale)

    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)
    height, width = image.shape[:2]
    side = max(height, width)
    out_side = max(1, int(round(side / scale)))
    y0 = _rand_start(side, out_side)
    x0 = _rand_start(side, out_side)

    square_image, square_mask = resize(image, mask, size=(side, side))
    small_image, small_mask = resize(
        square_image, square_mask, size=(out_side, out_side)
    )
    out_image = np.zeros((side, side) + image.shape[2:], dtype=np.uint8)
    out_mask = np.zeros((side, side) + mask.shape[2:], dtype=np.uint8)
    out_image[y0 : y0 + out_side, x0 : x0 + out_side] = small_image
    out_mask[y0 : y0 + out_side, x0 : x0 + out_side] = small_mask
    return out_image, out_mask


def _check_scale(scale: float) -> None:
    if scale < 1.0:
        raise ValueError("scale must be 1 or greater")


def _rand_start(size: int, crop_size: int) -> int:
    if size <= crop_size:
        return 0
    return int(np.random.randint(size - crop_size + 1))
