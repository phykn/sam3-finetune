import albumentations as A
import numpy as np


def resize(
    image: np.ndarray,
    mask: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = size
    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)
    out_h, out_w = _fit_size(image.shape[:2], size)
    transform = A.Resize(height=out_h, width=out_w, p=1)
    out = transform(
        image=image,
        mask=mask,
    )
    image = np.asarray(out["image"], dtype=np.uint8)
    mask = np.asarray(out["mask"], dtype=np.uint8)

    out_image = np.zeros((height, width) + image.shape[2:], dtype=np.uint8)
    out_mask = np.zeros((height, width) + mask.shape[2:], dtype=np.uint8)
    out_image[:out_h, :out_w] = image[:out_h, :out_w]
    out_mask[:out_h, :out_w] = mask[:out_h, :out_w]
    return out_image, out_mask


def _fit_size(
    source: tuple[int, int],
    target: tuple[int, int],
) -> tuple[int, int]:
    src_h, src_w = source
    dst_h, dst_w = target
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError("image size must be positive")

    scale = min(dst_h / src_h, dst_w / src_w)
    height = max(1, min(dst_h, int(round(src_h * scale))))
    width = max(1, min(dst_w, int(round(src_w * scale))))
    return height, width
