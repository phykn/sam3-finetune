import numpy as np
from PIL import Image, ImageFilter

OPS = ("none", "shift", "erode", "dilate", "blur", "resize")


def degrade_mask_prompt(
    target: np.ndarray,
    ops: tuple[str, ...] | None = None,
) -> np.ndarray:
    ops = OPS if ops is None else tuple(ops)
    op = str(np.random.choice(ops))

    if op == "none":
        return target.astype(np.float32, copy=False)
    if op == "shift":
        return _shift_mask(target).astype(np.float32, copy=False)
    if op == "erode":
        return _to_binary_float(_to_pil_mask(target).filter(ImageFilter.MinFilter(3)))
    if op == "dilate":
        return _to_binary_float(_to_pil_mask(target).filter(ImageFilter.MaxFilter(3)))
    if op == "blur":
        return _to_soft_float(_to_pil_mask(target).filter(ImageFilter.BoxBlur(2)))
    if op == "resize":
        return _make_coarse_mask(target)
    raise ValueError(f"unknown mask op: {op}")


def _shift_mask(target: np.ndarray) -> np.ndarray:
    height, width = target.shape
    box = _find_tight_box(target)
    if box is None:
        return np.zeros_like(target, dtype=np.uint8)

    x0, y0, x1, y1 = box
    max_dx = max(1, int(round((x1 - x0) * 0.05)))
    max_dy = max(1, int(round((y1 - y0) * 0.05)))
    dx = int(np.random.randint(-max_dx, max_dx + 1))
    dy = int(np.random.randint(-max_dy, max_dy + 1))

    out = np.zeros_like(target, dtype=np.uint8)
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(width, width - dx)
    src_y1 = min(height, height - dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    dst_x1 = min(width, width + dx)
    dst_y1 = min(height, height + dy)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = target[src_y0:src_y1, src_x0:src_x1]
    return out


def _make_coarse_mask(target: np.ndarray) -> np.ndarray:
    height, width = target.shape
    small = (max(1, width // 2), max(1, height // 2))
    image = _to_pil_mask(target)
    image = image.resize(small, Image.Resampling.BILINEAR)
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return _to_soft_float(image)


def _find_tight_box(target: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(target > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _to_pil_mask(target: np.ndarray) -> Image.Image:
    return Image.fromarray((target > 0).astype(np.uint8) * 255, mode="L")


def _to_binary_float(image: Image.Image) -> np.ndarray:
    return (np.asarray(image) > 127).astype(np.float32)


def _to_soft_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0
