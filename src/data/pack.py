from typing import Any

import numpy as np
from PIL import Image as PILImage

from .sample import png, rle


def image(value: PILImage.Image | np.ndarray) -> dict[str, Any]:
    if isinstance(value, PILImage.Image):
        array = np.asarray(value.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(value, dtype=np.uint8)
    return {
        "shape": list(array.shape),
        "dtype": "uint8",
        "color": "RGB",
        "format": "png",
        "encoding": "base64",
        "data": png.pack(array),
    }


def read_image(data: dict[str, Any]) -> PILImage.Image:
    return PILImage.fromarray(png.unpack(data["data"]), mode="RGB")


def mask(value: np.ndarray) -> dict[str, Any]:
    return rle.pack(np.asarray(value, dtype=np.uint8))


def read_mask(data: dict[str, Any]) -> np.ndarray:
    return rle.unpack(data).astype(bool)


def box_roi(value: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    mask = np.asarray(value, dtype=bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8)

    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    return (x0, y0, x1, y1), mask[y0:y1, x0:x1].astype(np.uint8)


def full(
    shape: tuple[int, ...] | list[int],
    box: tuple[int, int, int, int],
    roi: np.ndarray,
) -> np.ndarray:
    out = np.zeros(tuple(shape)[:2], dtype=bool)
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = np.asarray(roi, dtype=bool)
    return out
