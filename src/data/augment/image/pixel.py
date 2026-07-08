import numpy as np
from PIL import Image, ImageFilter

OPS = ("none", "brightness", "contrast", "saturation", "blur", "noise")


def augment_pixel(image: np.ndarray) -> np.ndarray:
    op = str(np.random.choice(OPS))
    if op == "none":
        return _uint8(image)
    if op == "brightness":
        return _brightness(image)
    if op == "contrast":
        return _contrast(image)
    if op == "saturation":
        return _saturation(image)
    if op == "blur":
        return _blur(image)
    if op == "noise":
        return _noise(image)
    raise ValueError(f"unknown pixel op: {op}")


def _brightness(image: np.ndarray) -> np.ndarray:
    factor = np.random.uniform(0.8, 1.2)
    return _clip(_uint8(image).astype(np.float32) * factor)


def _contrast(image: np.ndarray) -> np.ndarray:
    factor = np.random.uniform(0.8, 1.2)
    array = _uint8(image).astype(np.float32)
    mean = array.mean(axis=(0, 1), keepdims=True)
    return _clip((array - mean) * factor + mean)


def _saturation(image: np.ndarray) -> np.ndarray:
    factor = np.random.uniform(0.8, 1.2)
    array = _uint8(image).astype(np.float32)
    gray = (
        array[..., 0:1] * 0.299
        + array[..., 1:2] * 0.587
        + array[..., 2:3] * 0.114
    )
    return _clip((array - gray) * factor + gray)


def _blur(image: np.ndarray) -> np.ndarray:
    radius = np.random.uniform(0.5, 1.5)
    pil = Image.fromarray(_uint8(image), mode="RGB")
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)), dtype=np.uint8)


def _noise(image: np.ndarray) -> np.ndarray:
    std = np.random.uniform(3.0, 8.0)
    array = _uint8(image).astype(np.float32)
    return _clip(array + np.random.normal(0.0, std, array.shape))


def _uint8(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8)


def _clip(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)
