import albumentations as A
import numpy as np

OPS = ("none", "brightness", "contrast", "saturation", "blur", "noise", "dropout")


def augment_pixel(image: np.ndarray) -> np.ndarray:
    op = str(np.random.choice(OPS))
    return _apply_op(image, op)


def _apply_op(image: np.ndarray, op: str) -> np.ndarray:
    transform = _make_transform(op)
    return transform(image=_uint8(image))["image"]


def _make_transform(op: str) -> A.BasicTransform:
    if op == "none":
        return A.NoOp(p=1)
    if op == "brightness":
        return A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.0,
            p=1,
        )
    if op == "contrast":
        return A.RandomBrightnessContrast(
            brightness_limit=0.0,
            contrast_limit=0.2,
            p=1,
        )
    if op == "saturation":
        return A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=20,
            val_shift_limit=0,
            p=1,
        )
    if op == "blur":
        return A.GaussianBlur(
            blur_limit=(3, 5),
            sigma_limit=(0.5, 1.5),
            p=1,
        )
    if op == "noise":
        return A.GaussNoise(p=1)
    if op == "dropout":
        return A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(0.05, 0.2),
            hole_width_range=(0.05, 0.2),
            fill=0,
            p=1,
        )
    raise ValueError(f"unknown pixel op: {op}")


def _uint8(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8)
