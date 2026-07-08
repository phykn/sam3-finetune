import albumentations as A
import numpy as np


def random_pixel(
    image: np.ndarray,
    ops: list[str] | tuple[str, ...],
) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    op = str(np.random.choice(tuple(ops)))
    if op == "none":
        return image
    if op == "brightness":
        transform = A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.0,
            p=1,
        )
    elif op == "contrast":
        transform = A.RandomBrightnessContrast(
            brightness_limit=0.0,
            contrast_limit=0.2,
            p=1,
        )
    elif op == "saturation":
        transform = A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=20,
            val_shift_limit=0,
            p=1,
        )
    elif op == "blur":
        transform = A.GaussianBlur(
            blur_limit=(3, 5),
            sigma_limit=(0.5, 1.5),
            p=1,
        )
    elif op == "noise":
        transform = A.GaussNoise(p=1)
    elif op == "dropout":
        transform = A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(0.05, 0.2),
            hole_width_range=(0.05, 0.2),
            fill=0,
            p=1,
        )
    else:
        raise ValueError(f"unknown image op: {op}")

    out = transform(image=image)["image"]
    return np.asarray(out, dtype=np.uint8)
