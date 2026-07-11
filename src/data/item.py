from typing import Any

import numpy as np
from PIL import Image as PILImage

from .augment.prompt.box import jitter_mask_box
from .augment.prompt.mask import degrade_mask_prompt
from .augment.prompt.point import sample_point_prompt

MASK_OPS = ("none", "shift", "erode", "dilate", "blur", "resize")


def point(
    image: np.ndarray,
    target: np.ndarray,
    union: np.ndarray,
    bg_prob: float,
    mask_size: int,
) -> dict[str, Any]:
    out = sample_point_prompt(target, union, bg_prob=bg_prob)
    prompt = empty_prompt("point")
    prompt["points"] = out["points"]
    prompt["point_labels"] = out["point_labels"]
    return {
        "image": image,
        "prompt": prompt,
        "target": resize_mask(out["target"], mask_size, binary=True),
        "mask_valid": out["has_mask"],
        "is_auto_bg": out["is_auto_bg"],
    }


def box(
    image: np.ndarray,
    target: np.ndarray,
    jitter: float,
    mask_size: int,
) -> dict[str, Any]:
    prompt = empty_prompt("box")
    prompt["box"] = jitter_mask_box(target, image.shape, amount=jitter)
    return {
        "image": image,
        "prompt": prompt,
        "target": resize_mask(target, mask_size, binary=True),
        "mask_valid": True,
        "is_auto_bg": False,
    }


def mask(
    image: np.ndarray,
    target: np.ndarray,
    mask_size: int,
) -> dict[str, Any]:
    prompt = empty_prompt("mask")
    prompt["mask"] = resize_mask(
        degrade_mask_prompt(target, ops=MASK_OPS),
        mask_size,
    )
    return {
        "image": image,
        "prompt": prompt,
        "target": resize_mask(target, mask_size, binary=True),
        "mask_valid": True,
        "is_auto_bg": False,
    }


def resize_mask(value: np.ndarray, size: int, binary: bool = False) -> np.ndarray:
    value = np.asarray(value)
    if binary:
        value = value > 0
    image = PILImage.fromarray(
        np.clip(value.astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8),
        mode="L",
    )
    image = image.resize((size, size), PILImage.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def empty_prompt(kind: str) -> dict[str, Any]:
    return {
        "type": kind,
        "points": None,
        "point_labels": None,
        "box": None,
        "mask": None,
    }
