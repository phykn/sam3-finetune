from pathlib import Path

import numpy as np
from PIL import Image


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    mask_uint8 = mask.astype(np.uint8) * 255
    Image.fromarray(mask_uint8, mode="L").save(path)


def save_overlay(image: Image.Image, mask: np.ndarray, path: str | Path) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha = mask.astype(np.uint8) * 120
    overlay.putalpha(Image.fromarray(alpha, mode="L"))
    Image.alpha_composite(base, overlay).save(path)
