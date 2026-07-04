from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Normalize, Resize, ToTensor


class Sam3Transforms:
    def __init__(self, resolution: int = 1008, mask_threshold: float = 0.0) -> None:
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.to_tensor = ToTensor()
        self.resize = Resize((resolution, resolution))
        self.normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def preprocess_image(
        self,
        image: Image.Image | np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if isinstance(image, Image.Image):
            image_rgb = image.convert("RGB")
            width, height = image_rgb.size
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("NumPy images must have shape HxWx3")
            image_rgb = Image.fromarray(image.astype(np.uint8), mode="RGB")
            height, width = image.shape[:2]
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")

        tensor = self.normalize(self.resize(self.to_tensor(image_rgb))).unsqueeze(0)
        return tensor.to(device=device), (height, width)

    def transform_coords(
        self,
        coords: np.ndarray | torch.Tensor,
        orig_hw: tuple[int, int],
    ) -> torch.Tensor:
        coords_t = torch.as_tensor(coords, dtype=torch.float32).clone()
        if coords_t.shape[-1] != 2:
            raise ValueError("Point coordinates must end with dimension 2")
        h, w = orig_hw
        coords_t[..., 0] = coords_t[..., 0] / float(w)
        coords_t[..., 1] = coords_t[..., 1] / float(h)
        return coords_t * float(self.resolution)

    def transform_box(
        self,
        box: np.ndarray | torch.Tensor,
        orig_hw: tuple[int, int],
    ) -> torch.Tensor:
        box_t = torch.as_tensor(box, dtype=torch.float32)
        if box_t.numel() != 4:
            raise ValueError("Box must contain four values: x0, y0, x1, y1")
        return self.transform_coords(box_t.reshape(1, 2, 2), orig_hw)

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        orig_hw: tuple[int, int],
        return_logits: bool = False,
    ) -> torch.Tensor:
        masks = F.interpolate(masks.float(), orig_hw, mode="bilinear", align_corners=False)
        if return_logits:
            return masks
        return masks > self.mask_threshold


def save_mask_png(mask: np.ndarray, path: str | Path) -> None:
    mask_uint8 = mask.astype(np.uint8) * 255
    Image.fromarray(mask_uint8, mode="L").save(path)


def save_overlay_png(image: Image.Image, mask: np.ndarray, path: str | Path) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha = mask.astype(np.uint8) * 120
    overlay.putalpha(Image.fromarray(alpha, mode="L"))
    Image.alpha_composite(base, overlay).save(path)
