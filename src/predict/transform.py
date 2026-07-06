import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def to_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be PIL image or numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("numpy image must have shape HxWx3")
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32, copy=False)
        if image.size > 0 and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255)
    return Image.fromarray(image.astype(np.uint8, copy=False), mode="RGB")


def image_tensor(
    image: Image.Image | np.ndarray,
    size: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    image = to_rgb(image)
    width, height = image.size
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor((0.5, 0.5, 0.5))[:, None, None]
    std = torch.tensor((0.5, 0.5, 0.5))[:, None, None]
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device), (height, width)


def points(value, orig_hw: tuple[int, int], size: int, device) -> torch.Tensor:
    coords = torch.as_tensor(value, dtype=torch.float32, device=device).clone()
    if coords.shape[-1] != 2:
        raise ValueError("point coordinates must end with 2 values")
    height, width = orig_hw
    scale = torch.tensor(
        [size / float(width), size / float(height)],
        dtype=coords.dtype,
        device=coords.device,
    )
    return coords * scale


def box(value, orig_hw: tuple[int, int], size: int, device) -> torch.Tensor:
    boxes = torch.as_tensor(value, dtype=torch.float32, device=device)
    if boxes.shape[-1] != 4:
        raise ValueError("box must end with 4 values")
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, 2, 2)
    else:
        boxes = boxes.reshape(*boxes.shape[:-1], 2, 2)
    return points(boxes, orig_hw, size, device)


def resize_masks(
    masks: torch.Tensor,
    orig_hw: tuple[int, int],
    threshold: float,
) -> torch.Tensor:
    masks = F.interpolate(masks.float(), orig_hw, mode="bilinear", align_corners=False)
    return masks > threshold
