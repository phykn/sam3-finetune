import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tvf, InterpolationMode


def convert_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("numpy image must have shape HxWx3")

    if np.issubdtype(image.dtype, np.floating):
        if image.size > 0 and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255)
    return Image.fromarray(image.astype(np.uint8, copy=False), mode="RGB")


def to_tensor(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(image)
    return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).sub(0.5).div(0.5)


def resize_image(image: Image.Image | np.ndarray, size: int) -> np.ndarray:
    image = tvf.resize(
        convert_rgb(image),
        [size, size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return np.array(image, dtype=np.uint8, copy=True)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    single = mask.ndim == 2
    if single:
        mask = mask[..., None]
    if mask.ndim != 3:
        raise ValueError("mask must have shape HxW or HxWxC")

    tensor = torch.from_numpy(np.ascontiguousarray(mask)).permute(2, 0, 1)
    tensor = tvf.resize(
        tensor,
        [size, size],
        interpolation=InterpolationMode.NEAREST,
    )
    out = tensor.permute(1, 2, 0).numpy()
    return out[..., 0] if single else out


def make_tensor(
    image: Image.Image | np.ndarray,
    size: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    image = convert_rgb(image)
    width, height = image.size
    tensor = to_tensor(resize_image(image, size))
    return tensor.unsqueeze(0).to(device), (height, width)
