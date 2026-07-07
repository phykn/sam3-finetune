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


def make_tensor(
    image: Image.Image | np.ndarray,
    size: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    image = convert_rgb(image)
    width, height = image.size
    tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
    tensor = tensor.permute(2, 0, 1).float().div(255.0)
    tensor = tvf.resize(
        tensor,
        [size, size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = tensor.sub(0.5).div(0.5)
    return tensor.unsqueeze(0).to(device), (height, width)
