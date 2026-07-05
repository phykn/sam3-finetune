import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def to_uint8_rgb_array(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("NumPy images must have shape HxWx3")

    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32, copy=False)
        if image.size > 0 and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255)

    return image.astype(np.uint8, copy=False)


def to_rgb_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(to_uint8_rgb_array(image), mode="RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def preprocess_rgb_image(
    image: Image.Image | np.ndarray,
    *,
    resolution: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    image_rgb = to_rgb_pil(image)
    width, height = image_rgb.size
    resized = image_rgb.resize((resolution, resolution), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(dtype=dtype)
    mean = torch.tensor((0.5, 0.5, 0.5), dtype=dtype)[:, None, None]
    std = torch.tensor((0.5, 0.5, 0.5), dtype=dtype)[:, None, None]
    tensor = (tensor - mean) / std
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor, (height, width)


def preprocess_rgb_images(
    images: list[Image.Image | np.ndarray] | tuple[Image.Image | np.ndarray, ...],
    *,
    resolution: int,
    output_image_index: int = -1,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, tuple[int, int], list[tuple[int, int]]]:
    if not images:
        raise ValueError("images must be non-empty")
    if output_image_index < 0:
        output_image_index = len(images) + output_image_index
    if output_image_index < 0 or output_image_index >= len(images):
        raise IndexError("output_image_index is out of range")

    tensors = []
    frame_hws = []
    for image in images:
        tensor, frame_hw = preprocess_rgb_image(
            image,
            resolution=resolution,
            dtype=dtype,
            device=device,
        )
        tensors.append(tensor)
        frame_hws.append(frame_hw)

    return torch.stack(tensors, dim=0), frame_hws[output_image_index], frame_hws


def scale_coords(
    coords: np.ndarray | torch.Tensor,
    orig_hw: tuple[int, int],
    resolution: int,
) -> torch.Tensor:
    coords_t = torch.as_tensor(coords, dtype=torch.float32).clone()
    if coords_t.shape[-1] != 2:
        raise ValueError("Point coordinates must end with dimension 2")
    height, width = orig_hw
    scale = torch.tensor(
        [resolution / float(width), resolution / float(height)],
        dtype=coords_t.dtype,
        device=coords_t.device,
    )
    return coords_t * scale


def scale_box(
    box: np.ndarray | torch.Tensor,
    orig_hw: tuple[int, int],
    resolution: int,
) -> torch.Tensor:
    box_t = torch.as_tensor(box, dtype=torch.float32)
    if box_t.shape[-1] != 4:
        raise ValueError("Boxes must end with four values: x0, y0, x1, y1")
    if box_t.ndim == 1:
        box_t = box_t.reshape(1, 2, 2)
    else:
        box_t = box_t.reshape(*box_t.shape[:-1], 2, 2)
    return scale_coords(box_t, orig_hw, resolution)


class ImageTransforms:
    def __init__(self, resolution: int = 1008, mask_threshold: float = 0.0) -> None:
        self.resolution = resolution
        self.mask_threshold = mask_threshold

    def preprocess_image(
        self,
        image: Image.Image | np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        tensor, orig_hw = preprocess_rgb_image(
            image,
            resolution=self.resolution,
            dtype=torch.float32,
            device=device,
        )
        return tensor.unsqueeze(0), orig_hw

    def transform_coords(
        self,
        coords: np.ndarray | torch.Tensor,
        orig_hw: tuple[int, int],
    ) -> torch.Tensor:
        return scale_coords(coords, orig_hw, self.resolution)

    def transform_box(
        self,
        box: np.ndarray | torch.Tensor,
        orig_hw: tuple[int, int],
    ) -> torch.Tensor:
        return scale_box(box, orig_hw, self.resolution)

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        orig_hw: tuple[int, int],
        return_logits: bool = False,
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks.float(), orig_hw, mode="bilinear", align_corners=False
        )
        if return_logits:
            return masks
        return masks > self.mask_threshold
