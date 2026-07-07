import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_frames(
    path: str | Path,
    image_size: int,
    offload_video_to_cpu: bool,
    device: torch.device | str = "cuda",
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    async_loading_frames: bool = False,
) -> tuple[torch.Tensor, int, int]:
    if async_loading_frames:
        raise NotImplementedError("async frame loading is not implemented in src")

    path = Path(path)
    if not path.is_dir():
        raise NotImplementedError("src video memory currently expects image folders")

    frame_paths = [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    ]

    try:
        frame_paths.sort(key=lambda item: int(item.stem))
    except ValueError:
        frame_paths.sort(key=lambda item: item.name)

    if not frame_paths:
        raise RuntimeError(f"no images found in {os.fspath(path)}")

    frames = []
    video_height = video_width = None

    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        video_width, video_height = image.size
        image = image.resize((image_size, image_size))

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float16)
        frames.append(tensor)

    batch = torch.stack(frames, dim=0)

    mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
    std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]

    device = torch.device(device)
    if not offload_video_to_cpu:
        batch = batch.to(device)
        mean = mean.to(device)
        std = std.to(device)

    batch = (batch - mean) / std

    assert video_height is not None and video_width is not None

    return batch, video_height, video_width
