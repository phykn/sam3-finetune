
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False,
):
    if async_loading_frames:
        raise NotImplementedError("async frame loading is not implemented in src")
    path = Path(video_path)
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

    images = []
    video_height = video_width = None
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        video_width, video_height = image.size
        image = image.resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float16)
        images.append(tensor)

    batch = torch.stack(images, dim=0)
    mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
    std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
    if not offload_video_to_cpu:
        batch = batch.cuda()
        mean = mean.cuda()
        std = std.cuda()
    batch = (batch - mean) / std
    assert video_height is not None and video_width is not None
    return batch, video_height, video_width
