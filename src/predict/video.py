from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from ..ml.model import Sam3VideoModel
from .video_ops import session


class VideoPredictor:
    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        config: dict | None = None,
    ) -> "VideoPredictor":
        config = {} if config is None else config
        return cls(Sam3VideoModel(path=path), device=config.get("device", "cuda"))

    def autocast(self) -> AbstractContextManager:
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.inference_mode()
    def start(
        self,
        image: Image.Image | np.ndarray,
        mask: np.ndarray | torch.Tensor,
        obj_id: int = 0,
    ) -> dict[str, object]:
        with self.autocast():
            return session.start(self.model, image, mask, obj_id, self.device)

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image | np.ndarray,
        state: dict[str, object],
    ) -> dict[str, object]:
        with self.autocast():
            return session.predict(
                self.model,
                state,
                image,
                self.device,
                0.0,
            )

    @torch.inference_mode()
    def add_masks(
        self,
        state: dict[str, object],
        masks: np.ndarray | torch.Tensor,
        obj_ids: list[int],
        frame_idx: int | None = None,
    ) -> list[int]:
        with self.autocast():
            return session.add_masks(
                self.model,
                state,
                masks,
                obj_ids,
                self.device,
                frame_idx,
            )

    @torch.inference_mode()
    def remove_objects(
        self,
        state: dict[str, object],
        obj_ids: list[int],
        strict: bool = True,
    ) -> list[int]:
        return session.remove_objects(self.model, state, obj_ids, strict)
