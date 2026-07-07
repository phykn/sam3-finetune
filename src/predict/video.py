from contextlib import nullcontext

import torch

from .video_ops import session


class VideoPredictor:
    def __init__(self, model, config: dict | None = None) -> None:
        config = {} if config is None else config
        self.device = torch.device(config.get("device", "cuda"))
        self.mask_threshold = float(config.get("mask_threshold", 0.0))
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def start(self, image, mask, obj_id: int = 0):
        with self.autocast():
            return session.start(self.model, image, mask, obj_id, self.device)

    @torch.inference_mode()
    def predict(self, image, state):
        with self.autocast():
            return session.predict(
                self.model,
                state,
                image,
                self.device,
                self.mask_threshold,
            )

    def autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()
