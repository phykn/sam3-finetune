from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from ..data import ground, image as image_data
from ..ml.model import Sam3GroundingModel
from .ground_ops import output, reference, sim


class GroundPredictor:
    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cuda",
        score_thr: float = 0.0,
        nms_thr: float = 0.7,
        top_k: int | None = None,
        sim_thr: float = 0.0,
        prompt_batch_size: int = 4,
    ) -> None:
        if not 0 <= score_thr <= 1:
            raise ValueError("score_thr must be between zero and one")
        if not 0 <= nms_thr <= 1:
            raise ValueError("nms_thr must be between zero and one")
        if not -1 <= sim_thr <= 1:
            raise ValueError("sim_thr must be between minus one and one")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None")
        if prompt_batch_size <= 0:
            raise ValueError("prompt_batch_size must be positive")

        self.device = torch.device(device)
        self.image_size = 1008
        self.score_thr = float(score_thr)
        self.nms_thr = float(nms_thr)
        self.top_k = None if top_k is None else int(top_k)
        self.sim_thr = float(sim_thr)
        self.prompt_batch_size = int(prompt_batch_size)
        self.model = model.to(self.device).eval()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        visual_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        score_thr: float = 0.0,
        nms_thr: float = 0.7,
        top_k: int | None = None,
        sim_thr: float = 0.0,
        prompt_batch_size: int = 4,
    ) -> "GroundPredictor":
        model = Sam3GroundingModel(path=path, visual_path=visual_path)
        return cls(
            model,
            device=device,
            score_thr=score_thr,
            nms_thr=nms_thr,
            top_k=top_k,
            sim_thr=sim_thr,
            prompt_batch_size=prompt_batch_size,
        )

    def autocast(self) -> AbstractContextManager:
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def encode(self, image: Image.Image | np.ndarray) -> dict[str, object]:
        tensor, orig_hw = image_data.make_tensor(image, self.image_size, self.device)
        with self.autocast():
            encoded = self.model.encode_image(tensor)
        return {"image": encoded, "orig_hw": orig_hw}

    @torch.inference_mode()
    def encode_reference(
        self,
        image: Image.Image | np.ndarray,
        boxes: object,
        class_ids: object,
    ) -> dict[str, object]:
        encoded = self.encode(image)
        boxes, class_ids = reference.validate(
            boxes,
            class_ids,
            encoded["orig_hw"],
        )
        features = sim.box_vectors(encoded["image"], boxes, encoded["orig_hw"])
        prompt_classes, grouped = reference.groups(boxes, class_ids)
        box_batch, labels, box_mask = ground.build_box_batch(
            grouped,
            encoded["orig_hw"],
            self.device,
        )
        with self.autocast():
            prompt = self.model.encode_box_prompts(
                encoded["image"],
                box_batch,
                labels,
                box_mask,
            )
        return {
            "prompt": {
                "features": prompt["features"],
                "mask": prompt["mask"],
            },
            "prompt_classes": prompt_classes,
            "features": features,
            "feature_classes": class_ids,
        }

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image | np.ndarray,
        references: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not isinstance(references, list):
            raise TypeError("references must be a list")
        if not references:
            raise ValueError("references list is empty")

        bank = reference.feature_bank(references)
        prompts, class_ids = reference.prompt_groups(references)
        target = self.encode(image)
        items = []
        for start in range(0, len(class_ids), self.prompt_batch_size):
            end = start + self.prompt_batch_size
            prompt = {
                "features": prompts["features"][:, start:end],
                "mask": prompts["mask"][start:end],
            }
            with self.autocast():
                decoded = self.model.decode(target["image"], prompt)
            items.extend(
                output.candidates(
                    decoded,
                    target["image"],
                    class_ids[start:end],
                    bank,
                    target["orig_hw"],
                    self.score_thr,
                    self.sim_thr,
                )
            )
        return output.finish(items, self.nms_thr, self.top_k)
