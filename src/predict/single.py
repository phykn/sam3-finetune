from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from ..data import image as image_data, prompt
from ..ml.model import Sam3ImageModel
from .mask import format as mask_format


class SinglePredictor:
    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.image_size = 1008
        self.model = model.to(self.device).eval()
        self._image_pe = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        config: dict | None = None,
    ) -> "SinglePredictor":
        config = {} if config is None else config
        return cls(Sam3ImageModel(path=path), device=config.get("device", "cuda"))

    def autocast(self) -> AbstractContextManager:
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def get_image_pe(self) -> torch.Tensor:
        if self._image_pe is None or self._image_pe.device != self.device:
            self._image_pe = self.model.image_pe(self.device)
        return self._image_pe

    def _merge_prompt(
        self,
        first: tuple[torch.Tensor, torch.Tensor] | None,
        second: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if first is None:
            return second
        if second is None:
            return first
        return torch.cat([first[0], second[0]], dim=1), torch.cat(
            [first[1], second[1]],
            dim=1,
        )

    def _make_dummy_prompt(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, 1, 2, device=self.device),
            -torch.ones(batch_size, 1, dtype=torch.int, device=self.device),
        )

    def _make_prompt(
        self,
        embed: dict[str, object],
        point_coords: np.ndarray | torch.Tensor | None,
        point_labels: np.ndarray | torch.Tensor | None,
        box: np.ndarray | torch.Tensor | None,
        mask: np.ndarray | torch.Tensor | None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        point_prompt = self._merge_prompt(
            prompt.build_box(box, embed["orig_hw"], self.image_size, self.device),
            prompt.build_points(
                point_coords,
                point_labels,
                embed["orig_hw"],
                self.image_size,
                self.device,
            ),
        )
        mask_prompt = prompt.build_mask(
            mask,
            self.model.mask_input_size,
            self.device,
        )

        if point_prompt is None and mask_prompt is None:
            raise ValueError("prompt is required")
        if point_prompt is None:
            point_prompt = self._make_dummy_prompt(mask_prompt.shape[0])
        return point_prompt, mask_prompt

    def _decode(
        self,
        embed: dict[str, object],
        point_coords: np.ndarray | torch.Tensor | None,
        point_labels: np.ndarray | torch.Tensor | None,
        box: np.ndarray | torch.Tensor | None,
        mask: np.ndarray | torch.Tensor | None,
        multimask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sam_prompt = self._make_prompt(embed, point_coords, point_labels, box, mask)
        with self.autocast():
            encoded_prompt = self.model.encode_prompt(
                points=sam_prompt[0],
                boxes=None,
                masks=sam_prompt[1],
            )
            masks, scores, *_ = self.model.decode_masks(
                embed["image_embed"],
                embed["high_res"],
                encoded_prompt,
                self.get_image_pe(),
                multimask,
                True,
            )
        return masks, scores

    def encode(self, image: Image.Image | np.ndarray) -> dict[str, object]:
        tensor, orig_hw = image_data.make_tensor(image, self.image_size, self.device)
        with self.autocast():
            out = self.model.encode_image(tensor)
        return {
            "image_embed": out["image_embed"],
            "high_res": tuple(out["high_res_features"]),
            "orig_hw": orig_hw,
        }

    @torch.inference_mode()
    def predict_embed_low(
        self,
        embed: dict[str, object],
        point_coords: np.ndarray | torch.Tensor | None = None,
        point_labels: np.ndarray | torch.Tensor | None = None,
        box: np.ndarray | torch.Tensor | None = None,
        mask: np.ndarray | torch.Tensor | None = None,
        multimask: bool = True,
    ) -> dict[str, object]:
        masks, scores = self._decode(
            embed,
            point_coords,
            point_labels,
            box,
            mask,
            multimask,
        )
        return mask_format.make_low(masks, scores, 0.0)

    @torch.inference_mode()
    def predict_embed(
        self,
        embed: dict[str, object],
        point_coords: np.ndarray | torch.Tensor | None = None,
        point_labels: np.ndarray | torch.Tensor | None = None,
        box: np.ndarray | torch.Tensor | None = None,
        mask: np.ndarray | torch.Tensor | None = None,
        multimask: bool = True,
    ) -> dict[str, object]:
        masks, scores = self._decode(
            embed,
            point_coords,
            point_labels,
            box,
            mask,
            multimask,
        )
        return mask_format.make_full(masks, scores, embed["orig_hw"], 0.0)

    @torch.inference_mode()
    def refine_low(
        self,
        embed: dict[str, object],
        logit: np.ndarray | torch.Tensor,
        point_coords: np.ndarray | torch.Tensor | None = None,
        point_labels: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, object]:
        return self.predict_embed_low(
            embed,
            point_coords=point_coords,
            point_labels=point_labels,
            mask=logit,
            multimask=False,
        )

    @torch.inference_mode()
    def refine(
        self,
        image: Image.Image | np.ndarray,
        logit: np.ndarray | torch.Tensor,
    ) -> dict[str, object]:
        return self.predict(image, mask=logit, multimask=False)

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image | np.ndarray,
        point_coords: np.ndarray | torch.Tensor | None = None,
        point_labels: np.ndarray | torch.Tensor | None = None,
        box: np.ndarray | torch.Tensor | None = None,
        mask: np.ndarray | torch.Tensor | None = None,
        multimask: bool = True,
    ) -> dict[str, object]:
        embed = self.encode(image)
        return self.predict_embed(
            embed,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask=mask,
            multimask=multimask,
        )
