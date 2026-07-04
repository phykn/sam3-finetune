from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .builder import build_model
from .checkpoint import LoadReport
from .transforms import Sam3Transforms


class Sam3Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cuda",
        load_report: LoadReport | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.transforms = Sam3Transforms(resolution=1008, mask_threshold=0.0)
        self.load_report = load_report
        self._features: dict[str, object] | None = None
        self._orig_hw: tuple[int, int] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str = "cuda",
    ) -> "Sam3Predictor":
        model, report = build_model(str(checkpoint_path), device=device)
        return cls(model=model, device=device, load_report=report)

    @torch.inference_mode()
    def set_image(self, image: Image.Image | np.ndarray) -> None:
        input_tensor, orig_hw = self.transforms.preprocess_image(image, self.device)
        self._features = self.model.encode_image(input_tensor)
        self._orig_hw = orig_hw

    @torch.inference_mode()
    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._features is None or self._orig_hw is None:
            raise RuntimeError("Call set_image() before predict().")

        concat_points = None
        if point_coords is not None:
            if point_labels is None:
                raise ValueError("point_labels must be supplied with point_coords")
            coords = self.transforms.transform_coords(point_coords, self._orig_hw).to(
                self.device
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if coords.ndim == 2:
                coords = coords[None, ...]
                labels = labels[None, ...]
            concat_points = (coords, labels)

        if box is not None:
            box_coords = self.transforms.transform_box(box, self._orig_hw).to(self.device)
            box_labels = torch.tensor([2, 3], dtype=torch.int, device=self.device)
            box_labels = box_labels.expand(box_coords.shape[0], 2)
            if concat_points is None:
                concat_points = (box_coords, box_labels)
            else:
                concat_points = (
                    torch.cat([box_coords, concat_points[0]], dim=1),
                    torch.cat([box_labels, concat_points[1]], dim=1),
                )

        mask_prompt = None
        if mask_input is not None:
            mask_prompt = torch.as_tensor(
                mask_input,
                dtype=torch.float32,
                device=self.device,
            )
            if mask_prompt.ndim == 2:
                mask_prompt = mask_prompt[None, None, :, :]
            elif mask_prompt.ndim == 3:
                mask_prompt = mask_prompt[:, None, :, :]
            elif mask_prompt.ndim != 4:
                raise ValueError("mask_input must have 2, 3, or 4 dimensions")
            if mask_prompt.shape[-2:] != self.model.prompt_encoder.mask_input_size:
                mask_prompt = F.interpolate(
                    mask_prompt,
                    size=self.model.prompt_encoder.mask_input_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )

        if concat_points is None and mask_prompt is None:
            raise ValueError("Provide at least one point, box, or mask prompt.")
        if concat_points is None and mask_prompt is not None:
            concat_points = (
                torch.zeros(mask_prompt.shape[0], 1, 2, device=self.device),
                -torch.ones(
                    mask_prompt.shape[0],
                    1,
                    dtype=torch.int,
                    device=self.device,
                ),
            )

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_prompt,
        )
        low_res_masks, iou_predictions, _tokens, _obj_scores = self.model.mask_decoder(
            image_embeddings=self._features["image_embed"],
            image_pe=self.model.prompt_encoder.get_dense_pe().to(self.device),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=True,
            high_res_features=self._features["high_res_features"],
        )
        masks = self.transforms.postprocess_masks(
            low_res_masks,
            self._orig_hw,
            return_logits=return_logits,
        )
        return (
            masks.squeeze(0).detach().cpu().numpy(),
            iou_predictions.squeeze(0).float().detach().cpu().numpy(),
            torch.clamp(low_res_masks, -32.0, 32.0)
            .squeeze(0)
            .float()
            .detach()
            .cpu()
            .numpy(),
        )
