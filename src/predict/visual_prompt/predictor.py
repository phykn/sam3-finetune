from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from ...model.build import build_model
from ...model.grounding.prompt import Prompt
from ...model.structures import FindStage
from ...ops.box import convert_to_xyxy
from ...ops.tensor import interpolate
from .encoder import VisualPromptEncoder
from .types import PreparedVisualPrompts, VisualExemplar, VisualPromptPrediction


class VisualPromptPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | str = "cuda",
        image_size: int = 1008,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        if hasattr(model, "grounding"):
            model = model.grounding
        self.model = model.to(self.device).eval()
        self.confidence_threshold = float(confidence_threshold)
        self.encoder = VisualPromptEncoder(
            self.model,
            device=self.device,
            image_size=image_size,
        )
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(int(image_size), int(image_size))),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cuda",
        image_size: int = 1008,
        confidence_threshold: float = 0.5,
        bpe_path: str | Path | None = None,
    ) -> "VisualPromptPredictor":
        model = build_model(
            path,
            device=device,
            include_language=True,
            bpe_path=bpe_path,
        )
        return cls(
            model,
            device=device,
            image_size=image_size,
            confidence_threshold=confidence_threshold,
        )

    @torch.inference_mode()
    def prepare_exemplars(
        self,
        exemplars: Sequence[VisualExemplar],
    ) -> PreparedVisualPrompts:
        return self.encoder.prepare(exemplars)

    @torch.inference_mode()
    def predict(
        self,
        target_image: Image.Image | np.ndarray | torch.Tensor,
        exemplars: Sequence[VisualExemplar] | PreparedVisualPrompts,
        *,
        confidence_threshold: float | None = None,
    ) -> list[VisualPromptPrediction]:
        if isinstance(exemplars, PreparedVisualPrompts):
            prepared = exemplars
        else:
            prepared = self.prepare_exemplars(exemplars)
        if not prepared.concepts:
            raise ValueError("exemplars must be non-empty")

        image_tensor, original_hw = self._preprocess_image(target_image)
        image_backbone_out = self.model.backbone.forward_image(image_tensor)
        threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )

        predictions: list[VisualPromptPrediction] = []
        for concept in prepared.concepts:
            backbone_out = dict(image_backbone_out)
            backbone_out.update(
                {
                    "language_features": concept.language_features.to(
                        device=self.device,
                        dtype=backbone_out["vision_features"].dtype,
                    ),
                    "language_mask": concept.language_mask.to(
                        device=self.device,
                        dtype=torch.bool,
                    ),
                }
            )
            if concept.language_embeds is not None:
                backbone_out["language_embeds"] = concept.language_embeds.to(
                    device=self.device,
                    dtype=backbone_out["vision_features"].dtype,
                )
            outputs = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=FindStage(
                    img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
                    text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
                    input_boxes=None,
                    input_boxes_mask=None,
                    input_boxes_label=None,
                    input_points=None,
                    input_points_mask=None,
                ),
                geometric_prompt=Prompt(
                    box_embeddings=concept.boxes_cxcywh,
                    box_mask=torch.zeros(
                        1,
                        concept.boxes_cxcywh.shape[0],
                        device=self.device,
                        dtype=torch.bool,
                    ),
                    box_labels=concept.box_labels,
                ),
            )
            predictions.append(
                _format_outputs(
                    outputs,
                    original_hw,
                    threshold,
                    concept_id=concept.concept_id,
                )
            )
        return predictions

    def _preprocess_image(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, np.ndarray):
            height, width = image.shape[-3:-1] if image.ndim == 3 else image.shape[-2:]
        elif isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        else:
            raise TypeError("image must be a PIL image, NumPy array, or tensor")

        image_tensor = v2.functional.to_image(image).to(self.device)
        image_tensor = self.transform(image_tensor).unsqueeze(0)
        return image_tensor, (int(height), int(width))


def _format_outputs(
    outputs: dict[str, torch.Tensor],
    original_hw: tuple[int, int],
    confidence_threshold: float,
    *,
    concept_id: int,
) -> VisualPromptPrediction:
    boxes = outputs["pred_boxes"]
    logits = outputs["pred_logits"]
    masks = outputs["pred_masks"]
    probs = logits.sigmoid()
    if "presence_logit_dec" in outputs:
        probs = probs * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
    probs = probs.squeeze(-1)

    keep = probs > float(confidence_threshold)
    probs = probs[keep]
    masks = masks[keep]
    boxes = boxes[keep]

    boxes_xyxy = convert_to_xyxy(boxes)
    image_h, image_w = original_hw
    scale = torch.tensor(
        [image_w, image_h, image_w, image_h],
        dtype=boxes_xyxy.dtype,
        device=boxes_xyxy.device,
    )
    boxes_xyxy = boxes_xyxy * scale[None, :]

    mask_logits = interpolate(
        masks.unsqueeze(1),
        (image_h, image_w),
        mode="bilinear",
        align_corners=False,
    ).sigmoid()
    return VisualPromptPrediction(
        concept_id=int(concept_id),
        masks=(mask_logits > 0.5).squeeze(1).detach().cpu().numpy(),
        boxes_xyxy=boxes_xyxy.detach().float().cpu().numpy(),
        scores=probs.detach().float().cpu().numpy(),
    )
