from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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
        mask_prompt_encoder: torch.nn.Module | None = None,
    ) -> None:
        self.device = torch.device(device)
        if hasattr(model, "grounding"):
            mask_prompt_encoder = mask_prompt_encoder or getattr(
                getattr(model, "video", None),
                "maskmem_backbone",
                None,
            )
            model = model.grounding
        self.model = model.to(self.device).eval()
        if mask_prompt_encoder is not None:
            mask_prompt_encoder = mask_prompt_encoder.to(self.device).eval()
        self.confidence_threshold = float(confidence_threshold)
        self.encoder = VisualPromptEncoder(
            self.model,
            device=self.device,
            image_size=image_size,
            mask_prompt_encoder=mask_prompt_encoder,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cuda",
        image_size: int = 1008,
        confidence_threshold: float = 0.5,
    ) -> "VisualPromptPredictor":
        model = build_model(path, device=device)
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

        image_tensor, original_hw = self.encoder.preprocess_image(target_image)
        backbone_out = self.model.backbone.forward_image(image_tensor)
        find_stage = FindStage(
            img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            text_ids=torch.empty(0, device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        prompt = self._dummy_prompt()
        threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )

        predictions: list[VisualPromptPrediction] = []
        for concept in prepared.concepts:
            outputs = self.model.forward_grounding(
                backbone_out=dict(backbone_out),
                find_input=find_stage,
                geometric_prompt=prompt.clone(),
                visual_prompt_embed=concept.visual_prompt_embed,
                visual_prompt_mask=concept.visual_prompt_mask,
                encode_text=False,
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

    def _dummy_prompt(self) -> Prompt:
        if hasattr(self.model, "get_dummy_prompt"):
            return self.model.get_dummy_prompt()
        return Prompt(
            box_embeddings=torch.zeros(0, 1, 4, device=self.device),
            box_mask=torch.zeros(1, 0, dtype=torch.bool, device=self.device),
        )


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
