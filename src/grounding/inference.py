from pathlib import Path
from typing import Sequence

import numpy as np
import PIL.Image
import torch
from torchvision.transforms import v2

from ..ops.box import box_cxcywh_to_xyxy
from ..data.structures import FindStage, interpolate
from .builder import build_grounding_model
from .cache import VisualLanguageCache
from .geometry import Prompt
from .types import GroundingPrediction


class GroundingInference:
    def __init__(
        self,
        model: torch.nn.Module,
        visual_cache: VisualLanguageCache,
        *,
        device: torch.device | str = "cuda",
        confidence_threshold: float = 0.5,
        load_report=None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.visual_cache = visual_cache
        self.confidence_threshold = confidence_threshold
        self.load_report = load_report
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(1008, 1008)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        visual_cache_path: str | Path,
        *,
        device: torch.device | str = "cuda",
        confidence_threshold: float = 0.5,
    ) -> "GroundingInference":
        model, report = build_grounding_model(checkpoint_path, device=device)
        visual_cache = VisualLanguageCache.from_file(visual_cache_path)
        return cls(
            model=model,
            visual_cache=visual_cache,
            device=device,
            confidence_threshold=confidence_threshold,
            load_report=report,
        )

    @torch.inference_mode()
    def predict(
        self,
        image: PIL.Image.Image | np.ndarray | torch.Tensor,
        *,
        boxes_xyxy: np.ndarray | Sequence[Sequence[float]] | None = None,
        box_labels: Sequence[bool] | None = None,
        point_coords: np.ndarray | Sequence[Sequence[float]] | None = None,
        point_labels: Sequence[int] | None = None,
        confidence_threshold: float | None = None,
    ) -> GroundingPrediction:
        image_tensor, original_hw = self._preprocess_image(image)
        backbone_out = self.model.backbone.forward_image(image_tensor)
        backbone_out.update(
            self.visual_cache.to_backbone_out(
                device=self.device,
                dtype=backbone_out["vision_features"].dtype,
            )
        )
        prompt = self._make_prompt(
            boxes_xyxy=boxes_xyxy,
            box_labels=box_labels,
            point_coords=point_coords,
            point_labels=point_labels,
            original_hw=original_hw,
        )
        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=self.find_stage,
            geometric_prompt=prompt,
        )
        threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        return self._format_outputs(outputs, original_hw, threshold)

    def _preprocess_image(
        self,
        image: PIL.Image.Image | np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, np.ndarray):
            height, width = image.shape[-3:-1] if image.ndim == 3 else image.shape[-2:]
        elif isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image, NumPy array, or tensor")

        image_tensor = v2.functional.to_image(image).to(self.device)
        image_tensor = self.transform(image_tensor).unsqueeze(0)
        return image_tensor, (height, width)

    def _make_prompt(
        self,
        *,
        boxes_xyxy,
        box_labels,
        point_coords,
        point_labels,
        original_hw: tuple[int, int],
    ) -> Prompt:
        if boxes_xyxy is None and point_coords is None:
            raise ValueError("Provide at least one box or point prompt")
        prompt = self.model.get_dummy_prompt()
        if boxes_xyxy is not None:
            boxes = self._boxes_xyxy_to_norm_cxcywh(boxes_xyxy, original_hw)
            labels = [True] * boxes.shape[0] if box_labels is None else list(box_labels)
            if len(labels) != boxes.shape[0]:
                raise ValueError("box_labels length must match boxes_xyxy")
            prompt.append_boxes(
                boxes.to(self.device).view(boxes.shape[0], 1, 4),
                torch.as_tensor(labels, dtype=torch.bool, device=self.device).view(
                    boxes.shape[0], 1
                ),
            )
        if point_coords is not None:
            if point_labels is None:
                raise ValueError("point_labels must be supplied with point_coords")
            points = self._points_to_norm_xy(point_coords, original_hw)
            labels = torch.as_tensor(point_labels, dtype=torch.long, device=self.device)
            if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
                raise ValueError("point_labels length must match point_coords")
            prompt.append_points(
                points.to(self.device).view(points.shape[0], 1, 2),
                labels.view(points.shape[0], 1),
            )
        return prompt

    def _boxes_xyxy_to_norm_cxcywh(self, boxes_xyxy, original_hw) -> torch.Tensor:
        boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
        if boxes.shape[-1] != 4:
            raise ValueError("boxes_xyxy must end with four values")
        if boxes.ndim == 1:
            boxes = boxes.view(1, 4)
        h, w = original_hw
        x0, y0, x1, y1 = boxes.unbind(-1)
        cx = (x0 + x1) * 0.5 / float(w)
        cy = (y0 + y1) * 0.5 / float(h)
        bw = (x1 - x0) / float(w)
        bh = (y1 - y0) / float(h)
        return torch.stack([cx, cy, bw, bh], dim=-1)

    def _points_to_norm_xy(self, point_coords, original_hw) -> torch.Tensor:
        points = torch.as_tensor(point_coords, dtype=torch.float32)
        if points.shape[-1] != 2:
            raise ValueError("point_coords must end with two values")
        if points.ndim == 1:
            points = points.view(1, 2)
        h, w = original_hw
        points = points.clone()
        points[..., 0] /= float(w)
        points[..., 1] /= float(h)
        return points

    def _format_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        original_hw: tuple[int, int],
        confidence_threshold: float,
    ) -> GroundingPrediction:
        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        if "presence_logit_dec" in outputs:
            presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            out_probs = out_probs * presence_score
        out_probs = out_probs.squeeze(-1)

        keep = out_probs > confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = original_hw
        scale_fct = torch.tensor(
            [img_w, img_h, img_w, img_h],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        boxes = boxes * scale_fct[None, :]

        mask_logits = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        masks = mask_logits > 0.5
        return GroundingPrediction(
            masks=masks.squeeze(1).detach().cpu().numpy(),
            mask_logits=mask_logits.squeeze(1).detach().cpu().numpy(),
            boxes_xyxy=boxes.detach().float().cpu().numpy(),
            scores=out_probs.detach().float().cpu().numpy(),
        )
