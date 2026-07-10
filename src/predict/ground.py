from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from ..data import ground, image as image_data
from ..ml.model import Sam3GroundingModel
from ..ops.box import cxcywh_to_xyxy, nms_indices
from .ground_ops import sim


class GroundPredictor:
    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cuda",
        score_thresh: float = 0.0,
        nms: float = 0.7,
        top_k: int | None = None,
        sim_thr: float = 0.0,
    ) -> None:
        self.device = torch.device(device)
        self.image_size = 1008
        self.score_thresh = float(score_thresh)
        self.nms = float(nms)
        self.top_k = None if top_k is None else int(top_k)
        self.sim_thr = float(sim_thr)
        self.model = model.to(self.device).eval()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        visual_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        score_thresh: float = 0.0,
        nms: float = 0.7,
        top_k: int | None = None,
        sim_thr: float = 0.0,
    ) -> "GroundPredictor":
        model = Sam3GroundingModel(path=path, visual_path=visual_path)
        return cls(
            model,
            device=device,
            score_thresh=score_thresh,
            nms=nms,
            top_k=top_k,
            sim_thr=sim_thr,
        )

    def autocast(self) -> AbstractContextManager:
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def scores(logits: torch.Tensor | np.ndarray) -> np.ndarray:
        scores = torch.as_tensor(logits).float()
        if scores.dim() == 3 and scores.shape[-1] == 1:
            scores = scores[..., 0]
        if scores.dim() == 2:
            scores = scores[0]
        return scores.sigmoid().detach().cpu().numpy()

    @staticmethod
    def boxes(out: dict[str, object], orig_hw: tuple[int, int]) -> np.ndarray:
        raw = out.get("raw", {})
        boxes = raw.get("pred_boxes_xyxy", out["pred_boxes"])
        boxes = torch.as_tensor(boxes).float()
        if boxes.dim() == 3:
            boxes = boxes[0]
        if "pred_boxes_xyxy" not in raw:
            boxes = cxcywh_to_xyxy(boxes)

        height, width = orig_hw
        scale = boxes.new_tensor([width, height, width, height])
        boxes = (boxes * scale).clamp(min=0)
        boxes[:, 0::2].clamp_(max=width)
        boxes[:, 1::2].clamp_(max=height)
        return boxes.detach().cpu().numpy()

    @staticmethod
    def masks(
        mask_logits: torch.Tensor | np.ndarray,
        orig_hw: tuple[int, int],
    ) -> np.ndarray:
        masks = torch.as_tensor(mask_logits).float()
        if masks.dim() == 4:
            masks = masks[0]
        if masks.dim() == 3:
            masks = masks[:, None]
        masks = F.interpolate(
            masks,
            orig_hw,
            mode="bilinear",
            align_corners=False,
        )
        return (masks[:, 0] > 0).detach().cpu().numpy()

    @staticmethod
    def logits(mask_logits: torch.Tensor | np.ndarray) -> np.ndarray:
        masks = torch.as_tensor(mask_logits).float()
        if masks.dim() == 4:
            masks = masks[0]
        return masks.detach().cpu().numpy()

    def _make_prompt(
        self,
        embed: dict[str, object],
        point: object | None = None,
        point_label: object | None = None,
        box: object | None = None,
        mask: object | None = None,
    ) -> dict[str, object]:
        if point is not None and point_label is None:
            point_label = [1]

        points, point_labels = ground.build_points(
            point,
            point_label,
            embed["orig_hw"],
            self.device,
        )
        boxes, box_labels = ground.build_boxes(box, embed["orig_hw"], self.device)
        masks, mask_labels = ground.build_masks(mask, self.device)
        return {
            "points": points,
            "point_labels": point_labels,
            "boxes": boxes,
            "box_labels": box_labels,
            "masks": masks,
            "mask_labels": mask_labels,
        }

    def format(
        self,
        out: dict[str, object],
        orig_hw: tuple[int, int],
        filter_candidates: bool = True,
    ) -> dict[str, object]:
        scores = self.scores(out["pred_logits"])
        boxes = self.boxes(out, orig_hw)
        masks = self.masks(out["pred_masks"], orig_hw)
        logits = self.logits(out["pred_masks"])

        if filter_candidates:
            keep = np.flatnonzero(scores >= self.score_thresh)
            if len(keep) > 0:
                keep = keep[nms_indices(boxes[keep], scores[keep], self.nms)]
        else:
            keep = np.arange(len(scores))

        return {
            "scores": scores[keep],
            "boxes": boxes[keep],
            "masks": masks[keep],
            "logits": logits[keep],
            "raw": out,
        }

    def rerank(
        self,
        image: dict[str, object],
        ref: torch.Tensor,
        out: dict[str, object],
    ) -> dict[str, object]:
        if len(out["masks"]) == 0:
            out["similarities"] = np.zeros(0, dtype=np.float32)
            return out

        target = sim.mask_vectors(image, out["masks"])
        sims = sim.max_scores(ref, target).detach().cpu().numpy()
        keep = sim.select(sims, out["scores"], self.sim_thr, self.top_k)
        out["scores"] = out["scores"][keep]
        out["boxes"] = out["boxes"][keep]
        out["masks"] = out["masks"][keep]
        out["logits"] = out["logits"][keep]
        out["similarities"] = sims[keep]
        return out

    def encode(self, image: Image.Image | np.ndarray) -> dict[str, object]:
        tensor, orig_hw = image_data.make_tensor(image, self.image_size, self.device)
        with self.autocast():
            image = self.model.encode_image(tensor)
        return {
            "image": image,
            "orig_hw": orig_hw,
        }

    @torch.inference_mode()
    def encode_ref(
        self,
        image: Image.Image | np.ndarray,
        point: object | None = None,
        point_label: object | None = None,
        box: object | None = None,
        mask: object | None = None,
        name: str | None = None,
    ) -> dict[str, object]:
        embed = self.encode(image)
        prompt = self._make_prompt(
            embed,
            point=point,
            point_label=point_label,
            box=box,
            mask=mask,
        )
        with self.autocast():
            encoded = self.model.encode_prompt(embed["image"], **prompt)
        out = {"name": "ref" if name is None else str(name), "prompt": encoded}
        if mask is not None:
            out["feature"] = sim.mask_vectors(embed["image"], mask)
        return out

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image | np.ndarray,
        refs: dict[str, object] | list[dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        target = self.encode(image)
        if isinstance(refs, dict):
            refs = [refs]

        out = {}
        for index, ref in enumerate(refs):
            name = ref.get("name", f"ref_{index}")
            with self.autocast():
                raw = self.model.decode(target["image"], ref["prompt"])
            if "feature" in ref:
                full = self.format(raw, target["orig_hw"], filter_candidates=False)
                out[name] = self.rerank(target["image"], ref["feature"], full)
            else:
                out[name] = self.format(raw, target["orig_hw"])
        return out
