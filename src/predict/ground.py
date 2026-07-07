from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from ..data import ground, image as image_data
from ..ops.box import convert_to_xyxy, filter_boxes
from .ground_ops import sim


class GroundPredictor:
    def __init__(self, model, config: dict | None = None) -> None:
        config = {} if config is None else config
        self.device = torch.device(config.get("device", "cuda"))
        self.image_size = int(config.get("image_size", 1008))
        self.score_thresh = float(config.get("score_thresh", 0.0))
        self.nms = float(config.get("nms", 0.7))
        self.max_masks = int(config.get("max_masks", 50))
        top_k = config.get("top_k", None)
        self.top_k = None if top_k is None else int(top_k)
        self.sim_thr = float(config.get("sim_thr", 0.0))
        self.model = model.to(self.device).eval()

    def encode(self, image):
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
        image,
        point=None,
        point_label=None,
        box=None,
        mask=None,
        name=None,
    ):
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
            out["feature"] = sim.vectors(embed["image"], mask)
        return out

    @torch.inference_mode()
    def predict(self, image, refs):
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

    def _make_prompt(self, embed, point=None, point_label=None, box=None, mask=None):
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

    def autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def rerank(self, image, ref, out):
        if len(out["masks"]) == 0:
            out["similarities"] = np.zeros(0, dtype=np.float32)
            return out

        target = sim.vectors(image, out["masks"])
        sims = sim.scores(ref, target).detach().cpu().numpy()
        keep = sim.select(sims, out["scores"], self.sim_thr, self.top_k)
        out["scores"] = out["scores"][keep]
        out["boxes"] = out["boxes"][keep]
        out["masks"] = out["masks"][keep]
        out["logits"] = out["logits"][keep]
        out["similarities"] = sims[keep]
        return out

    def format(self, out, orig_hw: tuple[int, int], filter_candidates=True):
        scores = self.scores(out["pred_logits"])
        boxes = self.boxes(out, orig_hw)
        masks = self.masks(out["pred_masks"], orig_hw)
        logits = self.logits(out["pred_masks"])

        if filter_candidates:
            keep = np.flatnonzero(scores >= self.score_thresh)
            if len(keep) > 0:
                keep = keep[filter_boxes(boxes[keep], scores[keep], self.nms)]
            keep = keep[: self.max_masks]
        else:
            keep = np.arange(len(scores))

        return {
            "scores": scores[keep],
            "boxes": boxes[keep],
            "masks": masks[keep],
            "logits": logits[keep],
            "raw": out,
        }

    @staticmethod
    def scores(logits):
        scores = torch.as_tensor(logits).float()
        if scores.dim() == 3 and scores.shape[-1] == 1:
            scores = scores[..., 0]
        if scores.dim() == 2:
            scores = scores[0]
        return scores.sigmoid().detach().cpu().numpy()

    @staticmethod
    def boxes(out, orig_hw: tuple[int, int]):
        raw = out.get("raw", {})
        boxes = raw.get("pred_boxes_xyxy", out["pred_boxes"])
        boxes = torch.as_tensor(boxes).float()
        if boxes.dim() == 3:
            boxes = boxes[0]
        if "pred_boxes_xyxy" not in raw:
            boxes = convert_to_xyxy(boxes)

        height, width = orig_hw
        scale = boxes.new_tensor([width, height, width, height])
        boxes = (boxes * scale).clamp(min=0)
        boxes[:, 0::2].clamp_(max=width)
        boxes[:, 1::2].clamp_(max=height)
        return boxes.detach().cpu().numpy()

    @staticmethod
    def masks(masks, orig_hw: tuple[int, int]):
        masks = torch.as_tensor(masks).float()
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
    def logits(masks):
        masks = torch.as_tensor(masks).float()
        if masks.dim() == 4:
            masks = masks[0]
        return masks.detach().cpu().numpy()
