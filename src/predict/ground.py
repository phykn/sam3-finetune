from contextlib import AbstractContextManager, nullcontext
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from ..data import ground, image as image_data, pack, prompt as prompt_data
from ..ml.model import Sam3GroundingModel
from .ground_ops import output, reference, sim
from .mask import format as mask_format
from .mask.component import largest


class GroundPredictor:
    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cuda",
        score_thr: float = 0.0,
        nms_thr: float = 0.7,
        top_k: int | None = None,
        sim_thr: float = 0.0,
        negative_margin: float = 0.0,
        prompt_batch_size: int = 4,
        use_sam_masks: bool = False,
        largest_component: bool = False,
    ) -> None:
        if not 0 <= score_thr <= 1:
            raise ValueError("score_thr must be between zero and one")
        if not 0 <= nms_thr <= 1:
            raise ValueError("nms_thr must be between zero and one")
        if not -1 <= sim_thr <= 1:
            raise ValueError("sim_thr must be between minus one and one")
        if not 0 <= negative_margin <= 2:
            raise ValueError("negative_margin must be between zero and two")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None")
        if (
            isinstance(prompt_batch_size, bool)
            or not isinstance(prompt_batch_size, Integral)
            or prompt_batch_size <= 0
        ):
            raise ValueError("prompt_batch_size must be a positive integer")
        self.device = torch.device(device)
        self.image_size = 1008
        self.score_thr = float(score_thr)
        self.nms_thr = float(nms_thr)
        self.top_k = None if top_k is None else int(top_k)
        self.sim_thr = float(sim_thr)
        self.negative_margin = float(negative_margin)
        self.prompt_batch_size = int(prompt_batch_size)
        self.use_sam_masks = bool(use_sam_masks)
        self.largest_component = bool(largest_component)
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
        negative_margin: float = 0.0,
        prompt_batch_size: int = 4,
        use_sam_masks: bool = True,
        largest_component: bool = True,
    ) -> "GroundPredictor":
        model = Sam3GroundingModel(
            path=path,
            visual_path=visual_path,
            use_sam_masks=use_sam_masks,
        )
        return cls(
            model,
            device=device,
            score_thr=score_thr,
            nms_thr=nms_thr,
            top_k=top_k,
            sim_thr=sim_thr,
            negative_margin=negative_margin,
            prompt_batch_size=prompt_batch_size,
            use_sam_masks=use_sam_masks,
            largest_component=largest_component,
        )

    def autocast(self) -> AbstractContextManager:
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.inference_mode()
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
        box_labels: object | None = None,
    ) -> dict[str, object]:
        return self.encode_reference_embed(
            self.encode(image),
            boxes,
            class_ids,
            box_labels,
        )

    @torch.inference_mode()
    def encode_reference_embed(
        self,
        encoded: dict[str, object],
        boxes: object,
        class_ids: object,
        box_labels: object | None = None,
    ) -> dict[str, object]:
        boxes, class_ids = reference.validate(
            boxes,
            class_ids,
            encoded["orig_hw"],
        )
        box_labels = reference.validate_labels(box_labels, len(boxes))
        positive = box_labels == 1
        if set(class_ids) != set(class_ids[positive]):
            raise ValueError("each class needs a positive reference box")
        features = sim.box_vectors(encoded["image"], boxes, encoded["orig_hw"])
        prompt_classes, grouped = reference.groups(boxes, class_ids)
        _, label_groups = reference.groups(box_labels, class_ids)
        box_batch, labels, box_mask = ground.build_box_batch(
            grouped,
            encoded["orig_hw"],
            self.device,
            label_groups,
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
            "feature_labels": box_labels,
        }

    @torch.inference_mode()
    def start(self, image: Image.Image | np.ndarray) -> dict[str, object]:
        return {
            "target": self.encode(image),
            "boxes": [],
            "box_labels": [],
            "points": [],
        }

    @torch.inference_mode()
    def add_prompt(
        self,
        state: dict[str, object],
        box: object,
        positive: bool = True,
    ) -> list[dict[str, object]]:
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean")
        if not positive and not any(state["box_labels"]):
            raise ValueError("add a positive box before a negative box")
        boxes, _ = reference.validate([box], [0], state["target"]["orig_hw"])
        previous = self._copy_prompts(state)
        state["boxes"].append(boxes[0])
        state["box_labels"].append(int(positive))
        state["points"].append(None)
        return self._predict_prompt_change(state, previous)

    @torch.inference_mode()
    def add_point(
        self,
        state: dict[str, object],
        point: object,
        positive: bool = True,
    ) -> list[dict[str, object]]:
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean")
        if not positive and not any(state["box_labels"]):
            raise ValueError("add a positive point before a negative point")
        point = self._validate_point(point, state["target"]["orig_hw"])
        box = self._point_box(state["target"], point)
        previous = self._copy_prompts(state)
        state["points"].append(point)
        state["boxes"].append(box)
        state["box_labels"].append(int(positive))
        return self._predict_prompt_change(state, previous)

    @torch.inference_mode()
    def remove_prompt(
        self,
        state: dict[str, object],
    ) -> list[dict[str, object]]:
        if not state["boxes"]:
            return []
        previous = self._copy_prompts(state)
        state["boxes"].pop()
        state["box_labels"].pop()
        state["points"].pop()
        return self._predict_prompt_change(state, previous)

    @torch.inference_mode()
    def update_prompt(
        self,
        state: dict[str, object],
        index: int,
        box: object,
    ) -> list[dict[str, object]]:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("prompt index must be an integer")
        if not 0 <= index < len(state["boxes"]):
            raise IndexError("prompt index is out of range")
        boxes, _ = reference.validate([box], [0], state["target"]["orig_hw"])
        previous = self._copy_prompts(state)
        state["boxes"][index] = boxes[0]
        state["points"][index] = None
        return self._predict_prompt_change(state, previous)

    @torch.inference_mode()
    def update_point(
        self,
        state: dict[str, object],
        index: int,
        point: object,
    ) -> list[dict[str, object]]:
        self._validate_prompt_index(state, index)
        point = self._validate_point(point, state["target"]["orig_hw"])
        box = self._point_box(state["target"], point)
        previous = self._copy_prompts(state)
        state["points"][index] = point
        state["boxes"][index] = box
        return self._predict_prompt_change(state, previous)

    @torch.inference_mode()
    def remove_prompt_at(
        self,
        state: dict[str, object],
        index: int,
    ) -> list[dict[str, object]]:
        return self.remove_prompts_at(state, [index])

    @torch.inference_mode()
    def remove_prompts_at(
        self,
        state: dict[str, object],
        indices: object,
    ) -> list[dict[str, object]]:
        try:
            indices = list(indices)
        except TypeError as error:
            raise TypeError("prompt indices must be a list") from error
        if not indices:
            return self.predict_prompt(state)
        for index in indices:
            self._validate_prompt_index(state, index)
        indices = set(indices)
        remaining = [
            label
            for index, label in enumerate(state["box_labels"])
            if index not in indices
        ]
        if remaining and not any(remaining):
            raise ValueError("at least one positive point is required")
        previous = self._copy_prompts(state)
        for index in sorted(indices, reverse=True):
            state["boxes"].pop(index)
            state["box_labels"].pop(index)
            state["points"].pop(index)
        return self._predict_prompt_change(state, previous)

    @staticmethod
    def _copy_prompts(state):
        return {key: list(state[key]) for key in ("boxes", "box_labels", "points")}

    def _predict_prompt_change(self, state, previous):
        try:
            return self.predict_prompt(state)
        except Exception:
            for key, values in previous.items():
                state[key][:] = values
            raise

    @staticmethod
    def _validate_prompt_index(state, index):
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("prompt index must be an integer")
        if not 0 <= index < len(state["boxes"]):
            raise IndexError("prompt index is out of range")

    @staticmethod
    def _validate_point(point, orig_hw):
        point = np.asarray(point)
        if point.shape != (2,):
            raise ValueError("point must have two coordinates")
        if not np.issubdtype(point.dtype, np.number):
            raise ValueError("point coordinates must be numbers")
        point = point.astype(np.float32, copy=False)
        if not np.isfinite(point).all():
            raise ValueError("point coordinates must be finite")
        height, width = orig_hw
        point = point.copy()
        point[0] = point[0].clip(0, max(width - 1, 0))
        point[1] = point[1].clip(0, max(height - 1, 0))
        return point

    def _point_box(self, target, point):
        if not self.use_sam_masks:
            raise RuntimeError("point prompts require SAM masks")
        points = prompt_data.build_points(
            point[None],
            [1],
            target["orig_hw"],
            self.image_size,
            self.device,
        )
        with self.autocast():
            encoded = self.model.encode_mask_prompts(points)
            masks, scores = self.model.decode_point(target["image"], encoded)[:2]
        candidate = int(scores[0].argmax().item())
        mask = mask_format.resize_masks(
            masks[:, candidate : candidate + 1],
            target["orig_hw"],
            0.0,
        )[0, 0]
        mask = mask.detach().cpu().numpy()
        if self.largest_component:
            mask = largest(mask)
        box, roi = pack.box_roi(mask)
        if roi.size == 0:
            raise RuntimeError("point did not produce an object mask")
        return np.asarray(box, dtype=np.float32)

    @torch.inference_mode()
    def predict_prompt(
        self,
        state: dict[str, object],
    ) -> list[dict[str, object]]:
        if not state["boxes"]:
            return []
        class_ids = np.zeros(len(state["boxes"]), dtype=np.int64)
        item = self.encode_reference_embed(
            state["target"],
            state["boxes"],
            class_ids,
            state["box_labels"],
        )
        return self.predict_embed(state["target"], [item])

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image | np.ndarray,
        references: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        self._check_references(references)
        return self.predict_embed(self.encode(image), references)

    @torch.inference_mode()
    def predict_embed(
        self,
        target: dict[str, object],
        references: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        self._check_references(references)

        bank = reference.feature_bank(references)
        negative_bank = reference.feature_bank(references, label=0)
        prompts, class_ids = reference.prompt_groups(references)
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
                    negative_bank,
                    self.negative_margin,
                )
            )
        objects = output.finish(
            items,
            self.nms_thr,
            self.top_k,
            target["orig_hw"],
            self.prompt_batch_size,
            self.device,
        )
        if self.use_sam_masks:
            objects = self._add_sam_masks(target, objects)
        return objects

    @torch.inference_mode()
    def refine_objects(self, state, objects):
        if not self.use_sam_masks:
            raise RuntimeError("mask refinement requires SAM masks")
        if not objects:
            return []
        out = []
        target = state["target"]
        for start in range(0, len(objects), self.prompt_batch_size):
            chunk = objects[start : start + self.prompt_batch_size]
            masks = prompt_data.build_mask(
                np.stack([item["logit"] for item in chunk]),
                self.model.mask_input_size,
                self.device,
            )
            points = (
                torch.zeros(len(chunk), 1, 2, device=self.device),
                -torch.ones(len(chunk), 1, dtype=torch.int, device=self.device),
            )
            with self.autocast():
                encoded = self.model.encode_mask_prompts(points, masks)
                decoded, scores = self.model.decode_sam_mask(target["image"], encoded)[
                    :2
                ]
            full = mask_format.resize_masks(decoded[:, :1], target["orig_hw"], 0.0)[
                :, 0
            ]
            full = full.detach().cpu().numpy()
            if self.largest_component:
                full = largest(full)
            logits = decoded[:, 0].float().detach().cpu().numpy()
            scores = scores[:, 0].float().detach().cpu().numpy()
            stability = self.model.mask_stability(decoded[:, :1])[:, 0]
            stability = stability.float().detach().cpu().numpy()
            for item, mask, logit, score, stable in zip(
                chunk,
                full,
                logits,
                scores,
                stability,
                strict=True,
            ):
                box, roi = pack.box_roi(mask)
                if roi.size == 0 or float(stable) <= item["metrics"]["stability_score"]:
                    out.append(dict(item))
                    continue
                refined = dict(item)
                refined["box"] = box
                refined["roi"] = roi.astype(bool)
                refined["logit"] = logit
                refined["metrics"] = {
                    **item["metrics"],
                    "refine_score": float(score),
                    "stability_score": float(stable),
                }
                out.append(refined)
        for index, item in enumerate(out, 1):
            item["object_id"] = index
        return out

    def _add_sam_masks(self, target, objects):
        out = []
        for start in range(0, len(objects), self.prompt_batch_size):
            chunk = objects[start : start + self.prompt_batch_size]
            boxes = [item["box"] for item in chunk]
            points = prompt_data.build_box(
                boxes,
                target["orig_hw"],
                self.image_size,
                self.device,
            )
            with self.autocast():
                encoded = self.model.encode_mask_prompts(points)
                masks, scores = self.model.decode_sam_mask(target["image"], encoded)[:2]
            full = mask_format.resize_masks(masks[:, :1], target["orig_hw"], 0.0)
            full = full[:, 0].detach().cpu().numpy()
            if self.largest_component:
                full = largest(full)
            logits = masks[:, 0].float().detach().cpu().numpy()
            scores = scores[:, 0].float().detach().cpu().numpy()
            stability = self.model.mask_stability(masks[:, :1])[:, 0]
            stability = stability.float().detach().cpu().numpy()
            for item, mask, logit, score, stable in zip(
                chunk,
                full,
                logits,
                scores,
                stability,
                strict=True,
            ):
                box, roi = pack.box_roi(mask)
                if roi.size == 0:
                    continue
                refined = dict(item)
                refined["box"] = box
                refined["roi"] = roi.astype(bool)
                refined["logit"] = logit
                refined["metrics"] = {
                    **item["metrics"],
                    "mask_score": float(score),
                    "stability_score": float(stable),
                }
                out.append(refined)
        for index, item in enumerate(out, 1):
            item["object_id"] = index
        return out

    @staticmethod
    def _check_references(references: object) -> None:
        if not isinstance(references, list):
            raise TypeError("references must be a list")
        if not references:
            raise ValueError("references list is empty")
