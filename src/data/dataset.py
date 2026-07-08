from typing import Any

import numpy as np
from torch.utils.data import Dataset

from .augment.prompt import box as box_aug
from .augment.prompt import mask as mask_aug
from .augment.prompt import point as point_aug
from .sample import Sample, load

DEFAULT_CONFIG: dict[str, Any] = {
    "prompt": "point",
    "bg_prob": 0.2,
    "box_jitter": 0.1,
    "mask_ops": ("none", "shift", "erode", "dilate", "blur", "resize"),
}


class BaseDataset(Dataset):
    def __init__(self, paths: list[str], config: dict[str, Any] | None = None) -> None:
        self.paths = paths
        self.config = DEFAULT_CONFIG.copy()
        if config is not None:
            self.config.update(config)
        self.items = self._collect_object_items()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index, object_index = self.items[index]
        sample = self._load_sample(sample_index)
        obj = sample.objects[object_index]
        target = obj.mask(sample.image.shape).astype(np.uint8, copy=False)
        prompt = self.config["prompt"]

        if prompt == "point":
            return self._make_point_item(sample, target)
        if prompt == "box":
            return self._make_box_item(sample, target)
        if prompt == "mask":
            return self._make_mask_item(sample, target)
        raise ValueError(f"unknown prompt type: {prompt}")

    def _load_sample(self, index: int) -> Sample:
        return load(self.paths[index])

    def _collect_object_items(self) -> list[tuple[int, int]]:
        items: list[tuple[int, int]] = []
        for sample_index in range(len(self.paths)):
            sample = self._load_sample(sample_index)
            for object_index, obj in enumerate(sample.objects):
                if obj.mask(sample.image.shape).sum() > 0:
                    items.append((sample_index, object_index))
        return items

    def _make_point_item(self, sample: Sample, target: np.ndarray) -> dict[str, Any]:
        out = point_aug.sample_point_prompt(
            target,
            self._make_union_mask(sample),
            bg_prob=float(self.config["bg_prob"]),
        )
        prompt = self._empty_prompt("point")
        prompt["points"] = out["points"]
        prompt["point_labels"] = out["point_labels"]
        return {
            "image": sample.image.array,
            "prompt": prompt,
            "target": out["target"],
            "has_object": out["has_object"],
        }

    def _make_box_item(self, sample: Sample, target: np.ndarray) -> dict[str, Any]:
        prompt = self._empty_prompt("box")
        prompt["box"] = box_aug.jitter_mask_box(
            target,
            sample.image.shape,
            amount=float(self.config["box_jitter"]),
        )
        return {
            "image": sample.image.array,
            "prompt": prompt,
            "target": target,
            "has_object": True,
        }

    def _make_mask_item(self, sample: Sample, target: np.ndarray) -> dict[str, Any]:
        prompt = self._empty_prompt("mask")
        prompt["mask"] = mask_aug.degrade_mask_prompt(
            target,
            ops=tuple(self.config["mask_ops"]),
        )
        return {
            "image": sample.image.array,
            "prompt": prompt,
            "target": target,
            "has_object": True,
        }

    def _make_union_mask(self, sample: Sample) -> np.ndarray:
        out = np.zeros(sample.image.shape[:2], dtype=bool)
        for obj in sample.objects:
            out |= obj.mask(sample.image.shape).astype(bool)
        return out.astype(np.uint8)

    def _empty_prompt(self, kind: str) -> dict[str, Any]:
        return {
            "type": kind,
            "points": None,
            "point_labels": None,
            "box": None,
            "mask": None,
        }


class TrainDataset(BaseDataset):
    pass
