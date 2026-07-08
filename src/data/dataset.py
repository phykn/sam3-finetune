import numpy as np
from torch.utils.data import Dataset

from .augment.prompt import box as box_aug
from .augment.prompt import mask as mask_aug
from .augment.prompt import point as point_aug
from .sample import load

DEFAULT_CONFIG = {
    "prompt": "point",
    "bg_prob": 0.2,
    "box_jitter": 0.1,
    "mask_ops": ("none", "shift", "erode", "dilate", "blur", "resize"),
}


class BaseDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return load(self.paths[index])


class TrainDataset(BaseDataset):
    def __init__(self, paths, config=None):
        super().__init__(paths)
        self.config = DEFAULT_CONFIG.copy()
        if config is not None:
            self.config.update(config)
        self.items = self._items()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sample_index, object_index = self.items[index]
        sample = BaseDataset.__getitem__(self, sample_index)
        obj = sample.objects[object_index]
        target = obj.mask(sample.image.shape).astype(np.uint8, copy=False)
        prompt = self.config["prompt"]

        if prompt == "point":
            return self._point_item(sample, target)
        if prompt == "box":
            return self._box_item(sample, target)
        if prompt == "mask":
            return self._mask_item(sample, target)
        raise ValueError(f"unknown prompt type: {prompt}")

    def _items(self):
        items = []
        for sample_index in range(len(self.paths)):
            sample = BaseDataset.__getitem__(self, sample_index)
            for object_index, obj in enumerate(sample.objects):
                if obj.mask(sample.image.shape).sum() > 0:
                    items.append((sample_index, object_index))
        return items

    def _point_item(self, sample, target):
        out = point_aug.sample_point_prompt(
            target,
            self._union(sample),
            bg_prob=self.config["bg_prob"],
        )
        prompt = self._prompt("point")
        prompt["points"] = out["points"]
        prompt["point_labels"] = out["point_labels"]
        return {
            "image": sample.image.array,
            "prompt": prompt,
            "target": out["target"],
            "has_object": out["has_object"],
        }

    def _box_item(self, sample, target):
        prompt = self._prompt("box")
        prompt["box"] = box_aug.jitter_mask_box(
            target,
            sample.image.shape,
            amount=self.config["box_jitter"],
        )
        return {
            "image": sample.image.array,
            "prompt": prompt,
            "target": target,
            "has_object": True,
        }

    def _mask_item(self, sample, target):
        prompt = self._prompt("mask")
        prompt["mask"] = mask_aug.degrade_mask_prompt(
            target,
            ops=self.config["mask_ops"],
        )
        return {
            "image": sample.image.array,
            "prompt": prompt,
            "target": target,
            "has_object": True,
        }

    def _union(self, sample):
        out = np.zeros(sample.image.shape[:2], dtype=bool)
        for obj in sample.objects:
            out |= obj.mask(sample.image.shape).astype(bool)
        return out.astype(np.uint8)

    def _prompt(self, kind):
        return {
            "type": kind,
            "points": None,
            "point_labels": None,
            "box": None,
            "mask": None,
        }
