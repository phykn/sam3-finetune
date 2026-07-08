from typing import Any

import albumentations as A
import numpy as np
from torch.utils.data import Dataset

from .augment.prompt import box as box_aug
from .augment.prompt import mask as mask_aug
from .augment.prompt import point as point_aug
from .sample import Sample, load

PROMPTS = ("point", "box", "mask")
MASK_OPS = ("none", "shift", "erode", "dilate", "blur", "resize")
IMAGE_OPS = ("none", "brightness", "contrast", "saturation", "blur", "noise", "dropout")


class BaseDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        prompts: list[str] | tuple[str, ...] = PROMPTS,
        bg_prob: float = 0.2,
        box_jitter: float = 0.1,
        image_aug: bool = False,
        image_ops: list[str] | tuple[str, ...] = IMAGE_OPS,
    ) -> None:
        self.paths = paths
        self.prompts = tuple(prompts)
        self.bg_prob = bg_prob
        self.box_jitter = box_jitter
        self.image_aug = image_aug
        self.image_ops = tuple(image_ops)
        self._check_prompts()
        self._check_image_ops()
        self.items = self._collect_object_items()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index, object_index = self.items[index]
        sample = self._load_sample(sample_index)
        obj = sample.objects[object_index]
        image = self._augment_image(sample.image.array)
        target = obj.mask(sample.image.shape).astype(np.uint8, copy=False)
        prompt = self.prompts[int(np.random.randint(len(self.prompts)))]

        if prompt == "point":
            return self._make_point_item(sample, image, target)
        if prompt == "box":
            return self._make_box_item(image, target)
        if prompt == "mask":
            return self._make_mask_item(image, target)
        raise ValueError(f"unknown prompt type: {prompt}")

    def _load_sample(self, index: int) -> Sample:
        return load(self.paths[index])

    def _check_prompts(self) -> None:
        if len(self.prompts) == 0:
            raise ValueError("prompts is empty")
        for prompt in self.prompts:
            if prompt not in PROMPTS:
                raise ValueError(f"unknown prompt type: {prompt}")

    def _check_image_ops(self) -> None:
        if len(self.image_ops) == 0:
            raise ValueError("image_ops is empty")
        for op in self.image_ops:
            if op not in IMAGE_OPS:
                raise ValueError(f"unknown image op: {op}")

    def _collect_object_items(self) -> list[tuple[int, int]]:
        items: list[tuple[int, int]] = []
        for sample_index in range(len(self.paths)):
            sample = self._load_sample(sample_index)
            for object_index, obj in enumerate(sample.objects):
                if obj.mask(sample.image.shape).sum() > 0:
                    items.append((sample_index, object_index))
        return items

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)
        if not self.image_aug:
            return image

        op = str(np.random.choice(self.image_ops))
        if op == "none":
            return image
        if op == "brightness":
            transform = A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.0,
                p=1,
            )
        elif op == "contrast":
            transform = A.RandomBrightnessContrast(
                brightness_limit=0.0,
                contrast_limit=0.2,
                p=1,
            )
        elif op == "saturation":
            transform = A.HueSaturationValue(
                hue_shift_limit=0,
                sat_shift_limit=20,
                val_shift_limit=0,
                p=1,
            )
        elif op == "blur":
            transform = A.GaussianBlur(
                blur_limit=(3, 5),
                sigma_limit=(0.5, 1.5),
                p=1,
            )
        elif op == "noise":
            transform = A.GaussNoise(p=1)
        elif op == "dropout":
            transform = A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(0.05, 0.2),
                hole_width_range=(0.05, 0.2),
                fill=0,
                p=1,
            )
        else:
            raise ValueError(f"unknown image op: {op}")

        out = transform(image=image)["image"]
        return np.asarray(out, dtype=np.uint8)

    def _make_point_item(
        self,
        sample: Sample,
        image: np.ndarray,
        target: np.ndarray,
    ) -> dict[str, Any]:
        out = point_aug.sample_point_prompt(
            target,
            self._make_union_mask(sample),
            bg_prob=self.bg_prob,
        )
        prompt = self._empty_prompt("point")
        prompt["points"] = out["points"]
        prompt["point_labels"] = out["point_labels"]
        return {
            "image": image,
            "prompt": prompt,
            "target": out["target"],
            "has_object": out["has_object"],
        }

    def _make_box_item(self, image: np.ndarray, target: np.ndarray) -> dict[str, Any]:
        prompt = self._empty_prompt("box")
        prompt["box"] = box_aug.jitter_mask_box(
            target,
            image.shape,
            amount=self.box_jitter,
        )
        return {
            "image": image,
            "prompt": prompt,
            "target": target,
            "has_object": True,
        }

    def _make_mask_item(self, image: np.ndarray, target: np.ndarray) -> dict[str, Any]:
        prompt = self._empty_prompt("mask")
        prompt["mask"] = mask_aug.degrade_mask_prompt(
            target,
            ops=MASK_OPS,
        )
        return {
            "image": image,
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
