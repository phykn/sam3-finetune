from typing import Any

import numpy as np
from PIL import Image as PILImage
from torch.utils.data import Dataset

from .augment.image.crop import random_crop
from .augment.image.flip import random_flip
from .augment.image.pixel import random_pixel
from .augment.image.resize import resize
from .augment.image.rotate import random_rotate
from .augment.image.zoom_out import random_zoom_out
from .augment.prompt.box import jitter_mask_box
from .augment.prompt.mask import degrade_mask_prompt
from .augment.prompt.point import sample_point_prompt
from .sample import Sample, load

PROMPTS = ("point", "box", "mask")
MASK_OPS = ("none", "shift", "erode", "dilate", "blur", "resize")
IMAGE_OPS = (
    "none",
    "brightness",
    "contrast",
    "saturation",
    "blur",
    "noise",
    "dropout",
)
SIZE = 1008
MASK_SIZE = 288


class BaseDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        prompts: list[str] | tuple[str, ...] = PROMPTS,
        bg_prob: float = 0.2,
        box_jitter: float = 0.1,
        image_aug: bool = False,
        image_ops: list[str] | tuple[str, ...] = IMAGE_OPS,
        shape_aug: bool = False,
        scale: tuple[float, float] = (0.5, 1.5),
        size: int = SIZE,
        mask_size: int = MASK_SIZE,
    ) -> None:
        self.paths = paths
        self.prompts = tuple(prompts)
        self.bg_prob = bg_prob
        self.box_jitter = box_jitter
        self.image_aug = image_aug
        self.image_ops = tuple(image_ops)
        self.shape_aug = shape_aug
        self.scale = scale
        self.size = size
        self.mask_size = mask_size
        self._check_prompts()
        self._check_image_ops()
        self._check_scale()
        self._check_size()
        self.items = self._collect_object_items()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index, object_index = self.items[index]
        sample = self._load_sample(sample_index)
        obj = sample.objects[object_index]
        image = sample.image.array
        target = obj.mask(sample.image.shape).astype(np.uint8, copy=False)
        union = self._make_union_mask(sample)
        image, target, union = self._augment_shape(image, target, union)
        image = self._augment_image(image)
        prompt = self.prompts[int(np.random.randint(len(self.prompts)))]

        if prompt == "point":
            return self._make_point_item(image, target, union)
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

    def _check_scale(self) -> None:
        if not 0.0 <= self.scale[0] <= self.scale[1]:
            raise ValueError("scale must be 0 or greater")

    def _check_size(self) -> None:
        if self.size <= 0 or self.mask_size <= 0:
            raise ValueError("size must be positive")

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
        return random_pixel(image, self.image_ops)

    def _augment_shape(
        self,
        image: np.ndarray,
        target: np.ndarray,
        union: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = np.asarray(image, dtype=np.uint8)
        target = np.asarray(target, dtype=np.uint8)
        union = np.asarray(union, dtype=np.uint8)
        base_image = image
        base_target = target
        base_union = union
        if self.shape_aug:
            pair = np.stack([target, union], axis=-1)
            scale = float(np.random.uniform(self.scale[0], self.scale[1]))
            if scale < 1.0:
                image, pair = random_crop(image, pair, scale=scale)
            elif scale > 1.0:
                image, pair = random_zoom_out(image, pair, scale=scale)

            image, pair = random_rotate(image, pair)
            image, pair = random_flip(image, pair)
            pair = np.asarray(pair, dtype=np.uint8)
            target = pair[..., 0]
            union = pair[..., 1]
            if target.sum() == 0:
                image = base_image
                target = base_target
                union = base_union
        return self._resize_input(image, target, union)

    def _resize_input(
        self,
        image: np.ndarray,
        target: np.ndarray,
        union: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid = np.ones_like(target, dtype=np.uint8)
        pair = np.stack([target, union, valid], axis=-1)
        image, pair = resize(image, pair, size=(self.size, self.size))
        pair = np.asarray(pair, dtype=np.uint8)
        target = pair[..., 0]
        union = pair[..., 1]
        # Keep background clicks inside the real image, not resize padding.
        union[pair[..., 2] == 0] = 1
        return image, target, union

    def _make_point_item(
        self,
        image: np.ndarray,
        target: np.ndarray,
        union: np.ndarray,
    ) -> dict[str, Any]:
        out = sample_point_prompt(
            target,
            union,
            bg_prob=self.bg_prob,
        )
        prompt = self._empty_prompt("point")
        prompt["points"] = out["points"]
        prompt["point_labels"] = out["point_labels"]
        return {
            "image": image,
            "prompt": prompt,
            "target": self._resize_binary_mask(out["target"]),
            "has_object": out["has_object"],
        }

    def _make_box_item(self, image: np.ndarray, target: np.ndarray) -> dict[str, Any]:
        prompt = self._empty_prompt("box")
        prompt["box"] = jitter_mask_box(
            target,
            image.shape,
            amount=self.box_jitter,
        )
        return {
            "image": image,
            "prompt": prompt,
            "target": self._resize_binary_mask(target),
            "has_object": True,
        }

    def _make_mask_item(self, image: np.ndarray, target: np.ndarray) -> dict[str, Any]:
        prompt = self._empty_prompt("mask")
        mask = degrade_mask_prompt(
            target,
            ops=MASK_OPS,
        )
        prompt["mask"] = self._resize_float_mask(mask)
        return {
            "image": image,
            "prompt": prompt,
            "target": self._resize_binary_mask(target),
            "has_object": True,
        }

    def _resize_binary_mask(self, mask: np.ndarray) -> np.ndarray:
        image = PILImage.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        image = image.resize(
            (self.mask_size, self.mask_size),
            PILImage.Resampling.BILINEAR,
        )
        return (np.asarray(image) > 0).astype(np.uint8)

    def _resize_float_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=np.float32)
        image = PILImage.fromarray(
            np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8),
            mode="L",
        )
        image = image.resize(
            (self.mask_size, self.mask_size),
            PILImage.Resampling.BILINEAR,
        )
        return np.asarray(image, dtype=np.float32) / 255.0

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
    def __init__(
        self,
        paths: list[str],
        bg_prob: float = 0.2,
        box_jitter: float = 0.1,
    ) -> None:
        super().__init__(
            paths=paths,
            bg_prob=bg_prob,
            box_jitter=box_jitter,
            image_aug=True,
            shape_aug=True,
        )


class ValidDataset(BaseDataset):
    def __init__(
        self,
        paths: list[str],
        bg_prob: float = 0.2,
    ) -> None:
        super().__init__(
            paths=paths,
            prompts=("point",),
            bg_prob=bg_prob,
            box_jitter=0.0,
            image_aug=False,
            shape_aug=False,
        )
