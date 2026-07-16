from typing import Any

import numpy as np
from torch.utils.data import Dataset

from . import image as image_data, item
from .augment.image.crop import random_crop
from .augment.image.flip import random_flip
from .augment.image.pixel import random_pixel
from .augment.image.rotate import random_rotate
from .augment.image.zoom_out import random_zoom_out
from .sample import Sample, load

PROMPTS = ("point", "box", "mask")
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


def validate_cond(cond: int, num_conditions: int | None = None) -> int:
    if isinstance(cond, (bool, np.bool_)) or not isinstance(cond, (int, np.integer)):
        raise ValueError("cond must be an integer")
    cond = int(cond)
    if cond < 0:
        raise ValueError("cond must be non-negative")
    if num_conditions is not None and cond >= num_conditions:
        raise ValueError("cond must be in [0, num_conditions)")
    return cond


def validate_label(label: dict, num_classes: int | None = None) -> None:
    if "target" not in label or "weight" not in label:
        raise ValueError("label must contain target and weight")
    try:
        target = np.asarray(label["target"], dtype=np.float64)
        weight = np.asarray(label["weight"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("label target and weight must be numeric") from error
    if target.ndim != 1 or weight.ndim != 1:
        raise ValueError("label target and weight must be one-dimensional")
    if num_classes is not None and len(target) != num_classes:
        raise ValueError("label target length must match num_classes")
    if num_classes is not None and len(weight) != num_classes:
        raise ValueError("label weight length must match num_classes")
    if len(target) != len(weight):
        raise ValueError("label target and weight must have same length")
    if len(target) == 0:
        raise ValueError("label target is empty")
    if not np.isfinite(target).all():
        raise ValueError("label target must be finite")
    if ((target < 0) | (target > 1)).any():
        raise ValueError("label target must be in [0, 1]")
    if not np.isfinite(weight).all():
        raise ValueError("label weight must be finite")
    if (weight < 0).any():
        raise ValueError("label weight must be non-negative")


class BaseDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        prompts: list[str] | tuple[str, ...] = PROMPTS,
        conds: list[int] | tuple[int, ...] | None = None,
        num_conditions: int | None = None,
        labels: (
            list[dict[str, list[float]]] | tuple[dict[str, list[float]], ...] | None
        ) = None,
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
        if num_conditions is not None and num_conditions <= 0:
            raise ValueError("num_conditions must be positive")
        self.conds = (
            None
            if conds is None
            else tuple(validate_cond(cond, num_conditions) for cond in conds)
        )
        self.labels = None if labels is None else tuple(labels)
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
        self._check_conds()
        self._check_labels()
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
            out = item.point(image, target, union, self.bg_prob, self.mask_size)
        elif prompt == "box":
            out = item.box(image, target, self.box_jitter, self.mask_size)
        elif prompt == "mask":
            out = item.mask(image, target, self.mask_size)
        else:
            raise ValueError(f"unknown prompt type: {prompt}")
        self._add_sample_data(out, sample_index)
        return out

    def _load_sample(self, index: int) -> Sample:
        return load(self.paths[index])

    def _check_prompts(self) -> None:
        if len(self.prompts) == 0:
            raise ValueError("prompts is empty")
        for prompt in self.prompts:
            if prompt not in PROMPTS:
                raise ValueError(f"unknown prompt type: {prompt}")

    def _check_conds(self) -> None:
        if self.conds is not None and len(self.conds) != len(self.paths):
            raise ValueError("conds must match paths length")

    def _check_labels(self) -> None:
        if self.labels is None:
            return
        if len(self.labels) != len(self.paths):
            raise ValueError("labels must match paths length")
        for label in self.labels:
            validate_label(label)

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
        valid = np.ones_like(target, dtype=np.uint8)
        base_image = image
        base_target = target
        base_union = union
        base_valid = valid
        if self.shape_aug:
            pair = np.stack([target, union, valid], axis=-1)
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
            valid = pair[..., 2]
            if target.sum() == 0:
                image = base_image
                target = base_target
                union = base_union
                valid = base_valid
        return self._resize_input(image, target, union, valid)

    def _resize_input(
        self,
        image: np.ndarray,
        target: np.ndarray,
        union: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pair = np.stack([target, union, valid], axis=-1)
        image = image_data.resize_image(image, self.size)
        pair = image_data.resize_mask(pair, self.size)
        target = pair[..., 0]
        union = pair[..., 1]
        # Keep background clicks outside augmentation padding.
        union[pair[..., 2] == 0] = 1
        return image, target, union

    def _make_union_mask(self, sample: Sample) -> np.ndarray:
        out = np.zeros(sample.image.shape[:2], dtype=bool)
        for obj in sample.objects:
            out |= obj.mask(sample.image.shape).astype(bool)
        return out.astype(np.uint8)

    def _add_sample_data(self, item: dict[str, Any], sample_index: int) -> None:
        if self.conds is not None:
            item["cond"] = self.conds[sample_index]
        if self.labels is not None:
            label = self.labels[sample_index]
            target = np.asarray(label["target"], dtype=np.float32)
            weight = np.asarray(label["weight"], dtype=np.float32)
            if item["is_auto_bg"] or target[0] == 0:
                target = np.zeros_like(target)
                weight = np.zeros_like(weight)
                weight[0] = 1.0
                item["mask_valid"] = False
            else:
                item["mask_valid"] = True
            item["label_target"] = target
            item["label_weight"] = weight


class TrainDataset(BaseDataset):
    def __init__(
        self,
        paths: list[str],
        conds: list[int] | tuple[int, ...] | None = None,
        num_conditions: int | None = None,
        labels: (
            list[dict[str, list[float]]] | tuple[dict[str, list[float]], ...] | None
        ) = None,
        bg_prob: float = 0.2,
        box_jitter: float = 0.1,
    ) -> None:
        super().__init__(
            paths=paths,
            conds=conds,
            num_conditions=num_conditions,
            labels=labels,
            bg_prob=bg_prob,
            box_jitter=box_jitter,
            image_aug=True,
            shape_aug=True,
        )


class ValidDataset(BaseDataset):
    def __init__(
        self,
        paths: list[str],
        conds: list[int] | tuple[int, ...] | None = None,
        num_conditions: int | None = None,
        labels: (
            list[dict[str, list[float]]] | tuple[dict[str, list[float]], ...] | None
        ) = None,
        bg_prob: float = 0.2,
    ) -> None:
        super().__init__(
            paths=paths,
            conds=conds,
            num_conditions=num_conditions,
            labels=labels,
            prompts=("point",),
            bg_prob=bg_prob,
            box_jitter=0.0,
            image_aug=False,
            shape_aug=False,
        )
