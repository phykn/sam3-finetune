from numbers import Integral
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import TrainDataset, ValidDataset, validate_cond, validate_label
from .folder import expand
from .image import to_tensor


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    mask = np.ascontiguousarray(mask)
    return torch.from_numpy(mask).float().unsqueeze(0)


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "image": torch.stack([to_tensor(item["image"]) for item in batch]),
        "target": torch.stack([mask_to_tensor(item["target"]) for item in batch]),
        "mask_valid": torch.tensor(
            [float(item["mask_valid"]) for item in batch],
            dtype=torch.float32,
        ),
        "is_auto_bg": torch.tensor(
            [float(item["is_auto_bg"]) for item in batch],
            dtype=torch.float32,
        ),
        "prompt": [item["prompt"] for item in batch],
    }
    if "cond" in batch[0]:
        out["cond"] = torch.tensor([int(item["cond"]) for item in batch])
    if "label_target" in batch[0]:
        out["label_target"] = torch.stack(
            [
                torch.as_tensor(item["label_target"], dtype=torch.float32)
                for item in batch
            ]
        )
        out["label_weight"] = torch.stack(
            [
                torch.as_tensor(item["label_weight"], dtype=torch.float32)
                for item in batch
            ]
        )
    return out


class InfiniteLoader:
    def __init__(self, loader: DataLoader, sampler=None) -> None:
        self.loader = loader
        self.sampler = sampler
        self.epoch = 0
        self.iterator = iter(loader)

    def __iter__(self) -> "InfiniteLoader":
        return self

    def __next__(self) -> dict[str, Any]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            if self.sampler is not None:
                self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            return next(self.iterator)


def make_infinite_train_loader(
    paths: list[str],
    batch_size: int,
    num_classes: int,
    num_conditions: int,
    conds: list[int] | tuple[int, ...],
    labels: list[dict[str, list[float]]] | tuple[dict[str, list[float]], ...],
    num_workers: int = 4,
) -> InfiniteLoader:
    return make_finetune_loader(
        {
            "paths": paths,
            "batch_size": batch_size,
            "conds": conds,
            "labels": labels,
            "num_workers": num_workers,
        },
        num_classes=num_classes,
        num_conditions=num_conditions,
        train=True,
    )


def make_finetune_loader(
    config: dict[str, Any],
    num_classes: int,
    num_conditions: int,
    train: bool,
    rank: int = 0,
    world_size: int = 1,
) -> InfiniteLoader:
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral):
        raise ValueError("num_classes must be a positive integer")
    if num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if isinstance(num_conditions, bool) or not isinstance(num_conditions, Integral):
        raise ValueError("num_conditions must be a positive integer")
    if num_conditions <= 0:
        raise ValueError("num_conditions must be a positive integer")

    dataset_type = TrainDataset if train else ValidDataset
    if "folders" in config:
        folders = config["folders"]
        for folder in folders:
            if "cond" not in folder:
                raise ValueError("folder must contain cond")
        _check_conds([folder["cond"] for folder in folders], num_conditions)
        _check_labels(folders, num_classes)
        paths, conds, labels = expand(folders)
    else:
        paths = config["paths"]
        conds = config.get("conds")
        labels = config.get("labels")
        if conds is None:
            raise ValueError("conds are required for finetuning")
        if labels is None:
            raise ValueError("labels are required for finetuning")
        _check_conds(conds, num_conditions)
        _check_labels(labels, num_classes)
    dataset = dataset_type(
        paths,
        conds=conds,
        num_conditions=num_conditions,
        labels=labels,
    )
    if len(dataset) == 0:
        raise ValueError("finetune dataset contains no valid objects")

    batch_size = config["batch_size"]
    per_rank = len(dataset) // world_size if world_size > 1 else len(dataset)
    if train and per_rank < batch_size:
        raise ValueError(
            "per-rank dataset size must be at least batch_size when training"
        )

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            drop_last=train,
        )
    num_workers = config.get("num_workers", 4)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )
    return InfiniteLoader(loader, sampler=sampler)


def _check_conds(conds: list[int] | tuple[int, ...], num_conditions: int) -> None:
    for cond in conds:
        validate_cond(cond, num_conditions)


def _check_labels(labels: list[dict] | tuple[dict, ...], num_classes: int) -> None:
    for label in labels:
        validate_label(label, num_classes)
