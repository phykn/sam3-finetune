from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TrainDataset
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
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.iterator = iter(loader)

    def __iter__(self) -> "InfiniteLoader":
        return self

    def __next__(self) -> dict[str, Any]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def make_infinite_train_loader(
    paths: list[str],
    batch_size: int,
    conds: list[int] | tuple[int, ...] | None = None,
    labels: (
        list[dict[str, list[float]]] | tuple[dict[str, list[float]], ...] | None
    ) = None,
    num_workers: int = 4,
) -> InfiniteLoader:
    dataset = TrainDataset(paths, conds=conds, labels=labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )
    return InfiniteLoader(loader)
