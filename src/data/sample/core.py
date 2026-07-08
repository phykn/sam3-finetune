from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Image:
    array: np.ndarray
    shape: tuple[int, ...] | None = None
    id: str | int | None = None

    def __post_init__(self):
        self.array = np.asarray(self.array, dtype=np.uint8)
        if self.shape is None:
            self.shape = tuple(self.array.shape)
        else:
            self.shape = tuple(self.shape)


@dataclass
class Object:
    object_id: int | str | None
    class_id: int | str | None
    box: tuple[int, int, int, int]
    roi: np.ndarray
    points: list[Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.box = tuple(self.box)
        self.roi = np.asarray(self.roi, dtype=np.uint8)

    def mask(self, shape):
        out = np.zeros(tuple(shape)[:2], dtype=np.uint8)
        x0, y0, x1, y1 = self.box
        out[y0:y1, x0:x1] = self.roi
        return out


@dataclass
class Sample:
    image: Image
    objects: list[Object] = field(default_factory=list)
