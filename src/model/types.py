from dataclasses import dataclass
from typing import Any

import torch
from torch.utils import _pytree as pytree


class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def clone(self):
        new_tensors = self.tensors.clone()
        new_mask = None if self.mask is None else self.mask.clone()
        return NestedTensor(new_tensors, new_mask)

    def __getitem__(self, idx):
        return self.tensors[idx]

    def __len__(self):
        return len(self.tensors)

    @property
    def device(self):
        return self.tensors.device

    @property
    def shape(self):
        return self.tensors.shape

    def pin_memory(self, device=None):
        self.tensors = self.tensors.pin_memory(device)
        if self.mask is not None:
            self.mask = self.mask.pin_memory(device)


# tree_map_only must be able to traverse NestedTensor in the video tracker path.
pytree.register_pytree_node(
    NestedTensor,
    lambda x: ([x.tensors, x.mask], None),
    lambda values, _: NestedTensor(values[0], values[1]),
)


@dataclass
class FindStage:
    img_ids: torch.Tensor | list[Any]
    img_ids__type = torch.long
    text_ids: torch.Tensor | list[Any]
    text_ids__type = torch.long

    input_boxes: torch.Tensor | list[Any]
    input_boxes__type = torch.float
    input_boxes_mask: torch.Tensor | list[Any]
    input_boxes_mask__type = torch.bool
    input_boxes_label: torch.Tensor | list[Any]
    input_boxes_label__type = torch.long

    input_points: torch.Tensor | list[Any]
    input_points__type = torch.float
    input_points_mask: torch.Tensor | list[Any]
    input_points_mask__type = torch.bool

    object_ids: list[list] | None = None

    img_ids_np: Any | None = None
    input_boxes_before_embed: torch.Tensor | list[Any] | None = None
    input_boxes_before_embed__type = torch.float
    input_points_before_embed: torch.Tensor | list[Any] | None = None
    input_points_before_embed__type = torch.float
    ptrs: Any | None = None
    ptrs_seg: Any | None = None


@dataclass
class BatchedDatapoint:
    img_batch: torch.Tensor
    find_text_batch: list[str]
    find_inputs: list[FindStage]
    find_targets: list[Any]
    find_metadatas: list[Any]
    raw_images: list[Any] | None = None
    get_queries: Any | None = None
