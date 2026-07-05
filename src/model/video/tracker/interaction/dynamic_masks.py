import torch

from ..multiplex.state import MultiplexState
from ..outputs import StageOutput
from .dynamic_memory import encode_mask_memory, existing_obj_ptrs
from .dynamic_output import (
    append_input_masks,
    append_mask_outputs,
    append_obj_ptrs,
    match_high_res_size,
    merge_mask_outputs,
    replace_input_masks,
    replace_obj_ptrs,
    resize_low_res_masks,
    run_mask_output,
)


def _check_mask_inputs(new_masks, obj_idxs, obj_ids):
    assert new_masks.shape[0] == len(obj_idxs)
    if obj_ids is not None:
        assert len(obj_ids) == new_masks.shape[0]


def add_new_masks_to_existing_state(
    self,
    *,
    interactive_pix_feat: torch.Tensor,
    interactive_high_res_features: list[torch.Tensor],
    propagation_vision_feats: list[torch.Tensor] | None,
    propagation_feat_sizes: list[tuple[int, int]] | None,
    new_masks: torch.Tensor,
    obj_idxs_in_mask: list[int],
    obj_ids_in_mask: list[int] | None,
    prev_output: StageOutput,
    multiplex_state: MultiplexState,
    add_mask_to_memory: bool = True,
    are_masks_from_pts: bool = False,
    allow_new_buckets: bool = False,
    prefer_new_buckets: bool = False,
) -> None:
    assert self.use_mask_input_as_output_without_sam
    _check_mask_inputs(new_masks, obj_idxs_in_mask, obj_ids_in_mask)
    existing_pointers = existing_obj_ptrs(self, prev_output, multiplex_state)

    new_object_idx = multiplex_state.find_next_batch_of_available_indices(
        num_objects=new_masks.shape[0],
        allow_new_buckets=allow_new_buckets,
        prefer_new_buckets=prefer_new_buckets,
    )
    multiplex_state.add_objects(
        object_indices=new_object_idx,
        object_ids=obj_ids_in_mask,
        allow_new_buckets=allow_new_buckets,
        prefer_new_buckets=prefer_new_buckets,
    )

    mask_output = run_mask_output(
        self,
        interactive_pix_feat=interactive_pix_feat,
        interactive_high_res_features=interactive_high_res_features,
        new_masks=new_masks,
        multiplex_state=multiplex_state,
        objects_in_mask=new_object_idx,
    )
    match_high_res_size(prev_output, mask_output)
    resize_low_res_masks(mask_output, prev_output)
    append_mask_outputs(self, prev_output, mask_output)
    append_input_masks(prev_output, new_masks)
    append_obj_ptrs(self, prev_output, mask_output, existing_pointers, multiplex_state)

    prev_output["conditioning_objects"].update(new_object_idx)
    encode_mask_memory(
        self,
        prev_output=prev_output,
        propagation_vision_feats=propagation_vision_feats,
        propagation_feat_sizes=propagation_feat_sizes,
        multiplex_state=multiplex_state,
        add_mask_to_memory=add_mask_to_memory,
        are_masks_from_pts=are_masks_from_pts,
    )


def recondition_masks_in_existing_state(
    self,
    *,
    interactive_pix_feat: torch.Tensor,
    interactive_high_res_features: list[torch.Tensor],
    propagation_vision_feats: list[torch.Tensor] | None,
    propagation_feat_sizes: list[tuple[int, int]] | None,
    new_masks: torch.Tensor,
    obj_idxs_in_mask: list[int],
    obj_ids_in_mask: list[int] | None,
    prev_output: StageOutput,
    multiplex_state: MultiplexState,
    add_mask_to_memory: bool = True,
) -> None:
    assert self.use_mask_input_as_output_without_sam
    _check_mask_inputs(new_masks, obj_idxs_in_mask, obj_ids_in_mask)
    existing_pointers = existing_obj_ptrs(self, prev_output, multiplex_state)

    mask_output = run_mask_output(
        self,
        interactive_pix_feat=interactive_pix_feat,
        interactive_high_res_features=interactive_high_res_features,
        new_masks=new_masks,
        multiplex_state=multiplex_state,
        objects_in_mask=obj_idxs_in_mask,
    )
    resize_low_res_masks(mask_output, prev_output)
    merge_mask_outputs(self, prev_output, mask_output, obj_idxs_in_mask)
    replace_input_masks(prev_output, new_masks, obj_idxs_in_mask)
    replace_obj_ptrs(
        self,
        prev_output,
        mask_output,
        existing_pointers,
        multiplex_state,
        obj_idxs_in_mask,
    )

    prev_output["conditioning_objects"].update(obj_idxs_in_mask)
    encode_mask_memory(
        self,
        prev_output=prev_output,
        propagation_vision_feats=propagation_vision_feats,
        propagation_feat_sizes=propagation_feat_sizes,
        multiplex_state=multiplex_state,
        add_mask_to_memory=add_mask_to_memory,
        are_masks_from_pts=False,
    )
