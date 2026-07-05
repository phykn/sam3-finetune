import torch

from .frame_merge import merge_output_dict
from .merge_state import (
    copy_singleton_object_state,
    register_object,
    sync_multiplex_state,
)


@torch.inference_mode()
def merge_singleton_interaction_result(
    model,
    inference_state,
    singleton_state,
    obj_id,
):
    obj_idx = register_object(inference_state, obj_id)
    target_state = sync_multiplex_state(model, inference_state, obj_id, obj_idx)

    copy_singleton_object_state(inference_state, singleton_state, obj_idx)
    merge_output_dict(
        model,
        inference_state,
        singleton_state,
        obj_id,
        obj_idx,
        target_state,
    )
