from collections import OrderedDict
from collections.abc import Callable

import torch

OUTPUT_KEYS = ("cond_frame_outputs", "non_cond_frame_outputs")


def create_output_store(
    value_factory: Callable = dict,
    *,
    cond_frame_outputs=None,
    non_cond_frame_outputs=None,
):
    store = {key: value_factory() for key in OUTPUT_KEYS}
    if cond_frame_outputs is not None:
        store["cond_frame_outputs"] = cond_frame_outputs
    if non_cond_frame_outputs is not None:
        store["non_cond_frame_outputs"] = non_cond_frame_outputs
    return store


def create_inference_state(
    *,
    num_frames: int,
    video_height: int,
    video_width: int,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    images=None,
    cached_features=None,
    track_user_refinement: bool = False,
):
    device = torch.device("cuda")
    storage_device = torch.device("cpu") if offload_state_to_cpu else device

    state = {}
    if images is not None:
        state["images"] = images

    state["num_frames"] = num_frames
    state["offload_video_to_cpu"] = offload_video_to_cpu
    state["offload_state_to_cpu"] = offload_state_to_cpu
    state["video_height"] = video_height
    state["video_width"] = video_width
    state["device"] = device
    state["storage_device"] = storage_device

    state["point_inputs_per_obj"] = {}
    state["mask_inputs_per_obj"] = {}
    state["cached_features"] = {} if cached_features is None else cached_features
    state["constants"] = {}
    state["obj_id_to_idx"] = OrderedDict()
    state["obj_idx_to_id"] = OrderedDict()
    state["obj_ids"] = []

    state["output_dict"] = create_output_store()
    state["first_ann_frame_idx"] = None
    state["output_dict_per_obj"] = {}
    state["temp_output_dict_per_obj"] = {}
    state["consolidated_frame_inds"] = create_output_store(set)
    state["tracking_has_started"] = False
    state["frames_already_tracked"] = {}
    state["multiplex_state"] = None

    if track_user_refinement:
        state["user_refined_frames_per_obj"] = {}

    return state


def add_object_slot(state, obj_id):
    obj_idx = len(state["obj_id_to_idx"])
    state["obj_id_to_idx"][obj_id] = obj_idx
    state["obj_idx_to_id"][obj_idx] = obj_id
    state["obj_ids"] = list(state["obj_id_to_idx"])

    state["point_inputs_per_obj"][obj_idx] = {}
    state["mask_inputs_per_obj"][obj_idx] = {}
    state["output_dict_per_obj"][obj_idx] = create_output_store()
    state["temp_output_dict_per_obj"][obj_idx] = create_output_store()
    return obj_idx
