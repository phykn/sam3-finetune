from collections import OrderedDict

import torch

OUTPUT_KEYS = ("cond_frame_outputs", "non_cond_frame_outputs")


def output_store(factory=dict):
    return {key: factory() for key in OUTPUT_KEYS}


def create_state(
    *,
    num_frames,
    video_height,
    video_width,
    cached_features=None,
    device="cuda",
    offload_video_to_cpu=False,
    offload_state_to_cpu=False,
):
    if min(num_frames, video_height, video_width) <= 0:
        raise ValueError("frame count and video size must be positive")

    device = torch.device(device)
    state = {
        "num_frames": num_frames,
        "offload_video_to_cpu": offload_video_to_cpu,
        "offload_state_to_cpu": offload_state_to_cpu,
        "video_height": video_height,
        "video_width": video_width,
        "device": device,
        "storage_device": torch.device("cpu") if offload_state_to_cpu else device,
        "mask_inputs_per_obj": {},
        "cached_features": {} if cached_features is None else cached_features,
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict": output_store(),
        "first_ann_frame_idx": None,
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "consolidated_frame_inds": output_store(set),
        "tracking_has_started": False,
        "frames_already_tracked": {},
        "multiplex_state": None,
    }
    return state


def add_object(state, obj_id):
    current = state["obj_id_to_idx"].get(obj_id)
    if current is not None:
        return current

    index = len(state["obj_ids"])
    state["obj_id_to_idx"][obj_id] = index
    state["obj_idx_to_id"][index] = obj_id
    state["obj_ids"].append(obj_id)
    state["mask_inputs_per_obj"][index] = {}
    state["output_dict_per_obj"][index] = output_store()
    state["temp_output_dict_per_obj"][index] = output_store()
    return index


def cached_frame(state, frame_idx):
    try:
        return state["cached_features"][frame_idx]
    except KeyError:
        raise KeyError(f"frame {frame_idx} is not cached") from None


def forward_frames(start, count, num_frames):
    if start < 0:
        raise ValueError("start must be non-negative")
    if count <= 0:
        raise ValueError("count must be positive")
    return range(start, min(start + count, num_frames))
