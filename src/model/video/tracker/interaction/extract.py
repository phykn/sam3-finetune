import logging

import torch

from ..state import create_output_store, OUTPUT_KEYS


def _slice_obj(tensor, obj_idx):
    return tensor[obj_idx : obj_idx + 1].clone()


def _is_muxed(tensor, state):
    if tensor is None or state is None:
        return False
    return (
        tensor.dim() >= 2
        and tensor.shape[0] == state.num_buckets
        and tensor.shape[1] == state.multiplex_count
    )


def _demux_or_none(tensor, state, label):
    try:
        return state.demux(tensor)
    except AssertionError as exc:
        logging.warning(
            "[EXTRACT] demux failed for %s shape %s: %s",
            label,
            tuple(tensor.shape),
            exc,
        )
        return None


def _extract_maskmem_features(source_frame_out, multiplex_state, obj_idx):
    features = source_frame_out.get("maskmem_features")
    if features is None:
        return None

    if multiplex_state is None:
        return _slice_obj(features, obj_idx)

    if _is_muxed(features, multiplex_state):
        demuxed = _demux_or_none(features, multiplex_state, "maskmem_features")
        return _slice_obj(demuxed if demuxed is not None else features, obj_idx)

    if features.shape[0] == 0:
        return None

    if features.shape[0] >= obj_idx + 1:
        return _slice_obj(features, obj_idx)

    logging.warning(
        "[EXTRACT] maskmem_features shape %s incompatible with multiplex state; dropping tensor",
        tuple(features.shape),
    )
    return None


def _extract_maskmem_pos_level(level_enc, multiplex_state, obj_idx):
    if level_enc is None:
        return None

    if multiplex_state is None:
        return _slice_obj(level_enc, obj_idx)

    if _is_muxed(level_enc, multiplex_state):
        demuxed = _demux_or_none(level_enc, multiplex_state, "maskmem_pos_enc level")
        if demuxed is not None:
            return _slice_obj(demuxed, obj_idx)

    if level_enc.shape[0] >= obj_idx + 1:
        return _slice_obj(level_enc, obj_idx)

    logging.warning(
        "[EXTRACT] maskmem_pos_enc level shape %s incompatible with multiplex state; dropping level",
        tuple(level_enc.shape),
    )
    return None


def _extract_maskmem_pos_enc(source_frame_out, multiplex_state, obj_idx):
    pos_enc = source_frame_out.get("maskmem_pos_enc")
    if pos_enc is None:
        return None

    return [
        _extract_maskmem_pos_level(level_enc, multiplex_state, obj_idx)
        for level_enc in pos_enc
    ]


def _set_obj_ptr(
    model, singleton_frame_out, source_frame_out, multiplex_state, obj_idx
):
    if "obj_ptr" not in source_frame_out or not model.use_obj_ptrs_in_encoder:
        return

    source_obj_ptr = source_frame_out["obj_ptr"]
    if multiplex_state is not None:
        source_obj_ptr = multiplex_state.demux(source_obj_ptr)

    singleton_frame_out["obj_ptr"] = _slice_obj(source_obj_ptr, obj_idx)


def _set_conditioning_objects(singleton_frame_out, source_frame_out, obj_idx):
    if "conditioning_objects" not in source_frame_out:
        return

    if obj_idx in source_frame_out["conditioning_objects"]:
        singleton_frame_out["conditioning_objects"] = {0}
    else:
        singleton_frame_out["conditioning_objects"] = set()


def _extract_frame_out(model, source_frame_out, obj_id, obj_idx, multiplex_state):
    if source_frame_out["pred_masks"].shape[0] < obj_idx + 1:
        return None

    singleton_frame_out = {
        "pred_masks": _slice_obj(source_frame_out["pred_masks"], obj_idx),
        "object_score_logits": _slice_obj(
            source_frame_out["object_score_logits"],
            obj_idx,
        ),
        "image_features": source_frame_out.get("image_features"),
        "image_pos_enc": source_frame_out.get("image_pos_enc"),
        "local_obj_id_to_idx": {obj_id: 0},
        "maskmem_features": _extract_maskmem_features(
            source_frame_out,
            multiplex_state,
            obj_idx,
        ),
        "maskmem_pos_enc": _extract_maskmem_pos_enc(
            source_frame_out,
            multiplex_state,
            obj_idx,
        ),
    }
    _set_obj_ptr(model, singleton_frame_out, source_frame_out, multiplex_state, obj_idx)
    _set_conditioning_objects(singleton_frame_out, source_frame_out, obj_idx)

    return singleton_frame_out


def _extract_consolidated_outputs(model, source_state, obj_id, obj_idx):
    singleton_outputs = create_output_store()
    multiplex_state = source_state.get("multiplex_state")

    if "output_dict" not in source_state:
        return singleton_outputs

    for storage_key in OUTPUT_KEYS:
        source_outputs = source_state["output_dict"].get(storage_key, {})
        for frame_idx, source_frame_out in source_outputs.items():
            frame_out = _extract_frame_out(
                model,
                source_frame_out,
                obj_id,
                obj_idx,
                multiplex_state,
            )
            if frame_out is not None:
                singleton_outputs[storage_key][frame_idx] = frame_out

    return singleton_outputs


def _copy_object_bucket(source_state, key, obj_idx):
    if key not in source_state:
        return {}
    if obj_idx not in source_state[key]:
        return {}
    return source_state[key][obj_idx].copy()


def _copy_object_outputs(source_state, key, obj_idx):
    if key not in source_state or obj_idx not in source_state[key]:
        return {}, {}

    output_dict = source_state[key][obj_idx]
    return (
        output_dict.get("cond_frame_outputs", {}).copy(),
        output_dict.get("non_cond_frame_outputs", {}).copy(),
    )


def _clear_empty_multiplex_state(source_state):
    multiplex_state = source_state.get("multiplex_state")
    if multiplex_state is None:
        return

    if (
        getattr(multiplex_state, "assignments", None) is None
        or multiplex_state.total_valid_entries == 0
    ):
        source_state["multiplex_state"] = None


def _init_singleton_state(model, source_state, obj_id, point_inputs, mask_inputs):
    singleton_state = model.init_state(
        cached_features=source_state["cached_features"],
        video_height=source_state["video_height"],
        video_width=source_state["video_width"],
        num_frames=source_state["num_frames"],
    )

    singleton_state["obj_id_to_idx"] = {obj_id: 0}
    singleton_state["obj_idx_to_id"] = {0: obj_id}
    singleton_state["obj_ids"] = [obj_id]
    singleton_state["point_inputs_per_obj"] = {0: point_inputs}
    singleton_state["mask_inputs_per_obj"] = {0: mask_inputs}
    singleton_state["frames_already_tracked"] = source_state[
        "frames_already_tracked"
    ].copy()

    return singleton_state


def _set_singleton_output_stores(
    singleton_state,
    obj_outputs,
    temp_outputs,
):
    obj_cond_outputs, obj_non_cond_outputs = obj_outputs
    temp_cond_outputs, temp_non_cond_outputs = temp_outputs

    singleton_state["output_dict_per_obj"] = {
        0: create_output_store(
            cond_frame_outputs=obj_cond_outputs,
            non_cond_frame_outputs=obj_non_cond_outputs,
        )
    }
    singleton_state["temp_output_dict_per_obj"] = {
        0: create_output_store(
            cond_frame_outputs=temp_cond_outputs,
            non_cond_frame_outputs=temp_non_cond_outputs,
        )
    }


def _create_singleton_multiplex_state(model, source_state, obj_id):
    return model.multiplex_controller.get_state(
        num_valid_entries=1,
        device=source_state["device"],
        dtype=torch.float32,
        random=False,
        object_ids=[obj_id],
    )


def _clone_pos_enc(pos_enc):
    if pos_enc is None:
        return None

    return [None if level_enc is None else level_enc.clone() for level_enc in pos_enc]


def _prepare_singleton_outputs_for_new_state(model, singleton_outputs, multiplex_state):
    for storage_key in OUTPUT_KEYS:
        for frame_out in singleton_outputs[storage_key].values():
            if frame_out.get("maskmem_features") is not None:
                frame_out["maskmem_features"] = frame_out["maskmem_features"].clone()

            frame_out["maskmem_pos_enc"] = _clone_pos_enc(
                frame_out.get("maskmem_pos_enc")
            )

            if "obj_ptr" in frame_out and model.use_obj_ptrs_in_encoder:
                frame_out["obj_ptr"] = multiplex_state.mux(frame_out["obj_ptr"])


@torch.inference_mode()
def extract_object_for_interaction(model, inference_state, obj_id, frame_idx):
    source_state = inference_state
    obj_idx = source_state["obj_id_to_idx"][obj_id]

    singleton_outputs = _extract_consolidated_outputs(
        model,
        source_state,
        obj_id,
        obj_idx,
    )
    point_inputs = _copy_object_bucket(source_state, "point_inputs_per_obj", obj_idx)
    mask_inputs = _copy_object_bucket(source_state, "mask_inputs_per_obj", obj_idx)
    obj_outputs = _copy_object_outputs(source_state, "output_dict_per_obj", obj_idx)
    temp_outputs = _copy_object_outputs(
        source_state,
        "temp_output_dict_per_obj",
        obj_idx,
    )

    model.remove_object(
        source_state,
        obj_id,
        strict=False,
        need_output=False,
        clear_user_refined_map=False,
    )
    _clear_empty_multiplex_state(source_state)

    singleton_state = _init_singleton_state(
        model,
        source_state,
        obj_id,
        point_inputs,
        mask_inputs,
    )
    _set_singleton_output_stores(singleton_state, obj_outputs, temp_outputs)

    multiplex_state = _create_singleton_multiplex_state(model, source_state, obj_id)
    singleton_state["multiplex_state"] = multiplex_state
    _prepare_singleton_outputs_for_new_state(model, singleton_outputs, multiplex_state)
    singleton_state["output_dict"] = singleton_outputs

    return singleton_state
