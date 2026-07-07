from copy import deepcopy

from ..outputs import NO_OBJ_SCORE
from ..state import create_output_store, OUTPUT_KEYS
from .tensor import (
    demux_if_needed,
    filled_object_tensor,
    pad_first_dim,
    tensor_with_object_row,
)


def _mux_singleton_row(
    singleton_tensor, singleton_state, target_state, num_objs, obj_idx
):
    if singleton_tensor is None or target_state is None:
        return None

    singleton_data = demux_if_needed(singleton_tensor, singleton_state)
    data = tensor_with_object_row(singleton_data, num_objs, obj_idx)
    return target_state.mux(data)


def _merge_muxed_row(
    main_frame_out,
    key,
    singleton_frame_out,
    singleton_state,
    target_state,
    num_objs,
    obj_idx,
):
    singleton_tensor = singleton_frame_out.get(key)
    if singleton_tensor is None or target_state is None:
        return

    singleton_data = demux_if_needed(singleton_tensor, singleton_state)

    existing_tensor = main_frame_out.get(key)
    if existing_tensor is None:
        existing_data = filled_object_tensor(num_objs, singleton_data)
    else:
        existing_data = demux_if_needed(existing_tensor, target_state)
        existing_data = pad_first_dim(existing_data, num_objs)

    existing_data[obj_idx : obj_idx + 1] = singleton_data
    main_frame_out[key] = target_state.mux(existing_data)


def _mux_maskmem_pos_enc(
    singleton_frame_out, singleton_state, target_state, num_objs, obj_idx
):
    if singleton_frame_out.get("maskmem_pos_enc") is None or target_state is None:
        return None

    pos_enc = []
    for level_enc in singleton_frame_out["maskmem_pos_enc"]:
        if level_enc is None:
            pos_enc.append(None)
            continue

        level_data = demux_if_needed(level_enc, singleton_state)
        level_tensor = tensor_with_object_row(level_data, num_objs, obj_idx)
        pos_enc.append(target_state.mux(level_tensor))

    return pos_enc


def _merge_maskmem_pos_enc(
    main_frame_out,
    singleton_frame_out,
    singleton_state,
    target_state,
    num_objs,
    obj_idx,
):
    if singleton_frame_out.get("maskmem_pos_enc") is None or target_state is None:
        return

    existing_pos_enc = main_frame_out.get("maskmem_pos_enc") or []
    merged_pos_enc = []
    max_levels = max(len(singleton_frame_out["maskmem_pos_enc"]), len(existing_pos_enc))

    for level_idx in range(max_levels):
        singleton_level = (
            singleton_frame_out["maskmem_pos_enc"][level_idx]
            if level_idx < len(singleton_frame_out["maskmem_pos_enc"])
            else None
        )
        existing_level = (
            existing_pos_enc[level_idx] if level_idx < len(existing_pos_enc) else None
        )

        if singleton_level is None:
            merged_pos_enc.append(existing_level)
            continue

        singleton_data = demux_if_needed(singleton_level, singleton_state)
        if existing_level is None:
            existing_data = filled_object_tensor(num_objs, singleton_data)
        else:
            existing_data = demux_if_needed(existing_level, target_state)
            existing_data = pad_first_dim(existing_data, num_objs)

        existing_data[obj_idx : obj_idx + 1] = singleton_data
        merged_pos_enc.append(target_state.mux(existing_data))

    main_frame_out["maskmem_pos_enc"] = merged_pos_enc


def _new_frame_from_singleton(
    model,
    singleton_frame_out,
    singleton_state,
    target_state,
    obj_id,
    obj_idx,
    num_objs,
):
    singleton_obj_idx = 0
    frame_out = {
        "maskmem_features": _mux_singleton_row(
            singleton_frame_out.get("maskmem_features"),
            singleton_state,
            target_state,
            num_objs,
            obj_idx,
        ),
        "maskmem_pos_enc": _mux_maskmem_pos_enc(
            singleton_frame_out,
            singleton_state,
            target_state,
            num_objs,
            obj_idx,
        ),
        "image_features": singleton_frame_out.get("image_features"),
        "image_pos_enc": singleton_frame_out.get("image_pos_enc"),
        "local_obj_id_to_idx": {obj_id: obj_idx},
        "conditioning_objects": (
            {obj_idx}
            if singleton_obj_idx
            in singleton_frame_out.get("conditioning_objects", set())
            else set()
        ),
        "pred_masks": tensor_with_object_row(
            singleton_frame_out["pred_masks"],
            num_objs,
            obj_idx,
        ),
        "object_score_logits": tensor_with_object_row(
            singleton_frame_out["object_score_logits"],
            num_objs,
            obj_idx,
            fill_value=NO_OBJ_SCORE,
        ),
    }

    if "pred_masks_video_res" in singleton_frame_out:
        frame_out["pred_masks_video_res"] = tensor_with_object_row(
            singleton_frame_out["pred_masks_video_res"],
            num_objs,
            obj_idx,
        )

    if "obj_ptr" in singleton_frame_out and model.use_obj_ptrs_in_encoder:
        obj_ptr_data = singleton_state.demux(singleton_frame_out["obj_ptr"])
        frame_out["obj_ptr"] = target_state.mux(
            tensor_with_object_row(obj_ptr_data, num_objs, obj_idx)
        )

    return frame_out


def _merge_masks_and_scores(main_frame_out, singleton_frame_out, obj_idx):
    masks = singleton_frame_out["pred_masks"]
    scores = singleton_frame_out["object_score_logits"]

    num_objs = obj_idx + 1
    main_frame_out["pred_masks"] = pad_first_dim(
        main_frame_out["pred_masks"],
        num_objs,
    )
    main_frame_out["object_score_logits"] = pad_first_dim(
        main_frame_out["object_score_logits"],
        num_objs,
        fill_value=NO_OBJ_SCORE,
    )

    main_frame_out["pred_masks"][obj_idx : obj_idx + 1] = masks
    main_frame_out["object_score_logits"][obj_idx : obj_idx + 1] = scores


def _merge_video_res_masks(main_frame_out, singleton_frame_out, obj_idx, num_objs):
    if "pred_masks_video_res" not in singleton_frame_out:
        return

    masks = singleton_frame_out["pred_masks_video_res"]
    if "pred_masks_video_res" in main_frame_out:
        main_frame_out["pred_masks_video_res"] = pad_first_dim(
            main_frame_out["pred_masks_video_res"],
            obj_idx + 1,
        )
    else:
        main_frame_out["pred_masks_video_res"] = filled_object_tensor(num_objs, masks)

    main_frame_out["pred_masks_video_res"][obj_idx : obj_idx + 1] = masks


def _merge_obj_ptr(
    model,
    main_frame_out,
    singleton_frame_out,
    singleton_state,
    target_state,
    obj_idx,
    num_objs,
):
    if "obj_ptr" not in singleton_frame_out or not model.use_obj_ptrs_in_encoder:
        return

    singleton_obj_ptr_data = singleton_state.demux(singleton_frame_out["obj_ptr"])
    if "obj_ptr" not in main_frame_out:
        obj_ptr_data = tensor_with_object_row(
            singleton_obj_ptr_data,
            num_objs,
            obj_idx,
        )
        main_frame_out["obj_ptr"] = target_state.mux(obj_ptr_data)
        return

    old_obj_ptr = main_frame_out["obj_ptr"]
    if old_obj_ptr.shape[1] != target_state.num_buckets:
        obj_ptr_data = tensor_with_object_row(
            singleton_obj_ptr_data,
            num_objs,
            obj_idx,
        )
        main_frame_out["obj_ptr"] = target_state.mux(obj_ptr_data)
        return

    obj_ptr_data = target_state.demux(old_obj_ptr)
    obj_ptr_data = pad_first_dim(obj_ptr_data, obj_idx + 1)
    obj_ptr_data[obj_idx : obj_idx + 1] = singleton_obj_ptr_data
    main_frame_out["obj_ptr"] = target_state.mux(obj_ptr_data)


def _merge_existing_frame(
    model,
    inference_state,
    main_frame_out,
    singleton_frame_out,
    singleton_state,
    target_state,
    obj_id,
    obj_idx,
):
    num_objs = len(inference_state["obj_ids"])

    _merge_muxed_row(
        main_frame_out,
        "maskmem_features",
        singleton_frame_out,
        singleton_state,
        target_state,
        num_objs,
        obj_idx,
    )
    _merge_maskmem_pos_enc(
        main_frame_out,
        singleton_frame_out,
        singleton_state,
        target_state,
        num_objs,
        obj_idx,
    )
    _merge_masks_and_scores(main_frame_out, singleton_frame_out, obj_idx)

    if "local_obj_id_to_idx" not in main_frame_out:
        main_frame_out["local_obj_id_to_idx"] = deepcopy(
            inference_state["obj_id_to_idx"]
        )
    main_frame_out["local_obj_id_to_idx"][obj_id] = obj_idx

    _merge_video_res_masks(main_frame_out, singleton_frame_out, obj_idx, num_objs)
    _merge_obj_ptr(
        model,
        main_frame_out,
        singleton_frame_out,
        singleton_state,
        target_state,
        obj_idx,
        num_objs,
    )

    if 0 in singleton_frame_out.get("conditioning_objects", set()):
        main_frame_out["conditioning_objects"].add(obj_idx)


def merge_output_dict(
    model, inference_state, singleton_state, obj_id, obj_idx, target_state
):
    if "output_dict" not in singleton_state:
        return

    singleton_state_mux = singleton_state.get("multiplex_state")
    for storage_key in OUTPUT_KEYS:
        singleton_outputs = singleton_state["output_dict"].get(storage_key, {})
        if not singleton_outputs:
            continue

        if "output_dict" not in inference_state:
            inference_state["output_dict"] = create_output_store()

        target_outputs = inference_state["output_dict"][storage_key]
        for frame_idx, singleton_frame_out in singleton_outputs.items():
            if frame_idx not in target_outputs:
                num_objs = max(len(inference_state["obj_ids"]), obj_idx + 1)
                target_outputs[frame_idx] = _new_frame_from_singleton(
                    model,
                    singleton_frame_out,
                    singleton_state_mux,
                    target_state,
                    obj_id,
                    obj_idx,
                    num_objs,
                )
                continue

            _merge_existing_frame(
                model,
                inference_state,
                target_outputs[frame_idx],
                singleton_frame_out,
                singleton_state_mux,
                target_state,
                obj_id,
                obj_idx,
            )
