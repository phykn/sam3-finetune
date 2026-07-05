import torch

from .extract import extract_object_for_interaction
from .merge import merge_singleton_interaction_result


def get_user_refined_frames(inference_state, obj_id):
    user_refined_frames_map = inference_state.get("user_refined_frames_per_obj", {})
    user_refined_frames = user_refined_frames_map.get(obj_id)
    return set() if user_refined_frames is None else user_refined_frames


def get_previous_refinement_logits(singleton_state, frame_idx, is_cond):
    storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
    singleton_output_dict = singleton_state["output_dict_per_obj"][0]
    singleton_temp_output_dict = singleton_state["temp_output_dict_per_obj"][0]

    prev_out = singleton_temp_output_dict[storage_key].get(frame_idx)
    if prev_out is None:
        prev_out = singleton_output_dict["cond_frame_outputs"].get(frame_idx)
    if prev_out is None:
        prev_out = singleton_output_dict["non_cond_frame_outputs"].get(frame_idx)
    if prev_out is None or prev_out["pred_masks"] is None:
        return None

    logits = prev_out["pred_masks"].to(
        singleton_state["device"],
        non_blocking=True,
    )
    return torch.clamp(logits, -32.0, 32.0)


def get_singleton_refinement_args(is_first_refinement, prev_logits):
    if is_first_refinement:
        return True, None
    return False, [0] if prev_logits is not None else None


def run_singleton_refinement(
    self,
    singleton_state,
    *,
    frame_idx,
    obj_id,
    point_inputs,
    is_init_cond_frame,
    prev_sam_mask_logits,
    objects_to_interact,
):
    current_out, _ = self._run_single_frame_inference(
        inference_state=singleton_state,
        output_dict=singleton_state["output_dict"],
        frame_idx=frame_idx,
        batch_size=1,
        is_init_cond_frame=is_init_cond_frame,
        point_inputs=point_inputs,
        mask_inputs=None,
        reverse=False,
        run_mem_encoder=False,
        prev_sam_mask_logits=prev_sam_mask_logits,
        add_to_existing_state=False,
        new_obj_idxs=[0],
        new_obj_ids=[obj_id],
        allow_new_buckets=False,
        objects_to_interact=objects_to_interact,
    )
    return current_out


def mark_user_refined_frame(inference_state, obj_id, frame_idx):
    if "user_refined_frames_per_obj" not in inference_state:
        inference_state["user_refined_frames_per_obj"] = {}
    if obj_id not in inference_state["user_refined_frames_per_obj"]:
        inference_state["user_refined_frames_per_obj"][obj_id] = set()

    inference_state["user_refined_frames_per_obj"][obj_id].add(frame_idx)


def store_refined_object_output(
    self,
    inference_state,
    *,
    obj_idx,
    frame_idx,
    storage_key,
):
    merged_frame_out = inference_state["output_dict"][storage_key][frame_idx]
    obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
    obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]

    if "pred_masks_video_res" in merged_frame_out:
        pred_masks_video_res_slice = merged_frame_out["pred_masks_video_res"][
            obj_idx : obj_idx + 1
        ]
    else:
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, merged_frame_out["pred_masks"]
        )
        pred_masks_video_res_slice = video_res_masks[obj_idx : obj_idx + 1]

    obj_temp_output_dict[storage_key][frame_idx] = {
        "pred_masks": merged_frame_out["pred_masks"][obj_idx : obj_idx + 1],
        "pred_masks_video_res": pred_masks_video_res_slice,
        "object_score_logits": merged_frame_out["object_score_logits"][
            obj_idx : obj_idx + 1
        ],
    }
    obj_output_dict[storage_key][frame_idx] = obj_temp_output_dict[storage_key][
        frame_idx
    ]


def run_refine_point_object(
    self,
    inference_state,
    *,
    frame_idx,
    obj_id,
    point_inputs,
    is_cond,
):
    singleton_state = extract_object_for_interaction(
        self, inference_state, obj_id, frame_idx
    )

    user_refined_frames = get_user_refined_frames(inference_state, obj_id)
    is_first_refinement = frame_idx not in user_refined_frames

    prev_logits = None
    if not is_first_refinement:
        prev_logits = get_previous_refinement_logits(
            singleton_state, frame_idx, is_cond
        )

    singleton_is_init_cond, objects_to_interact = get_singleton_refinement_args(
        is_first_refinement, prev_logits
    )
    current_out = run_singleton_refinement(
        self,
        singleton_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        point_inputs=point_inputs,
        is_init_cond_frame=singleton_is_init_cond,
        prev_sam_mask_logits=prev_logits,
        objects_to_interact=objects_to_interact,
    )

    storage_key = (
        "cond_frame_outputs" if singleton_is_init_cond else "non_cond_frame_outputs"
    )
    _, video_res_masks = self._get_orig_video_res_output(
        singleton_state, current_out["pred_masks"]
    )
    current_out["pred_masks_video_res"] = video_res_masks
    singleton_state["output_dict"][storage_key][frame_idx] = current_out

    merge_singleton_interaction_result(self, inference_state, singleton_state, obj_id)
    obj_idx = inference_state["obj_id_to_idx"][obj_id]

    mark_user_refined_frame(inference_state, obj_id, frame_idx)
    store_refined_object_output(
        self,
        inference_state,
        obj_idx=obj_idx,
        frame_idx=frame_idx,
        storage_key=storage_key,
    )

    return current_out, obj_idx
