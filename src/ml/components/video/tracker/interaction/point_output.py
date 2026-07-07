from copy import deepcopy


def remove_stale_non_cond_output(inference_state, frame_idx, storage_key):
    if storage_key != "cond_frame_outputs":
        return
    if frame_idx not in inference_state["output_dict"]["non_cond_frame_outputs"]:
        return

    del inference_state["output_dict"]["non_cond_frame_outputs"][frame_idx]
    if "consolidated_frame_inds" in inference_state:
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].discard(
            frame_idx
        )


def add_consolidated_frame(inference_state, frame_idx, storage_key):
    if "consolidated_frame_inds" in inference_state:
        inference_state["consolidated_frame_inds"][storage_key].add(frame_idx)


def write_object_frame_output(
    inference_state,
    *,
    obj_idx,
    frame_idx,
    storage_key,
    pred_masks,
    pred_masks_video_res,
    object_score_logits,
):
    obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
    obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
    obj_temp_output_dict[storage_key][frame_idx] = {
        "pred_masks": pred_masks[obj_idx : obj_idx + 1],
        "pred_masks_video_res": pred_masks_video_res[obj_idx : obj_idx + 1],
        "object_score_logits": object_score_logits[obj_idx : obj_idx + 1],
    }
    obj_output_dict[storage_key][frame_idx] = obj_temp_output_dict[storage_key][
        frame_idx
    ]


def run_gap_fill_point_object(
    self,
    inference_state,
    *,
    frame_idx,
    obj_id,
    obj_idx,
    point_inputs,
    storage_key,
):
    current_out, _ = self._run_single_frame_inference(
        inference_state=inference_state,
        output_dict=inference_state["output_dict"],
        frame_idx=frame_idx,
        batch_size=self._get_obj_num(inference_state),
        is_init_cond_frame=True,
        point_inputs=point_inputs,
        mask_inputs=None,
        reverse=False,
        run_mem_encoder=False,
        prev_sam_mask_logits=None,
        add_to_existing_state=False,
        new_obj_idxs=[obj_idx],
        new_obj_ids=[obj_id],
        allow_new_buckets=False,
        prefer_new_buckets=False,
        objects_to_interact=[obj_idx],
    )

    current_out["local_obj_id_to_idx"] = deepcopy(inference_state["obj_id_to_idx"])
    _, video_res_masks = self._get_orig_video_res_output(
        inference_state, current_out["pred_masks"]
    )
    current_out["pred_masks_video_res"] = video_res_masks

    remove_stale_non_cond_output(inference_state, frame_idx, storage_key)
    inference_state["output_dict"][storage_key][frame_idx] = current_out
    add_consolidated_frame(inference_state, frame_idx, storage_key)
    write_object_frame_output(
        inference_state,
        obj_idx=obj_idx,
        frame_idx=frame_idx,
        storage_key=storage_key,
        pred_masks=current_out["pred_masks"],
        pred_masks_video_res=video_res_masks,
        object_score_logits=current_out["object_score_logits"],
    )

    return current_out


def store_new_point_output(
    self,
    inference_state,
    *,
    frame_idx,
    obj_idx,
    storage_key,
    is_cond,
    current_out,
):
    _, video_res_masks = self._get_orig_video_res_output(
        inference_state, current_out["pred_masks"]
    )
    current_out["pred_masks_video_res"] = video_res_masks
    current_out["local_obj_id_to_idx"] = deepcopy(inference_state["obj_id_to_idx"])

    if is_cond:
        remove_stale_non_cond_output(inference_state, frame_idx, storage_key)

    inference_state["output_dict"][storage_key][frame_idx] = current_out
    add_consolidated_frame(inference_state, frame_idx, storage_key)
    write_object_frame_output(
        inference_state,
        obj_idx=obj_idx,
        frame_idx=frame_idx,
        storage_key=storage_key,
        pred_masks=current_out["pred_masks"],
        pred_masks_video_res=current_out["pred_masks_video_res"],
        object_score_logits=current_out["object_score_logits"],
    )

    return video_res_masks[obj_idx : obj_idx + 1]


def get_refine_or_gap_return_masks(
    self, inference_state, current_out, storage_key, frame_idx
):
    _, video_res_masks = self._get_orig_video_res_output(
        inference_state, current_out["pred_masks"]
    )
    add_consolidated_frame(inference_state, frame_idx, storage_key)
    return video_res_masks[0:1]
