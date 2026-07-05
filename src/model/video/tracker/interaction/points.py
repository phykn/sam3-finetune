import torch

from ..consolidation.merge import consolidate_temp_output_across_obj
from .point_output import (
    get_refine_or_gap_return_masks,
    run_gap_fill_point_object,
    store_new_point_output,
)
from .point_refinement import run_refine_point_object
from .point_setup import (
    ensure_point_multiplex_state,
    get_point_case,
    get_point_frame_context,
    prepare_point_inputs,
    run_new_point_object,
)


@torch.inference_mode()
def add_new_points(
    self,
    inference_state,
    frame_idx,
    obj_id,
    points,
    labels,
    clear_old_points,
    rel_coordinates=True,
    use_prev_mem_frame=False,
):
    obj_idx = self._obj_id_to_idx(inference_state, obj_id)
    obj_idxs = [obj_idx]
    obj_ids = [obj_id]

    point_inputs = prepare_point_inputs(
        self,
        inference_state,
        obj_idx=obj_idx,
        frame_idx=frame_idx,
        points=points,
        labels=labels,
        clear_old_points=clear_old_points,
        rel_coordinates=rel_coordinates,
    )
    context = get_point_frame_context(self, inference_state, frame_idx)
    multiplex_state, is_new_state = ensure_point_multiplex_state(
        self, inference_state, obj_ids
    )
    point_case = get_point_case(
        multiplex_state=multiplex_state,
        is_new_state=is_new_state,
        obj_id=obj_id,
        is_init_cond_frame=context["is_init_cond_frame"],
    )

    video_res_masks_to_return = run_point_frame(
        self,
        inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        obj_idx=obj_idx,
        obj_idxs=obj_idxs,
        obj_ids=obj_ids,
        point_inputs=point_inputs,
        point_case=point_case,
        is_new_state=is_new_state,
        context=context,
    )

    low_res_masks = None
    return frame_idx, obj_ids, low_res_masks, video_res_masks_to_return


def run_point_frame(
    self,
    inference_state,
    *,
    frame_idx,
    obj_id,
    obj_idx,
    obj_idxs,
    obj_ids,
    point_inputs,
    point_case,
    is_new_state,
    context,
):
    if point_case == "new_object":
        current_out = run_new_point_object(
            self,
            inference_state,
            frame_idx=frame_idx,
            point_inputs=point_inputs,
            obj_idxs=obj_idxs,
            obj_ids=obj_ids,
            is_new_state=is_new_state,
        )
        return store_new_point_output(
            self,
            inference_state,
            frame_idx=frame_idx,
            obj_idx=obj_idx,
            storage_key=context["storage_key"],
            is_cond=context["is_cond"],
            current_out=current_out,
        )

    if point_case == "refine":
        current_out, _ = run_refine_point_object(
            self,
            inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            point_inputs=point_inputs,
            is_cond=context["is_cond"],
        )
        return get_refine_or_gap_return_masks(
            self,
            inference_state,
            current_out,
            context["storage_key"],
            frame_idx,
        )

    obj_idx = inference_state["obj_id_to_idx"][obj_id]
    current_out = run_gap_fill_point_object(
        self,
        inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        obj_idx=obj_idx,
        point_inputs=point_inputs,
        storage_key=context["storage_key"],
    )
    return get_refine_or_gap_return_masks(
        self,
        inference_state,
        current_out,
        context["storage_key"],
        frame_idx,
    )


@torch.inference_mode()
def clear_all_points_in_frame(
    self,
    inference_state,
    frame_idx,
    obj_id,
    need_output=True,
    preserve_user_refined: bool = False,
):
    obj_idx = self._obj_id_to_idx(inference_state, obj_id)

    inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
    inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

    if not preserve_user_refined and "user_refined_frames_per_obj" in inference_state:
        user_refined_map = inference_state["user_refined_frames_per_obj"]
        if obj_id in user_refined_map:
            user_refined_map[obj_id].discard(frame_idx)

    temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
    temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
    temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

    batch_size = self._get_obj_num(inference_state)
    frame_has_input = False
    for obj_idx2 in range(batch_size):
        if obj_idx2 not in inference_state["point_inputs_per_obj"]:
            continue
        if obj_idx2 not in inference_state["mask_inputs_per_obj"]:
            continue
        if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
            frame_has_input = True
            break
        if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
            frame_has_input = True
            break

    if not frame_has_input:
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
        consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
        out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
        if out is not None:
            output_dict["non_cond_frame_outputs"][frame_idx] = out
            inference_state["frames_already_tracked"].pop(frame_idx, None)

        for obj_idx2 in range(batch_size):
            if obj_idx2 not in inference_state["output_dict_per_obj"]:
                continue
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
            obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if obj_out is not None:
                obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out

        if len(output_dict["cond_frame_outputs"]) == 0:
            self._reset_tracking_results(inference_state)

    if not need_output:
        return

    obj_ids = inference_state["obj_ids"]
    is_cond = any(
        frame_idx in obj_temp_output_dict["cond_frame_outputs"]
        for obj_temp_output_dict in temp_output_dict_per_obj.values()
    )
    consolidated_out = consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond=is_cond,
        run_mem_encoder=False,
        consolidate_at_video_res=True,
    )
    _, video_res_masks = self._get_orig_video_res_output(
        inference_state, consolidated_out["pred_masks_video_res"]
    )
    low_res_masks = None
    return frame_idx, obj_ids, low_res_masks, video_res_masks


@torch.inference_mode()
def clear_all_points_in_video(self, inference_state):
    self._reset_tracking_results(inference_state)
    inference_state["obj_id_to_idx"].clear()
    inference_state["obj_idx_to_id"].clear()
    inference_state["obj_ids"].clear()
    inference_state["point_inputs_per_obj"].clear()
    inference_state["mask_inputs_per_obj"].clear()
    inference_state["output_dict_per_obj"].clear()
    inference_state["temp_output_dict_per_obj"].clear()
    inference_state["multiplex_state"] = None
