from copy import deepcopy

import torch
from tqdm import tqdm

from ..consolidation.merge import consolidate_temp_output_across_obj


@torch.inference_mode()
def propagate_in_video_preflight(self, inference_state, run_mem_encoder=True):
    inference_state["tracking_has_started"] = True
    batch_size = self._get_obj_num(inference_state)

    temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
    output_dict = inference_state["output_dict"]
    consolidated_frame_inds = inference_state["consolidated_frame_inds"]
    for is_cond in [False, True]:
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        temp_frame_inds = set()
        for obj_temp_output_dict in temp_output_dict_per_obj.values():
            temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
        consolidated_frame_inds[storage_key].update(temp_frame_inds)
        for frame_idx in temp_frame_inds:
            consolidated_out = consolidate_temp_output_across_obj(
                self,
                inference_state,
                frame_idx,
                is_cond=is_cond,
                run_mem_encoder=run_mem_encoder,
            )
            output_dict[storage_key][frame_idx] = consolidated_out
            self._add_output_per_object(
                inference_state, frame_idx, consolidated_out, storage_key
            )
            clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
            )
            if clear_non_cond_mem:
                self._clear_non_cond_mem_around_input(inference_state, frame_idx)

        for obj_temp_output_dict in temp_output_dict_per_obj.values():
            obj_temp_output_dict[storage_key].clear()

    for frame_idx in output_dict["cond_frame_outputs"]:
        output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
    for obj_output_dict in inference_state["output_dict_per_obj"].values():
        for frame_idx in obj_output_dict["cond_frame_outputs"]:
            obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
    for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
        assert frame_idx in output_dict["cond_frame_outputs"]
        consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

    all_consolidated_frame_inds = (
        consolidated_frame_inds["cond_frame_outputs"]
        | consolidated_frame_inds["non_cond_frame_outputs"]
    )

    input_frames_inds = set()
    for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
        input_frames_inds.update(point_inputs_per_frame.keys())
    for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
        input_frames_inds.update(mask_inputs_per_frame.keys())
    assert all_consolidated_frame_inds == input_frames_inds
    if inference_state["first_ann_frame_idx"] is None:
        inference_state["first_ann_frame_idx"] = min(input_frames_inds, default=None)
    if inference_state["first_ann_frame_idx"] not in output_dict["cond_frame_outputs"]:
        inference_state["first_ann_frame_idx"] = min(
            output_dict["cond_frame_outputs"], default=None
        )


def get_processing_order(
    self, inference_state, start_frame_idx, max_frame_num_to_track, reverse
):
    num_frames = inference_state["num_frames"]
    if self.always_start_from_first_ann_frame:
        start_frame_idx = inference_state["first_ann_frame_idx"]
    if start_frame_idx is None:
        start_frame_idx = min(inference_state["output_dict"]["cond_frame_outputs"])
    if max_frame_num_to_track is None:
        max_frame_num_to_track = num_frames
    if reverse:
        end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
        if start_frame_idx > 0:
            processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
        else:
            processing_order = [0]
    else:
        end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
        processing_order = range(start_frame_idx, end_frame_idx + 1)
    return processing_order


@torch.inference_mode()
def propagate_in_video(
    self,
    inference_state,
    start_frame_idx,
    max_frame_num_to_track,
    reverse,
    tqdm_disable=False,
    obj_ids=None,
    run_mem_encoder=True,
):
    output_dict = inference_state["output_dict"]
    consolidated_frame_inds = inference_state["consolidated_frame_inds"]
    if obj_ids is not None:
        raise NotImplementedError(
            "Per-object tracking yet for batched inference if not implemented."
        )
    obj_ids = inference_state["obj_ids"]
    batch_size = self._get_obj_num(inference_state)
    if len(output_dict["cond_frame_outputs"]) == 0:
        raise RuntimeError("No points are provided; please add points first")
    clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
        self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
    )
    assert clear_non_cond_mem is False, "Not implemented"

    processing_order = self._get_processing_order(
        inference_state,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
    )

    for frame_idx in tqdm(
        processing_order, desc="propagate in video", disable=tqdm_disable
    ):
        if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame_idx]
            pred_masks = current_out["pred_masks"]
            if clear_non_cond_mem:
                self._clear_non_cond_mem_around_input(inference_state, frame_idx)
        elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame_idx]
            pred_masks = current_out["pred_masks"]
        else:
            storage_key = "non_cond_frame_outputs"
            with torch.profiler.record_function(
                "VideoTrackingMultiplexDemo._run_single_frame_inference"
            ):
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=run_mem_encoder,
                )
            current_out["local_obj_id_to_idx"] = deepcopy(
                inference_state["obj_id_to_idx"]
            )
            output_dict[storage_key][frame_idx] = current_out
        self._add_output_per_object(
            inference_state, frame_idx, current_out, storage_key
        )
        inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

        low_res_masks, video_res_masks = self._get_orig_video_res_output(
            inference_state, pred_masks
        )
        yield frame_idx, obj_ids, low_res_masks, video_res_masks


@torch.inference_mode()
def propagate_sam3_in_video(
    self,
    inference_state,
    start_frame_idx,
    max_frame_num_to_track,
    reverse,
    tqdm_disable=False,
    obj_ids=None,
    run_mem_encoder=True,
):
    output_dict = inference_state["output_dict"]
    consolidated_frame_inds = inference_state["consolidated_frame_inds"]
    if obj_ids is not None:
        raise NotImplementedError(
            "Per-object tracking yet for batched inference if not implemented."
        )
    obj_ids = inference_state["obj_ids"]
    batch_size = self._get_obj_num(inference_state)
    if len(output_dict["cond_frame_outputs"]) == 0:
        raise RuntimeError("No points are provided; please add points first")
    clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
        self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
    )

    processing_order = self._get_processing_order(
        inference_state,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
    )

    for frame_idx in tqdm(
        processing_order, desc="propagate in video", disable=tqdm_disable
    ):
        if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame_idx]
            pred_masks = current_out["pred_masks"]
            obj_scores = current_out["object_score_logits"]
            if clear_non_cond_mem:
                self._clear_non_cond_mem_around_input(inference_state, frame_idx)
        elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame_idx]
            pred_masks = current_out["pred_masks"]
            obj_scores = current_out["object_score_logits"]
        else:
            storage_key = "non_cond_frame_outputs"
            with torch.profiler.record_function(
                "VideoTrackingMultiplexDemo._run_single_frame_inference"
            ):
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=run_mem_encoder,
                )
                obj_scores = current_out["object_score_logits"]
                current_out["local_obj_id_to_idx"] = deepcopy(
                    inference_state["obj_id_to_idx"]
                )
            output_dict[storage_key][frame_idx] = current_out

        self._add_output_per_object(
            inference_state, frame_idx, current_out, storage_key
        )
        inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

        low_res_masks, video_res_masks = self._get_orig_video_res_output(
            inference_state, pred_masks
        )
        yield frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores
