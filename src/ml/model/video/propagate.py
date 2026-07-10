from copy import deepcopy

import torch
from tqdm import tqdm

from ...components.video.tracker.consolidation.merge import (
    consolidate_temp_output_across_obj,
)
from .state import forward_frames


@torch.inference_mode()
def preflight(model, state, run_mem_encoder=True):
    state["tracking_has_started"] = True
    batch_size = model._get_obj_num(state)
    temp_outputs = state["temp_output_dict_per_obj"]
    outputs = state["output_dict"]
    consolidated = state["consolidated_frame_inds"]

    for is_cond in (False, True):
        key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        frame_indices = set()
        for output in temp_outputs.values():
            frame_indices.update(output[key])
        consolidated[key].update(frame_indices)

        for frame_idx in frame_indices:
            current = consolidate_temp_output_across_obj(
                model,
                state,
                frame_idx,
                is_cond=is_cond,
                run_mem_encoder=run_mem_encoder,
            )
            outputs[key][frame_idx] = current
            model._add_output_per_object(state, frame_idx, current, key)
            if model.clear_non_cond_mem_around_input and (
                model.clear_non_cond_mem_for_multi_obj or batch_size <= 1
            ):
                model._clear_non_cond_mem_around_input(state, frame_idx)

        for output in temp_outputs.values():
            output[key].clear()

    for frame_idx in outputs["cond_frame_outputs"]:
        outputs["non_cond_frame_outputs"].pop(frame_idx, None)
    for output in state["output_dict_per_obj"].values():
        for frame_idx in output["cond_frame_outputs"]:
            output["non_cond_frame_outputs"].pop(frame_idx, None)
    for frame_idx in consolidated["cond_frame_outputs"]:
        consolidated["non_cond_frame_outputs"].discard(frame_idx)

    all_consolidated = (
        consolidated["cond_frame_outputs"] | consolidated["non_cond_frame_outputs"]
    )
    input_frames = set()
    for inputs in state["mask_inputs_per_obj"].values():
        input_frames.update(inputs)
    if all_consolidated != input_frames:
        raise RuntimeError("consolidated frames do not match mask input frames")

    if state["first_ann_frame_idx"] is None:
        state["first_ann_frame_idx"] = min(input_frames, default=None)
    if state["first_ann_frame_idx"] not in outputs["cond_frame_outputs"]:
        state["first_ann_frame_idx"] = min(
            outputs["cond_frame_outputs"],
            default=None,
        )


@torch.inference_mode()
def propagate(
    model,
    state,
    start_frame_idx=None,
    max_frame_num_to_track=None,
    tqdm_disable=False,
    run_mem_encoder=True,
):
    outputs = state["output_dict"]
    consolidated = state["consolidated_frame_inds"]
    if not outputs["cond_frame_outputs"]:
        raise RuntimeError("add at least one mask before tracking")

    if start_frame_idx is None:
        start_frame_idx = min(outputs["cond_frame_outputs"])
    if max_frame_num_to_track is None:
        max_frame_num_to_track = state["num_frames"] - start_frame_idx

    obj_ids = list(state["obj_ids"])
    batch_size = model._get_obj_num(state)
    order = forward_frames(
        start_frame_idx,
        max_frame_num_to_track,
        state["num_frames"],
    )
    for frame_idx in tqdm(order, desc="propagate in video", disable=tqdm_disable):
        if frame_idx in consolidated["cond_frame_outputs"]:
            key = "cond_frame_outputs"
            current = outputs[key][frame_idx]
            pred_masks = current["pred_masks"]
            object_logits = current["object_score_logits"]
        elif frame_idx in consolidated["non_cond_frame_outputs"]:
            key = "non_cond_frame_outputs"
            current = outputs[key][frame_idx]
            pred_masks = current["pred_masks"]
            object_logits = current["object_score_logits"]
        else:
            key = "non_cond_frame_outputs"
            current, pred_masks = model._run_single_frame_inference(
                inference_state=state,
                output_dict=outputs,
                frame_idx=frame_idx,
                batch_size=batch_size,
                is_init_cond_frame=False,
                mask_inputs=None,
                run_mem_encoder=run_mem_encoder,
            )
            object_logits = current["object_score_logits"]
            current["local_obj_id_to_idx"] = deepcopy(state["obj_id_to_idx"])
            outputs[key][frame_idx] = current

        model._add_output_per_object(state, frame_idx, current, key)
        state["frames_already_tracked"][frame_idx] = True
        low_res_masks, video_res_masks = model._get_orig_video_res_output(
            state,
            pred_masks,
        )
        yield frame_idx, obj_ids, low_res_masks, video_res_masks, object_logits
