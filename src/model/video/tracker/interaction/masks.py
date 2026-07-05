from copy import deepcopy

import numpy as np
import torch

from ..consolidation.merge import consolidate_temp_output_across_obj
from ..outputs import NO_OBJ_SCORE


@torch.inference_mode()
def add_new_masks(
    self,
    inference_state,
    frame_idx,
    obj_ids,
    masks,
    add_mask_to_memory=False,
    reconditioning=False,
):
    obj_ids = _normalize_obj_ids(obj_ids)
    obj_idxs = _resolve_obj_idxs(self, inference_state, obj_ids, reconditioning)
    mask_inputs, mask_inputs_video_res = _prepare_masks(
        self,
        inference_state,
        masks,
        len(obj_ids),
    )
    multiplex_state, is_new_state = _ensure_multiplex_state(
        self,
        inference_state,
        obj_ids,
        len(obj_ids),
        reconditioning,
    )
    _store_mask_inputs(inference_state, frame_idx, obj_idxs, mask_inputs_video_res)

    context = get_mask_frame_context(self, inference_state, frame_idx)
    current_out = run_mask_frame(
        self,
        inference_state,
        frame_idx=frame_idx,
        obj_idxs=obj_idxs,
        obj_ids=obj_ids,
        mask_inputs=mask_inputs,
        multiplex_state=multiplex_state,
        is_new_state=is_new_state,
        reconditioning=reconditioning,
        context=context,
    )
    _write_video_res_masks(
        self,
        inference_state,
        current_out,
        obj_idxs,
        mask_inputs_video_res,
    )
    _store_frame_output(
        inference_state,
        frame_idx,
        context["storage_key"],
        context["is_cond"],
        current_out,
    )
    _store_object_outputs(
        inference_state,
        frame_idx,
        context["storage_key"],
        obj_idxs,
        current_out,
    )
    _suppress_overlapping_outputs(
        inference_state,
        frame_idx,
        context["storage_key"],
        obj_idxs,
        mask_inputs_video_res,
    )

    video_res_masks = get_consolidated_mask_return(
        self,
        inference_state,
        frame_idx,
        is_cond=context["is_cond"],
        current_out=current_out,
    )
    low_res_masks = None

    return frame_idx, inference_state["obj_ids"], low_res_masks, video_res_masks


def get_consolidated_mask_return(
    self,
    inference_state,
    frame_idx,
    *,
    is_cond,
    current_out,
):
    consolidated_out = consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond=is_cond,
        run_mem_encoder=False,
        consolidate_at_video_res=True,
    )
    _, video_res_masks = self._get_orig_video_res_output(
        inference_state,
        consolidated_out["pred_masks_video_res"],
    )
    consolidated_out["local_obj_id_to_idx"] = current_out["local_obj_id_to_idx"]
    return video_res_masks


def get_mask_frame_context(self, inference_state, frame_idx):
    is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
    if is_init_cond_frame:
        reverse = False
    else:
        reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]

    is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
    storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
    return {
        "is_init_cond_frame": is_init_cond_frame,
        "is_cond": is_cond,
        "reverse": reverse,
        "storage_key": storage_key,
    }


def run_mask_frame(
    self,
    inference_state,
    *,
    frame_idx,
    obj_idxs,
    obj_ids,
    mask_inputs,
    multiplex_state,
    is_new_state,
    reconditioning,
    context,
):
    current_out, _ = self._run_single_frame_inference(
        inference_state=inference_state,
        output_dict=inference_state["output_dict"],
        frame_idx=frame_idx,
        batch_size=len(obj_ids),
        is_init_cond_frame=context["is_init_cond_frame"],
        point_inputs=None,
        mask_inputs=mask_inputs,
        reverse=context["reverse"],
        run_mem_encoder=False,
        add_to_existing_state=_should_add_to_existing_state(
            inference_state,
            frame_idx,
            is_new_state,
            reconditioning,
        ),
        new_obj_idxs=obj_idxs,
        new_obj_ids=obj_ids,
        allow_new_buckets=_allow_new_buckets(
            multiplex_state,
            len(obj_ids),
            is_new_state,
            reconditioning,
        ),
        reconditioning=reconditioning,
    )
    return current_out


def _normalize_obj_ids(obj_ids):
    if isinstance(obj_ids, np.ndarray):
        return obj_ids.tolist()
    return obj_ids


def _resolve_obj_idxs(self, inference_state, obj_ids, reconditioning):
    return [
        self._obj_id_to_idx(inference_state, obj_id, error_if_new=reconditioning)
        for obj_id in obj_ids
    ]


def _prepare_masks(self, inference_state, masks, num_objects):
    assert masks.dim() == 3
    assert masks.shape[0] == num_objects

    mask_inputs = masks[:, None].float().to(inference_state["device"])
    model_masks = _resize_masks(
        mask_inputs,
        (self.input_mask_size, self.input_mask_size),
    )
    video_masks = _resize_masks(
        mask_inputs,
        (inference_state["video_height"], inference_state["video_width"]),
    )
    return model_masks, video_masks > 0.5


def _resize_masks(masks, size):
    if masks.shape[-2:] == size:
        return masks

    return torch.nn.functional.interpolate(
        masks,
        size=size,
        align_corners=False,
        mode="bilinear",
        antialias=True,
    )


def _ensure_multiplex_state(
    self, inference_state, obj_ids, num_objects, reconditioning
):
    multiplex_state = inference_state["multiplex_state"]
    is_new_state = multiplex_state is None

    if reconditioning:
        return multiplex_state, is_new_state

    if is_new_state:
        multiplex_state = self.multiplex_controller.get_state(
            num_valid_entries=num_objects,
            device=inference_state["device"],
            dtype=torch.float32,
            random=False,
            object_ids=obj_ids,
        )
        inference_state["multiplex_state"] = multiplex_state
    else:
        assert self.is_dynamic_model, "New objects are not allowed after state creation"

    return multiplex_state, is_new_state


def _store_mask_inputs(inference_state, frame_idx, obj_idxs, mask_inputs_video_res):
    for index, obj_idx in enumerate(obj_idxs):
        inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = (
            mask_inputs_video_res[index : index + 1]
        )
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)


def _allow_new_buckets(multiplex_state, num_objects, is_new_state, reconditioning):
    if is_new_state or reconditioning or multiplex_state is None:
        return False
    return multiplex_state.available_slots < num_objects


def _should_add_to_existing_state(
    inference_state,
    frame_idx,
    is_new_state,
    reconditioning,
):
    if is_new_state or reconditioning:
        return False

    return (
        frame_idx in inference_state["output_dict"]["cond_frame_outputs"]
        or frame_idx in inference_state["output_dict"]["non_cond_frame_outputs"]
    )


def _write_video_res_masks(
    self,
    inference_state,
    current_out,
    obj_idxs,
    mask_inputs_video_res,
):
    _, video_res_masks = self._get_orig_video_res_output(
        inference_state, current_out["pred_masks"]
    )
    obj_idxs_t = torch.as_tensor(obj_idxs, device=video_res_masks.device)
    video_res_masks[obj_idxs_t] = torch.where(
        mask_inputs_video_res,
        -NO_OBJ_SCORE,
        NO_OBJ_SCORE,
    )
    current_out["pred_masks_video_res"] = video_res_masks
    current_out["local_obj_id_to_idx"] = deepcopy(inference_state["obj_id_to_idx"])


def _store_frame_output(inference_state, frame_idx, storage_key, is_cond, current_out):
    if (
        is_cond
        and frame_idx in inference_state["output_dict"]["non_cond_frame_outputs"]
    ):
        del inference_state["output_dict"]["non_cond_frame_outputs"][frame_idx]
        if "consolidated_frame_inds" in inference_state:
            inference_state["consolidated_frame_inds"][
                "non_cond_frame_outputs"
            ].discard(frame_idx)

    inference_state["output_dict"][storage_key][frame_idx] = current_out
    if "consolidated_frame_inds" in inference_state:
        inference_state["consolidated_frame_inds"][storage_key].add(frame_idx)


def _store_object_outputs(
    inference_state,
    frame_idx,
    storage_key,
    obj_idxs,
    current_out,
):
    for obj_idx in obj_idxs:
        obj_frame_out = {
            "pred_masks_video_res": current_out["pred_masks_video_res"][
                obj_idx : obj_idx + 1
            ]
        }
        inference_state["temp_output_dict_per_obj"][obj_idx][storage_key][
            frame_idx
        ] = obj_frame_out
        inference_state["output_dict_per_obj"][obj_idx][storage_key][
            frame_idx
        ] = obj_frame_out


def _suppress_overlapping_outputs(
    inference_state,
    frame_idx,
    storage_key,
    obj_idxs,
    mask_inputs_video_res,
):
    combined_new_mask = mask_inputs_video_res.any(dim=0, keepdim=True)
    exclude_self_masks = _make_exclude_self_masks(obj_idxs, mask_inputs_video_res)
    obj_idxs_set = set(obj_idxs)

    for obj_idx, output_dict in inference_state["temp_output_dict_per_obj"].items():
        current_out = output_dict[storage_key].get(frame_idx)
        if current_out is None:
            continue

        if obj_idx not in obj_idxs_set:
            suppress_mask = combined_new_mask
        elif obj_idx in exclude_self_masks:
            suppress_mask = exclude_self_masks[obj_idx]
        else:
            continue

        current_out["pred_masks_video_res"] = torch.where(
            suppress_mask,
            NO_OBJ_SCORE,
            current_out["pred_masks_video_res"],
        )


def _make_exclude_self_masks(obj_idxs, mask_inputs_video_res):
    if len(obj_idxs) <= 1:
        return {}

    exclude_self_masks = {}
    for index, obj_idx in enumerate(obj_idxs):
        other_indices = torch.cat(
            [
                torch.arange(index, device=mask_inputs_video_res.device),
                torch.arange(
                    index + 1,
                    len(obj_idxs),
                    device=mask_inputs_video_res.device,
                ),
            ]
        )
        exclude_self_masks[obj_idx] = mask_inputs_video_res[other_indices].any(
            dim=0,
            keepdim=True,
        )
    return exclude_self_masks
