from collections.abc import Iterable

import torch

from ...components.video.tracker.consolidation.merge import (
    consolidate_temp_output_across_obj,
)
from ...components.video.tracker.multiplex.state import MultiplexState
from .state import output_store


@torch.inference_mode()
def remove_object(
    self,
    inference_state,
    obj_id: int,
    strict=False,
    need_output=True,
    clear_user_refined_map: bool = True,
):
    return self.remove_objects(
        inference_state,
        obj_ids=[obj_id],
        strict=strict,
        need_output=need_output,
        clear_user_refined_map=clear_user_refined_map,
    )


@torch.inference_mode()
def remove_objects(
    self,
    inference_state,
    obj_ids: Iterable[int],
    strict=False,
    need_output=True,
    clear_user_refined_map: bool = True,
):
    obj_ids = list(obj_ids)
    if len(obj_ids) != len(set(obj_ids)):
        raise ValueError("obj_ids in one request must be unique")
    updated_frames = []

    removed_idxs, removed_ids = resolve_removed_objects(
        inference_state, obj_ids, strict
    )
    if not removed_idxs:
        return inference_state["obj_ids"], updated_frames

    if clear_user_refined_map:
        clear_user_refined_objects(inference_state, removed_ids)

    input_frame_indices = clear_removed_object_inputs(
        inference_state,
        removed_idxs,
    )

    old_obj_inds, old_idx_to_new_idx = reindex_objects(
        inference_state,
        removed_idxs,
    )
    if not inference_state["obj_ids"]:
        reset_objects(inference_state)
        return inference_state["obj_ids"], updated_frames

    remap_object_containers(inference_state, old_obj_inds, old_idx_to_new_idx)

    multiplex_state: MultiplexState = inference_state["multiplex_state"]
    buckets_to_keep = multiplex_state.remove_objects(removed_idxs, strict=True)
    slice_output_states(
        self,
        inference_state,
        buckets_to_keep,
        set(removed_ids),
    )

    if need_output:
        updated_frames = rebuild_removed_object_frames(
            self,
            inference_state,
            input_frame_indices,
        )

    return inference_state["obj_ids"], updated_frames


def resolve_removed_objects(inference_state, obj_ids, strict):
    removed_idxs = []
    removed_ids = []
    for obj_id in obj_ids:
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id)
        if obj_idx is None:
            if strict:
                raise ValueError(
                    f"Object id {obj_id} does not exist in the tracking state."
                )
            continue

        removed_idxs.append(obj_idx)
        removed_ids.append(obj_id)
    return removed_idxs, removed_ids


def clear_user_refined_objects(inference_state, removed_ids):
    if "user_refined_frames_per_obj" not in inference_state:
        return

    user_refined_map = inference_state["user_refined_frames_per_obj"]
    for removed_id in removed_ids:
        user_refined_map.pop(removed_id, None)


def clear_removed_object_inputs(inference_state, removed_idxs):
    frame_indices = set()
    for obj_idx in removed_idxs:
        mask_inputs = inference_state["mask_inputs_per_obj"][obj_idx]
        frame_indices.update(mask_inputs)
        mask_inputs.clear()
    return frame_indices


def reset_objects(inference_state):
    for key in (
        "mask_inputs_per_obj",
        "output_dict_per_obj",
        "temp_output_dict_per_obj",
    ):
        inference_state[key].clear()
    inference_state["output_dict"] = output_store()
    inference_state["consolidated_frame_inds"] = output_store(set)
    inference_state["frames_already_tracked"].clear()
    inference_state["multiplex_state"] = None
    inference_state["tracking_has_started"] = False
    inference_state["first_ann_frame_idx"] = None


def reindex_objects(inference_state, removed_idxs):
    old_obj_ids = inference_state["obj_ids"]
    old_obj_inds = list(range(len(old_obj_ids)))
    remain_old_obj_inds = [idx for idx in old_obj_inds if idx not in removed_idxs]
    new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
    new_obj_inds = list(range(len(new_obj_ids)))
    old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))

    inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
    inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
    inference_state["obj_ids"] = new_obj_ids

    return old_obj_inds, old_idx_to_new_idx


def remap_object_containers(inference_state, old_obj_inds, old_idx_to_new_idx):
    for key in (
        "mask_inputs_per_obj",
        "output_dict_per_obj",
        "temp_output_dict_per_obj",
    ):
        remap_container_keys(inference_state[key], old_obj_inds, old_idx_to_new_idx)


def remap_container_keys(container, old_obj_inds, old_idx_to_new_idx):
    new_items = []
    for key in old_obj_inds:
        value = container.pop(key)
        if key in old_idx_to_new_idx:
            new_items.append((old_idx_to_new_idx[key], value))
    container.update(new_items)


def slice_output_states(self, inference_state, buckets_to_keep, removed_ids):
    output_dict = inference_state["output_dict"]
    for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
        for frame_idx, out in output_dict[storage_key].items():
            slice_frame_output(
                self,
                inference_state,
                frame_idx,
                out,
                storage_key,
                buckets_to_keep,
                removed_ids,
            )


def slice_frame_output(
    self,
    inference_state,
    frame_idx,
    out,
    storage_key,
    buckets_to_keep,
    removed_ids,
):
    out["maskmem_features"] = out["maskmem_features"][buckets_to_keep]
    out["maskmem_pos_enc"] = [x[buckets_to_keep] for x in out["maskmem_pos_enc"]]
    out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(inference_state, out)
    out["obj_ptr"] = out["obj_ptr"][buckets_to_keep]

    local_obj_id_to_idx = out["local_obj_id_to_idx"]
    keep_indices = get_local_keep_indices(out, local_obj_id_to_idx, removed_ids)
    out["pred_masks"] = out["pred_masks"][keep_indices]
    out["object_score_logits"] = out["object_score_logits"][keep_indices]
    if self.use_memory_selection:
        out["iou_score"] = out["iou_score"][keep_indices]
        out["eff_iou_score"] = self.score_memory(
            out["object_score_logits"], out["iou_score"]
        )

    rewrite_local_mapping(out, local_obj_id_to_idx, keep_indices, removed_ids)
    self._add_output_per_object(inference_state, frame_idx, out, storage_key)


def get_local_keep_indices(out, local_obj_id_to_idx, removed_ids):
    local_remain_old_obj_inds = [
        obj_idx
        for obj_id, obj_idx in local_obj_id_to_idx.items()
        if obj_id not in removed_ids
    ]
    max_pred = out["pred_masks"].shape[0]
    max_scores = out["object_score_logits"].shape[0]
    return [
        idx
        for idx in local_remain_old_obj_inds
        if 0 <= idx < max_pred and 0 <= idx < max_scores
    ]


def rewrite_local_mapping(out, local_obj_id_to_idx, keep_indices, removed_ids):
    old_to_new = {old_idx: new_i for new_i, old_idx in enumerate(keep_indices)}
    new_local_obj_id_to_idx = {}
    conditioning_objects = set()

    for obj_id, old_idx in local_obj_id_to_idx.items():
        if obj_id in removed_ids or old_idx not in old_to_new:
            continue

        new_idx = old_to_new[old_idx]
        new_local_obj_id_to_idx[obj_id] = new_idx
        if old_idx in out["conditioning_objects"]:
            conditioning_objects.add(new_idx)

    out["local_obj_id_to_idx"] = new_local_obj_id_to_idx
    out["conditioning_objects"] = conditioning_objects


def rebuild_removed_object_frames(self, inference_state, frame_indices):
    updated_frames = []
    temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]

    for frame_idx in frame_indices:
        is_cond = any(
            frame_idx in output_dict["cond_frame_outputs"]
            for output_dict in temp_output_dict_per_obj.values()
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
        updated_frames.append((frame_idx, video_res_masks))

    return updated_frames


def clear_non_cond_mem_around_input(self, inference_state, frame_idx):
    r = self.memory_temporal_stride_for_eval
    frame_idx_begin = frame_idx - r * self.num_maskmem
    frame_idx_end = frame_idx + r * self.num_maskmem
    output_dict = inference_state["output_dict"]
    non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]
    for t in range(frame_idx_begin, frame_idx_end + 1):
        non_cond_frame_outputs.pop(t, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            obj_output_dict["non_cond_frame_outputs"].pop(t, None)
