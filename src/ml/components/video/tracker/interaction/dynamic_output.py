import torch
import torch.nn.functional as F


def run_mask_output(
    self,
    *,
    interactive_pix_feat,
    interactive_high_res_features,
    new_masks,
    multiplex_state,
    objects_in_mask,
):
    return self._use_mask_as_output(
        backbone_features=interactive_pix_feat,
        high_res_features=interactive_high_res_features,
        mask_inputs=new_masks,
        multiplex_state=multiplex_state,
        objects_in_mask=objects_in_mask,
    )


def resize_low_res_masks(mask_output, prev_output):
    h, w = prev_output["pred_masks"].shape[-2:]
    mask_output["low_res_masks"] = F.interpolate(
        mask_output["low_res_masks"],
        size=(h, w),
        align_corners=False,
        mode="bilinear",
        antialias=True,
    )


def match_high_res_size(prev_output, mask_output):
    if (
        "pred_masks_high_res" not in prev_output
        or prev_output["pred_masks_high_res"] is None
    ):
        return

    interactive_resolution = mask_output["high_res_masks"].shape[-1]
    existing_resolution = prev_output["pred_masks_high_res"].shape[-1]
    if existing_resolution == interactive_resolution:
        return

    prev_output["pred_masks_high_res"] = F.interpolate(
        prev_output["pred_masks_high_res"],
        size=(interactive_resolution, interactive_resolution),
        mode="bilinear",
        align_corners=False,
    )


def append_mask_outputs(self, prev_output, mask_output):
    append_stage_output(prev_output, mask_output, "pred_masks", "low_res_masks")
    append_stage_output(
        prev_output,
        mask_output,
        "pred_masks_high_res",
        "high_res_masks",
        strict=False,
    )
    append_stage_output(
        prev_output,
        mask_output,
        "object_score_logits",
        "object_score_logits",
    )
    if self.use_memory_selection:
        mask_output["ious"] = mask_output["ious"].squeeze(-1)
        append_stage_output(prev_output, mask_output, "iou_score", "ious")


def append_stage_output(
    target,
    source,
    target_key,
    source_key,
    dim=0,
    strict=True,
):
    if target_key not in target:
        if strict:
            raise KeyError(f"{target_key} not found")
        return
    target[target_key] = torch.cat([target[target_key], source[source_key]], dim=dim)


def merge_mask_outputs(self, prev_output, mask_output, obj_idxs):
    merge_stage_output(
        prev_output, mask_output, "pred_masks", "low_res_masks", obj_idxs
    )
    merge_stage_output(
        prev_output,
        mask_output,
        "pred_masks_high_res",
        "high_res_masks",
        obj_idxs,
        strict=False,
    )
    merge_stage_output(
        prev_output,
        mask_output,
        "object_score_logits",
        "object_score_logits",
        obj_idxs,
    )
    if self.use_memory_selection:
        mask_output["ious"] = mask_output["ious"].squeeze(-1)
        merge_stage_output(prev_output, mask_output, "iou_score", "ious", obj_idxs)


def merge_stage_output(
    target,
    source,
    target_key,
    source_key,
    source_indices,
    strict=True,
):
    if target_key not in target:
        if strict:
            raise KeyError(f"{target_key} not found")
        return
    target[target_key][source_indices] = source[source_key].to(
        dtype=target[target_key].dtype
    )


def append_input_masks(prev_output, new_masks):
    if "input_masks" in prev_output:
        prev_output["input_masks"] = torch.cat(
            [prev_output["input_masks"], new_masks], dim=0
        )


def replace_input_masks(prev_output, new_masks, obj_idxs):
    if "input_masks" in prev_output:
        prev_output["input_masks"][obj_idxs] = new_masks


def append_obj_ptrs(self, prev_output, mask_output, existing_pointers, multiplex_state):
    if not self.use_obj_ptrs_in_encoder:
        return

    new_pointers = mask_output["obj_ptr"].to(existing_pointers.dtype)
    combined_pointers = torch.cat([existing_pointers, new_pointers], dim=0)
    prev_output["obj_ptr"] = multiplex_state.mux(combined_pointers)


def replace_obj_ptrs(
    self,
    prev_output,
    mask_output,
    existing_pointers,
    multiplex_state,
    obj_idxs,
):
    if not self.use_obj_ptrs_in_encoder:
        return

    new_pointers = mask_output["obj_ptr"].to(existing_pointers.dtype)
    existing_pointers[obj_idxs] = new_pointers
    prev_output["obj_ptr"] = multiplex_state.mux(existing_pointers)
