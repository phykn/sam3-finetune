import torch

from ..outputs import NO_OBJ_SCORE
from .frame import resize_mask


def merge_per_object_outputs(
    model,
    inference_state,
    out,
    frame_idx,
    storage_key,
    target,
    object_count,
    reconstruct_from_objects,
):
    obj_scores = []
    iou_scores = [] if reconstruct_from_objects and model.use_memory_selection else None

    for obj_idx in range(object_count):
        obj_out = _find_object_output(inference_state, obj_idx, frame_idx, storage_key)
        if obj_out is None:
            continue

        _write_object_mask(out, target["mask_key"], obj_idx, obj_out)

        if reconstruct_from_objects:
            if "object_score_logits" in obj_out:
                obj_scores.append(obj_out["object_score_logits"])
            if model.use_memory_selection and "iou_score" in obj_out:
                iou_scores.append(obj_out["iou_score"])

    return obj_scores, iou_scores


def _find_object_output(inference_state, obj_idx, frame_idx, storage_key):
    if obj_idx not in inference_state["temp_output_dict_per_obj"]:
        return None
    if obj_idx not in inference_state["output_dict_per_obj"]:
        return None

    temp_out = inference_state["temp_output_dict_per_obj"][obj_idx]
    obj_out = inference_state["output_dict_per_obj"][obj_idx]
    out = temp_out[storage_key].get(frame_idx, None)
    if out is not None:
        return out

    out = obj_out["cond_frame_outputs"].get(frame_idx, None)
    if out is not None:
        return out

    return obj_out["non_cond_frame_outputs"].get(frame_idx, None)


def _write_object_mask(out, mask_key, obj_idx, obj_out):
    obj_mask = obj_out.get("pred_masks_video_res")
    if obj_mask is None:
        obj_mask = obj_out.get("pred_masks")

    pred_masks = _pad_object_slots(out, mask_key, obj_idx)
    resized = resize_mask(
        obj_mask,
        size=pred_masks.shape[-2:],
        is_downsampling="pred_masks_video_res" in obj_out,
        dtype=pred_masks.dtype,
    )
    pred_masks[obj_idx : obj_idx + 1] = resized


def _pad_object_slots(out, mask_key, obj_idx):
    pred_masks = out[mask_key]
    if obj_idx < pred_masks.shape[0]:
        return pred_masks

    pad_size = obj_idx + 1 - pred_masks.shape[0]
    padding = torch.zeros(
        (pad_size, 1, pred_masks.shape[-2], pred_masks.shape[-1]),
        dtype=pred_masks.dtype,
        device=pred_masks.device,
    )
    pred_masks = torch.cat([pred_masks, padding], dim=0)
    out[mask_key] = pred_masks

    if "object_score_logits" in out:
        out["object_score_logits"] = _pad_object_scores(
            out["object_score_logits"], pad_size
        )

    return pred_masks


def _pad_object_scores(scores, pad_size):
    padding = torch.full(
        (pad_size, 1),
        NO_OBJ_SCORE,
        dtype=scores.dtype,
        device=scores.device,
    )
    return torch.cat([scores, padding], dim=0)
