import torch
import torch.nn.functional as F

from ..outputs import NO_OBJ_SCORE


def get_target(model, inference_state, run_mem_encoder, consolidate_at_video_res):
    if consolidate_at_video_res:
        assert not run_mem_encoder, "memory encoder cannot run at video resolution"
        return {
            "height": inference_state["video_height"],
            "width": inference_state["video_width"],
            "mask_key": "pred_masks_video_res",
        }

    return {
        "height": model.low_res_mask_size,
        "width": model.low_res_mask_size,
        "mask_key": "pred_masks",
    }


def new_output(inference_state, batch_size, target):
    return {
        "conditioning_objects": None,
        "maskmem_features": None,
        "maskmem_pos_enc": None,
        "image_features": None,
        "image_pos_enc": None,
        "obj_ptr": None,
        target["mask_key"]: torch.full(
            size=(batch_size, 1, target["height"], target["width"]),
            fill_value=NO_OBJ_SCORE,
            dtype=torch.float32,
            device=inference_state["storage_device"],
        ),
    }


def find_frame_output(inference_state, frame_idx):
    out = inference_state["output_dict"]["cond_frame_outputs"].get(frame_idx, None)
    if out is not None:
        return out
    return inference_state["output_dict"]["non_cond_frame_outputs"].get(frame_idx, None)


def find_conditioning_objects(inference_state, frame_idx, batch_size):
    conditioning_objects = set()
    for obj_idx in range(batch_size):
        if _has_frame_input(inference_state["mask_inputs_per_obj"], obj_idx, frame_idx):
            conditioning_objects.add(obj_idx)

    return conditioning_objects


def _has_frame_input(inputs_per_obj, obj_idx, frame_idx):
    if obj_idx not in inputs_per_obj:
        return False
    obj_inputs = inputs_per_obj[obj_idx]
    return frame_idx in obj_inputs and obj_inputs[frame_idx] is not None


def copy_frame_output(model, out, frame_out, target):
    out["conditioning_objects"] = frame_out.get("conditioning_objects", set())
    out["obj_ptr"] = frame_out["obj_ptr"]
    out["object_score_logits"] = frame_out["object_score_logits"]
    if model.use_memory_selection:
        out["iou_score"] = frame_out["iou_score"]

    # Singleton extraction can omit memory/image fields.
    out["maskmem_features"] = frame_out.get("maskmem_features")
    out["maskmem_pos_enc"] = frame_out.get("maskmem_pos_enc")
    out["image_features"] = frame_out.get("image_features")
    out["image_pos_enc"] = frame_out.get("image_pos_enc")
    out["local_obj_id_to_idx"] = frame_out.get("local_obj_id_to_idx", {})

    mask = frame_out.get("pred_masks_video_res", frame_out["pred_masks"])
    out[target["mask_key"]] = resize_mask(
        mask,
        size=(target["height"], target["width"]),
        is_downsampling=mask.shape[-1] > target["width"],
    )


def resize_mask(mask, size, is_downsampling, dtype=None):
    if mask.shape[-2:] == size:
        resized = mask
    else:
        resized = F.interpolate(
            mask,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=is_downsampling,
        )

    if dtype is not None and resized.dtype != dtype:
        return resized.to(dtype)
    return resized
