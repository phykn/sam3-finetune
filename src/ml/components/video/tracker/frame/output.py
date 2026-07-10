import torch

from ..outputs import StageOutput


def compact_frame_output(self, inference_state, current_out):
    storage_device = inference_state["storage_device"]
    maskmem_features = _move_maskmem_features(current_out, storage_device)
    image_features, image_pos_enc = _move_image_features(current_out, storage_device)

    pred_masks_gpu = current_out["pred_masks"]
    pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
    with torch.profiler.record_function("VideoTrackingMultiplexDemo.maskmem_pos_enc"):
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)

    obj_ptr = current_out["obj_ptr"]
    object_score_logits = current_out["object_score_logits"]
    compact = {
        "maskmem_features": maskmem_features,
        "maskmem_pos_enc": maskmem_pos_enc,
        "image_features": image_features,
        "image_pos_enc": image_pos_enc,
        "pred_masks": pred_masks,
        "obj_ptr": obj_ptr,
        "object_score_logits": object_score_logits,
        "conditioning_objects": current_out["conditioning_objects"],
    }
    if self.use_memory_selection:
        with torch.profiler.record_function(
            "VideoTrackingMultiplexDemo.use_memory_selection"
        ):
            compact["iou_score"] = current_out["iou_score"]
            compact["eff_iou_score"] = self.score_memory(
                object_score_logits, current_out["iou_score"]
            )
    return compact, pred_masks_gpu


def trim_output_and_memory(
    self,
    frame_idx: int,
    output_dict: dict[str, dict[int, StageOutput]],
    current_out: StageOutput,
    memory_encoder_was_used: bool,
) -> StageOutput:
    if self.offload_output_to_cpu_for_eval and not self.training:
        current_out = _offload_output(self, current_out, memory_encoder_was_used)

    if self.trim_past_non_cond_mem_for_eval and not self.training:
        _trim_old_non_cond_outputs(self, frame_idx, output_dict)

    return current_out


def score_memory(self, object_score_logits, iou_score):
    object_score_norm = torch.where(
        object_score_logits > 0,
        object_score_logits.sigmoid() * 2 - 1,
        torch.zeros_like(object_score_logits),
    )
    return (object_score_norm * iou_score).mean()


def _offload_output(self, current_out, memory_encoder_was_used):
    trimmed_out: StageOutput = {
        "conditioning_objects": current_out["conditioning_objects"],
        "pred_masks": current_out["pred_masks"].cpu(),
        "pred_masks_high_res": current_out["pred_masks_high_res"].cpu(),
        "object_score_logits": current_out["object_score_logits"],
    }
    if self.use_obj_ptrs_in_encoder:
        trimmed_out["obj_ptr"] = current_out["obj_ptr"]
    if memory_encoder_was_used and self.num_maskmem > 0:
        trimmed_out["maskmem_features"] = current_out["maskmem_features"].cpu()
        trimmed_out["maskmem_pos_enc"] = [
            x.cpu() for x in current_out["maskmem_pos_enc"]
        ]
    if self.save_image_features:
        trimmed_out["image_features"] = current_out["image_features"].cpu()
        trimmed_out["image_pos_enc"] = current_out["image_pos_enc"].cpu()
    return trimmed_out


def _trim_old_non_cond_outputs(self, frame_idx, output_dict):
    stride = self.memory_temporal_stride_for_eval
    past_frame_idx = frame_idx - stride * self.num_maskmem
    _trim_non_cond_output(self, output_dict, past_frame_idx)

    if self.use_memory_selection and not self.offload_output_to_cpu_for_eval:
        far_old_frame_idx = frame_idx - 20 * self.max_obj_ptrs_in_encoder
        _trim_non_cond_output(self, output_dict, far_old_frame_idx, force=True)


def _trim_non_cond_output(self, output_dict, frame_idx, force=False):
    past_out = output_dict["non_cond_frame_outputs"].get(frame_idx)
    if past_out is None:
        return

    should_trim = force
    if not should_trim and not self.use_memory_selection:
        should_trim = True
    if (
        not should_trim
        and self.use_memory_selection
        and past_out.get("eff_iou_score", 0) < self.mf_threshold
    ):
        should_trim = True

    if should_trim:
        output_dict["non_cond_frame_outputs"][frame_idx] = _trim_past_output(
            self, past_out
        )


def _trim_past_output(self, past_out: StageOutput) -> StageOutput:
    trimmed_out: StageOutput = {
        "conditioning_objects": past_out["conditioning_objects"],
        "pred_masks": past_out["pred_masks"],
        "object_score_logits": past_out["object_score_logits"],
    }
    if self.use_obj_ptrs_in_encoder:
        trimmed_out["obj_ptr"] = past_out["obj_ptr"]
    return trimmed_out


def _move_maskmem_features(current_out, storage_device):
    if current_out.get("maskmem_features") is None:
        return None

    maskmem_features = current_out["maskmem_features"]
    return maskmem_features.to(
        device=storage_device,
        dtype=torch.bfloat16,
        non_blocking=True,
    )


def _move_image_features(current_out, storage_device):
    if current_out.get("image_features") is None:
        return None, None

    assert "image_pos_enc" in current_out
    image_features = current_out["image_features"].to(storage_device, non_blocking=True)
    image_pos_enc = current_out["image_pos_enc"].to(storage_device, non_blocking=True)
    return image_features, image_pos_enc
