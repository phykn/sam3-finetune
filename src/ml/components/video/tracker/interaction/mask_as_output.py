import torch
import torch.nn.functional as F

from ..outputs import SAMOutput


def use_mask_as_output(
    self,
    backbone_features: torch.Tensor,
    high_res_features: list[torch.Tensor],
    mask_inputs: torch.Tensor,
    multiplex_state,
    objects_in_mask: list[int] | None = None,
) -> SAMOutput:
    if objects_in_mask is None:
        objects_in_mask = list(range(multiplex_state.total_valid_entries))

    out_scale = 20.0
    out_bias = -10.0
    mask_inputs_float = mask_inputs.to(backbone_features.dtype)
    assert mask_inputs.shape[0] == len(objects_in_mask), (
        f"{mask_inputs.shape[0]} != {len(objects_in_mask)}"
    )

    low_res_masks, high_res_masks, ious, object_score_logits, appearing = (
        make_mask_outputs(
            mask_inputs,
            mask_inputs_float,
            backbone_features.dtype,
            out_scale,
            out_bias,
        )
    )

    obj_ptr = get_mask_input_obj_ptr(
        self,
        backbone_features,
        high_res_features,
        mask_inputs,
        mask_inputs_float,
        multiplex_state,
        objects_in_mask,
        appearing,
    )
    return make_mask_as_output_dict(
        self,
        low_res_masks,
        high_res_masks,
        ious,
        object_score_logits,
        obj_ptr,
    )


def make_mask_outputs(mask_inputs, mask_inputs_float, dtype, out_scale, out_bias):
    high_res_masks = mask_inputs_float * out_scale + out_bias
    low_res_masks = F.interpolate(
        high_res_masks,
        size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
        align_corners=False,
        mode="bilinear",
        antialias=True,
    )
    ious = mask_inputs.new_ones(mask_inputs.size(0), 1, dtype=dtype)
    appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)[..., None]
    appearing = appearing.float()
    object_score_logits = out_scale * appearing + out_bias
    return low_res_masks, high_res_masks, ious, object_score_logits, appearing


def get_mask_input_obj_ptr(
    self,
    backbone_features,
    high_res_features,
    mask_inputs,
    mask_inputs_float,
    multiplex_state,
    objects_in_mask,
    appearing,
):
    if not self.use_obj_ptrs_in_encoder:
        return None

    sam_outputs = self._forward_sam_heads(
        backbone_features=backbone_features,
        mask_inputs=self.interactive_mask_downsample(mask_inputs_float),
        interactive_high_res_features=high_res_features,
        gt_masks=mask_inputs,
        objects_to_interact=objects_in_mask,
        multiplex_state=multiplex_state,
    )
    obj_ptr = sam_outputs["obj_ptr"]
    if self.pred_obj_scores and self.use_no_obj_ptr:
        obj_ptr = _blend_no_object_ptr(
            self,
            obj_ptr,
            appearing,
            multiplex_state,
            objects_in_mask,
        )
    return obj_ptr


def make_mask_as_output_dict(
    self,
    low_res_masks,
    high_res_masks,
    ious,
    object_score_logits,
    obj_ptr,
) -> SAMOutput:
    output: SAMOutput = {
        "low_res_multimasks": low_res_masks,
        "high_res_multimasks": high_res_masks,
        "ious": ious,
        "low_res_masks": low_res_masks,
        "high_res_masks": high_res_masks,
        "object_score_logits": object_score_logits,
    }
    if self.use_obj_ptrs_in_encoder:
        output["obj_ptr"] = obj_ptr
    return output


def _blend_no_object_ptr(
    self,
    obj_ptr,
    lambda_is_obj_appearing,
    multiplex_state,
    objects_in_mask,
):
    if self.use_linear_no_obj_ptr:
        return lambda_is_obj_appearing * obj_ptr + (
            1 - lambda_is_obj_appearing
        ) * self.no_obj_ptr_linear(obj_ptr)

    if self.fixed_no_obj_ptr:
        obj_ptr = lambda_is_obj_appearing * obj_ptr

    selected_no_obj_ptr = self.no_obj_ptr.unsqueeze(0).repeat(
        multiplex_state.num_buckets,
        1,
        1,
    )
    selected_no_obj_ptr = multiplex_state.demux(selected_no_obj_ptr)
    selected_no_obj_ptr = selected_no_obj_ptr[objects_in_mask]
    return obj_ptr + (1 - lambda_is_obj_appearing) * selected_no_obj_ptr
