import torch
import torch.nn.functional as F

from ...sam import get_propagation_dense_pe
from ..multiplex.state import MultiplexState
from ..outputs import NO_OBJ_SCORE, SAMOutput


def forward_sam_heads(
    self,
    backbone_features: torch.Tensor,
    *,
    point_inputs: dict[str, torch.Tensor] | None = None,
    mask_inputs: torch.Tensor | None = None,
    interactive_high_res_features: list[torch.Tensor] | None = None,
    propagation_high_res_features: list[torch.Tensor] | None = None,
    multimask_output: bool = False,
    gt_masks=None,
    multiplex_state: MultiplexState,
    objects_to_interact: list[int] | None = None,
) -> SAMOutput:
    device = backbone_features.device
    check_backbone_features(self, backbone_features)
    is_interactive = point_inputs is not None or mask_inputs is not None

    if is_interactive:
        assert objects_to_interact is not None
        sam_out = run_interactive_sam(
            self,
            backbone_features=backbone_features,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=interactive_high_res_features,
            multimask_output=multimask_output,
            device=device,
        )
    else:
        sam_out = run_propagation_sam(
            self,
            backbone_features=backbone_features,
            high_res_features=propagation_high_res_features,
            multimask_output=multimask_output,
            multiplex_state=multiplex_state,
        )

    low_res_multimasks, ious, sam_output_tokens, object_score_logits = (
        clone_sam_outputs(self, sam_out)
    )
    low_res_multimasks, is_obj_appearing = apply_object_scores(
        self, low_res_multimasks, object_score_logits
    )
    low_res_multimasks = low_res_multimasks.float()
    high_res_multimasks = upscale_masks(self, low_res_multimasks)

    low_res_masks, high_res_masks, sam_output_token, ious = select_mask_outputs(
        self,
        low_res_multimasks,
        high_res_multimasks,
        ious,
        sam_output_tokens,
        multimask_output,
        is_interactive,
        device,
    )
    obj_ptr = project_object_pointer(
        self,
        sam_output_token,
        is_interactive,
        is_obj_appearing,
        multiplex_state,
        objects_to_interact,
    )

    return make_sam_output(
        self,
        low_res_multimasks=low_res_multimasks,
        high_res_multimasks=high_res_multimasks,
        ious=ious,
        low_res_masks=low_res_masks,
        high_res_masks=high_res_masks,
        object_score_logits=object_score_logits,
        obj_ptr=obj_ptr,
    )


def upscale_masks(self, masks):
    return F.interpolate(
        masks,
        size=(self.image_size, self.image_size),
        mode="bilinear",
        align_corners=False,
    )


def make_sam_output(
    self,
    *,
    low_res_multimasks,
    high_res_multimasks,
    ious,
    low_res_masks,
    high_res_masks,
    object_score_logits,
    obj_ptr,
) -> SAMOutput:
    output: SAMOutput = {
        "low_res_multimasks": low_res_multimasks,
        "high_res_multimasks": high_res_multimasks,
        "ious": ious,
        "low_res_masks": low_res_masks,
        "high_res_masks": high_res_masks,
        "object_score_logits": object_score_logits,
    }
    if self.use_obj_ptrs_in_encoder:
        output["obj_ptr"] = obj_ptr
    return output


def check_backbone_features(self, backbone_features):
    assert backbone_features.size(1) == self.hidden_dim
    assert backbone_features.size(2) == self.sam_image_embedding_size
    assert backbone_features.size(3) == self.sam_image_embedding_size


def run_interactive_sam(
    self,
    *,
    backbone_features,
    point_inputs,
    mask_inputs,
    high_res_features,
    multimask_output,
    device,
):
    assert high_res_features is not None
    sam_point_coords, sam_point_labels = prepare_interactive_points(
        point_inputs,
        mask_inputs,
        device,
    )
    sam_mask_prompt = prepare_interactive_mask(self, mask_inputs)
    sparse_embeddings, dense_embeddings = self.interactive_sam_prompt_encoder(
        points=(sam_point_coords, sam_point_labels),
        boxes=None,
        masks=sam_mask_prompt,
    )
    sparse_embeddings = self._maybe_clone(sparse_embeddings)
    dense_embeddings = self._maybe_clone(dense_embeddings)
    image_pe = self._maybe_clone(self.interactive_sam_prompt_encoder.get_dense_pe())
    return self.interactive_sam_mask_decoder(
        image_embeddings=backbone_features,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        repeat_image=True,
        high_res_features=high_res_features,
    )


def prepare_interactive_points(point_inputs, mask_inputs, device):
    if point_inputs is not None:
        return point_inputs["point_coords"], point_inputs["point_labels"]

    assert mask_inputs is not None
    coords = torch.zeros(mask_inputs.shape[0], 1, 2, device=device)
    labels = -torch.ones(mask_inputs.shape[0], 1, dtype=torch.int32, device=device)
    return coords, labels


def prepare_interactive_mask(self, mask_inputs):
    if mask_inputs is None:
        return None

    assert len(mask_inputs.shape) == 4
    if mask_inputs.shape[-2:] == self.interactive_sam_prompt_encoder.mask_input_size:
        return mask_inputs

    return F.interpolate(
        mask_inputs.float(),
        size=self.interactive_sam_prompt_encoder.mask_input_size,
        align_corners=False,
        mode="bilinear",
        antialias=True,
    )


def run_propagation_sam(
    self,
    *,
    backbone_features,
    high_res_features,
    multimask_output,
    multiplex_state,
):
    assert high_res_features is not None
    assert multiplex_state is not None

    image_pe = self._maybe_clone(get_propagation_dense_pe(self))
    out = self.sam_mask_decoder(
        image_embeddings=backbone_features,
        image_pe=image_pe,
        high_res_features=high_res_features,
        multimask_output=multimask_output,
        extra_per_object_embeddings=output_suppression_embeddings(
            self, multiplex_state
        ),
    )
    return (
        multiplex_state.demux(out["masks"]),
        multiplex_state.demux(out["iou_pred"]),
        multiplex_state.demux(out["sam_tokens_out"]),
        multiplex_state.demux(out["object_score_logits"]),
    )


def output_suppression_embeddings(self, multiplex_state):
    if not self.add_output_suppression_embeddings:
        return None

    output_valid_embed = self.output_valid_embed.unsqueeze(0)
    output_invalid_embed = self.output_invalid_embed.unsqueeze(0)
    valid_object_mask = multiplex_state.get_valid_object_mask().unsqueeze(-1).float()
    return (
        valid_object_mask * output_valid_embed
        + (1 - valid_object_mask) * output_invalid_embed
    )


def clone_sam_outputs(self, sam_out):
    low_res_multimasks, ious, sam_output_tokens, object_score_logits = sam_out
    return (
        self._maybe_clone(low_res_multimasks),
        self._maybe_clone(ious),
        self._maybe_clone(sam_output_tokens),
        self._maybe_clone(object_score_logits),
    )


def apply_object_scores(self, low_res_multimasks, object_score_logits):
    if not self.pred_obj_scores:
        return low_res_multimasks, None

    is_obj_appearing = object_score_logits > self.object_score_logit_threshold
    low_res_multimasks = torch.where(
        is_obj_appearing[:, None, None],
        low_res_multimasks,
        NO_OBJ_SCORE,
    )
    return low_res_multimasks, is_obj_appearing


def select_mask_outputs(
    self,
    low_res_multimasks,
    high_res_multimasks,
    ious,
    sam_output_tokens,
    multimask_output,
    is_interactive,
    device,
):
    sam_output_token = sam_output_tokens[:, 0]
    if not multimask_output:
        return low_res_multimasks, high_res_multimasks, sam_output_token, ious

    if not self.decode_mask_with_shared_tokens or is_interactive:
        if self.stability_score_attentuation:
            stability_score = self.sam_mask_decoder.get_stability_scores(
                low_res_multimasks
            )
            ious = ious * stability_score

        best_iou_inds = torch.argmax(ious, dim=-1)
        batch_inds = torch.arange(ious.shape[0], device=device)
        low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        if sam_output_tokens.size(1) > 1:
            sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        return low_res_masks, high_res_masks, sam_output_token, ious

    assert self.decode_mask_with_shared_tokens
    return (
        low_res_multimasks[:, 0:1],
        high_res_multimasks[:, 0:1],
        sam_output_token,
        ious,
    )


def project_object_pointer(
    self,
    sam_output_token,
    is_interactive,
    is_obj_appearing,
    multiplex_state,
    objects_to_interact,
):
    if not self.use_obj_ptrs_in_encoder:
        return None

    if is_interactive:
        assert objects_to_interact is not None
        obj_ptr = self.interactive_obj_ptr_proj(sam_output_token)
    else:
        obj_ptr = self.obj_ptr_proj(sam_output_token)

    if self.pred_obj_scores and self.use_no_obj_ptr:
        obj_ptr = blend_no_object_pointer(
            self,
            obj_ptr,
            is_obj_appearing.float(),
            multiplex_state,
            is_interactive,
            objects_to_interact,
        )
    return obj_ptr


def blend_no_object_pointer(
    self,
    obj_ptr,
    is_obj_appearing,
    multiplex_state,
    is_interactive,
    objects_to_interact,
):
    if self.use_linear_no_obj_ptr:
        return is_obj_appearing * obj_ptr + (1 - is_obj_appearing) * (
            self.no_obj_ptr_linear(obj_ptr)
        )

    if self.fixed_no_obj_ptr:
        obj_ptr = is_obj_appearing * obj_ptr

    selected_no_obj_ptr = self.no_obj_ptr.unsqueeze(0).repeat(
        multiplex_state.num_buckets, 1, 1
    )
    selected_no_obj_ptr = multiplex_state.demux(selected_no_obj_ptr)
    if is_interactive:
        selected_no_obj_ptr = selected_no_obj_ptr[objects_to_interact]

    return obj_ptr + (1 - is_obj_appearing) * selected_no_obj_ptr
