from typing import Literal, Optional

import torch
import torch.nn as nn

from ...types import BatchedDatapoint
from .decoder.heads import build_sam_heads, forward_sam_heads
from .frame.features import (
    forward_image,
    get_interactive_pix_mem,
    prepare_backbone_features,
    prepare_backbone_features_per_frame,
)
from .frame.output import score_memory, trim_output_and_memory
from .interaction.dynamic_masks import (
    add_new_masks_to_existing_state,
    recondition_masks_in_existing_state,
)
from .interaction.mask_as_output import use_mask_as_output
from .memory.conditioning import prepare_memory_conditioned_features
from .memory.encoding import encode_new_memory
from .multiplex.state import MultiplexController, MultiplexState
from .outputs import StageOutput
from .prompt.inputs import (
    prepare_conditional_frames,
    prepare_dynamic_prompt_inputs,
    prepare_prompt_inputs,
    prepare_prompt_inputs_meta,
)
from .runtime.compile import compile_components, maybe_clone
from .runtime.init import init_tracking_model
from .runtime.loop import forward_dynamic_tracking, forward_static_tracking
from .runtime.step import run_track_step_aux


class VideoTrackingMultiplex(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        maskmem_backbone: nn.Module,
        multiplex_controller: MultiplexController,
        num_maskmem: int = 7,
        image_size: int = 512,
        backbone_stride: int = 16,
        prob_to_use_pt_input_for_train: float = 0.0,
        prob_to_use_pt_input_for_eval: float = 0.0,
        prob_to_use_box_input_for_train: float = 0.0,
        prob_to_use_box_input_for_eval: float = 0.0,
        apply_sigmoid_to_mask_logits_for_mem_enc: bool = False,
        sigmoid_scale_for_mem_enc: float = 1.0,
        sigmoid_bias_for_mem_enc: float = 0.0,
        binarize_mask_from_pts_for_mem_enc: bool = False,
        use_mask_input_as_output_without_sam: bool = False,
        num_frames_to_correct_for_train: int = 1,
        num_frames_to_correct_for_eval: int = 1,
        rand_frames_to_correct_for_train: bool = False,
        rand_frames_to_correct_for_eval: bool = False,
        prob_correct_all_objects_for_train: float = 0.0,
        ratio_of_objects_to_correct_for_train: float = 1.0,
        force_correct_all_for_conditional_inputs: bool = False,
        rand_objects_to_correct_for_train: bool = True,
        num_init_cond_frames_for_train: int = 1,
        num_init_cond_frames_for_eval: int = 1,
        rand_init_cond_frames_for_train: bool = True,
        rand_init_cond_frames_for_eval: bool = False,
        max_cond_frames_in_attn: int = -1,
        keep_first_cond_frame=False,
        add_all_frames_to_correct_as_cond: bool = False,
        num_correction_pt_per_frame: int = 7,
        pt_sampling_for_eval: Literal["uniform", "center"] = "center",
        prob_to_sample_from_gt_for_train: float = 0.0,
        directly_add_no_mem_embed: bool = False,
        use_high_res_features_in_sam: bool = False,
        multimask_output_in_sam: bool = False,
        multimask_min_pt_num: int = 1,
        multimask_max_pt_num: int = 1,
        multimask_output_for_tracking: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        use_best_iou_mask_for_mem_enc: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        iter_use_prev_mask_pred: bool = False,
        forward_backbone_per_frame_for_eval: bool = False,
        memory_temporal_stride_for_eval: int = 1,
        offload_output_to_cpu_for_eval: bool = False,
        trim_past_non_cond_mem_for_eval: bool = False,
        non_overlap_masks_for_mem_enc: bool = False,
        use_obj_ptrs_in_encoder: bool = False,
        max_obj_ptrs_in_encoder: int = 16,
        add_tpos_enc_to_obj_ptrs: bool = True,
        proj_tpos_enc_in_obj_ptrs: bool = False,
        use_signed_tpos_enc_to_obj_ptrs: bool = False,
        only_obj_ptrs_in_the_past_for_eval: bool = False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        use_no_obj_ptr: bool = True,
        use_mlp_for_obj_ptr_proj: bool = False,
        use_linear_no_obj_ptr: bool = False,
        no_obj_embed_spatial: bool = False,
        sincos_tpos_enc: bool = True,
        sam_mask_decoder_extra_args: Optional[dict] = None,
        compile_all_components: bool = False,
        save_image_features: bool = False,
        num_multimask_outputs: int = 3,
        decode_mask_with_shared_tokens: bool = False,
        decode_mask_attribute_with_shared_tokens: bool = False,
        share_necks: bool = False,
        randomness_fix: bool = False,
        add_output_suppression_embeddings: bool = False,
        add_object_conditional_embeddings: bool = False,
        add_object_unconditional_embeddings: Optional[bool] = None,
        condition_as_mask_input: bool = False,
        condition_as_mask_input_fg: float = 1.0,
        condition_as_mask_input_bg: float = 0.0,
        use_maskmem_tpos_v2: bool = False,
        use_memory_selection: bool = False,
        mf_threshold: float = 0.01,
        is_dynamic_model: bool = False,
        object_score_logit_threshold: float = 0.0,
        stability_score_attentuation: bool = False,
    ):
        super().__init__()

        config = locals().copy()
        config.pop("self")
        init_tracking_model(self, config)

    _build_sam_heads = build_sam_heads
    _get_interactive_pix_mem = get_interactive_pix_mem

    _forward_sam_heads = forward_sam_heads
    _use_mask_as_output = use_mask_as_output

    def forward(self, input: BatchedDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            backbone_out = self.forward_image(
                input.img_batch, need_interactive_out=True, need_propagation_out=True
            )
        else:
            backbone_out = {}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)

        return previous_stages_out, None

    forward_image = forward_image

    _prepare_prompt_inputs_meta = prepare_prompt_inputs_meta
    _prepare_conditional_frames = prepare_conditional_frames
    prepare_prompt_inputs = prepare_prompt_inputs

    _prepare_backbone_features = prepare_backbone_features
    _prepare_backbone_features_per_frame = prepare_backbone_features_per_frame

    _prepare_memory_conditioned_features = prepare_memory_conditioned_features

    _encode_new_memory = encode_new_memory

    forward_tracking = forward_static_tracking

    _trim_output_and_memory = trim_output_and_memory
    score_memory = score_memory

    def track_step(
        self,
        *,
        frame_idx,
        is_init_cond_frame,
        backbone_features_interactive,
        backbone_features_propagation,
        image,
        point_inputs,
        mask_inputs,
        gt_masks,
        frames_to_add_correction_pt,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
        multiplex_state: MultiplexState,
        objects_to_interact: Optional[list[int]] = None,
    ) -> StageOutput:
        current_out, _ = run_track_step_aux(
            self,
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            backbone_features_interactive=backbone_features_interactive,
            backbone_features_propagation=backbone_features_propagation,
            image=image,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            gt_masks=gt_masks,
            frames_to_add_correction_pt=frames_to_add_correction_pt,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            multiplex_state=multiplex_state,
            objects_to_interact=objects_to_interact,
            need_aux_output=False,
        )
        current_out = self._trim_output_and_memory(
            frame_idx, output_dict, current_out, memory_encoder_was_used=run_mem_encoder
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
            and self.num_multimask_outputs > 0
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    _compile_all_components = compile_components
    _maybe_clone = maybe_clone


class VideoTrackingDynamicMultiplex(VideoTrackingMultiplex):
    def __init__(
        self,
        enable_dynamic_training: bool = True,
        rand_num_transition_points: bool = True,
        max_num_transition_points: int = 3,
        add_all_transition_frames_as_cond: bool = True,
        max_trans_frames_in_attn: int = 4,
        is_dynamic_model: bool = True,
        is_dynamic_vos_evaluation: bool = False,
        **kwargs,
    ):
        super().__init__(is_dynamic_model=is_dynamic_model, **kwargs)

        self.enable_dynamic_training = enable_dynamic_training
        self.rand_num_transition_points = rand_num_transition_points
        self.max_num_transition_points = max_num_transition_points

        self.add_all_transition_frames_as_cond = add_all_transition_frames_as_cond
        self.max_trans_frames_in_attn = max_trans_frames_in_attn
        self.is_dynamic_vos_evaluation = is_dynamic_vos_evaluation

    prepare_prompt_inputs = prepare_dynamic_prompt_inputs

    add_new_masks_to_existing_state = add_new_masks_to_existing_state
    recondition_masks_in_existing_state = recondition_masks_in_existing_state

    def track_step(
        self,
        *,
        frame_idx,
        is_init_cond_frame,
        backbone_features_interactive,
        backbone_features_propagation,
        image,
        point_inputs,
        mask_inputs,
        gt_masks,
        frames_to_add_correction_pt,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
        multiplex_state: MultiplexState,
        objects_to_interact: Optional[list[int]] = None,
        new_object_masks: Optional[torch.Tensor] = None,
        new_object_idxs: Optional[list[int]] = None,
        new_object_ids: Optional[list[int]] = None,
        are_new_masks_from_pts: bool = False,
    ) -> StageOutput:
        current_out, aux_out = run_track_step_aux(
            self,
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            backbone_features_interactive=backbone_features_interactive,
            backbone_features_propagation=backbone_features_propagation,
            image=image,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            gt_masks=gt_masks,
            frames_to_add_correction_pt=frames_to_add_correction_pt,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            run_mem_encoder=(run_mem_encoder and new_object_masks is None),
            prev_sam_mask_logits=prev_sam_mask_logits,
            multiplex_state=multiplex_state,
            objects_to_interact=objects_to_interact,
            need_aux_output=(new_object_masks is not None),
        )

        if new_object_masks is not None:
            assert new_object_idxs is not None
            self.add_new_masks_to_existing_state(
                interactive_pix_feat=aux_out["interactive_pix_feat"],
                interactive_high_res_features=aux_out["interactive_high_res_features"],
                propagation_vision_feats=aux_out["propagation_vision_feats"],
                propagation_feat_sizes=aux_out["propagation_feat_sizes"],
                new_masks=new_object_masks,
                obj_idxs_in_mask=new_object_idxs,
                obj_ids_in_mask=new_object_ids,
                prev_output=current_out,
                multiplex_state=multiplex_state,
                add_mask_to_memory=run_mem_encoder,
                are_masks_from_pts=are_new_masks_from_pts,
            )

        current_out = self._trim_output_and_memory(
            frame_idx=frame_idx,
            output_dict=output_dict,
            current_out=current_out,
            memory_encoder_was_used=run_mem_encoder,
        )

        return current_out

    forward_tracking = forward_dynamic_tracking
