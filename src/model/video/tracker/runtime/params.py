import logging
from copy import deepcopy

import numpy as np
import torch
from timm.models.layers import trunc_normal_
from torch import nn


def prepare_sam_decoder_args(sam_mask_decoder_extra_args):
    interactive_args = deepcopy(sam_mask_decoder_extra_args)
    if sam_mask_decoder_extra_args is None:
        return sam_mask_decoder_extra_args, interactive_args

    dynamic_multimask = sam_mask_decoder_extra_args.get(
        "dynamic_multimask_via_stability",
        False,
    )
    if dynamic_multimask:
        sam_mask_decoder_extra_args["dynamic_multimask_via_stability"] = False
        print(
            "dynamic_multimask_via_stability is reset to False in the multiplex model"
        )
    return sam_mask_decoder_extra_args, interactive_args


def init_backbone_and_transformer(
    self,
    *,
    backbone,
    transformer,
    use_high_res_features_in_sam,
    use_obj_ptrs_in_encoder,
    max_obj_ptrs_in_encoder,
    multiplex_controller,
    save_image_features,
):
    self.backbone = backbone
    self.use_high_res_features_in_sam = use_high_res_features_in_sam
    self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
    self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
    self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
    if use_obj_ptrs_in_encoder:
        self.interactive_mask_downsample = torch.nn.Conv2d(
            1, 1, kernel_size=4, stride=4
        )

    self.multiplex_controller = multiplex_controller
    self.save_image_features = save_image_features
    self.multiplex_count = self.multiplex_controller.multiplex_count

    assert transformer.decoder is None, "transformer should be encoder-only"
    self.transformer = transformer
    self.hidden_dim = transformer.d_model


def init_temporal_pointer_config(
    self,
    *,
    add_tpos_enc_to_obj_ptrs,
    proj_tpos_enc_in_obj_ptrs,
    use_signed_tpos_enc_to_obj_ptrs,
    only_obj_ptrs_in_the_past_for_eval,
):
    self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
    if proj_tpos_enc_in_obj_ptrs:
        assert add_tpos_enc_to_obj_ptrs
    self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
    self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
    self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval


def init_memory_encoder(
    self,
    *,
    maskmem_backbone,
    num_maskmem,
    sincos_tpos_enc,
    use_maskmem_tpos_v2,
    directly_add_no_mem_embed,
):
    self.maskmem_backbone = maskmem_backbone
    self.mem_dim = self.hidden_dim
    if hasattr(self.maskmem_backbone, "out_proj") and hasattr(
        self.maskmem_backbone.out_proj, "weight"
    ):
        mem_dim = self.maskmem_backbone.out_proj.weight.shape[0]
        assert (
            mem_dim == self.hidden_dim
        ), "there should be no compression of memory embeddings"

    self.num_maskmem = num_maskmem
    self.sincos_tpos_enc = sincos_tpos_enc
    self.use_maskmem_tpos_v2 = use_maskmem_tpos_v2
    self.maskmem_tpos_enc = torch.nn.Parameter(
        torch.zeros(num_maskmem, 1, 1, self.mem_dim)
    )
    trunc_normal_(self.maskmem_tpos_enc, std=0.02)

    self.interactivity_no_mem_embed = torch.nn.Parameter(
        torch.zeros(1, 1, self.hidden_dim)
    )
    trunc_normal_(self.interactivity_no_mem_embed, std=0.02)
    self.directly_add_no_mem_embed = directly_add_no_mem_embed


def init_memory_mask_config(
    self,
    *,
    apply_sigmoid_to_mask_logits_for_mem_enc,
    sigmoid_scale_for_mem_enc,
    sigmoid_bias_for_mem_enc,
    binarize_mask_from_pts_for_mem_enc,
    non_overlap_masks_for_mem_enc,
    memory_temporal_stride_for_eval,
):
    self.apply_sigmoid_to_mask_logits_for_mem_enc = (
        apply_sigmoid_to_mask_logits_for_mem_enc
    )
    if apply_sigmoid_to_mask_logits_for_mem_enc:
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc

        if binarize_mask_from_pts_for_mem_enc:
            logging.warning("""
            The current model is not trained with binarize_mask_from_pts_for_mem_enc;
            We force it to False here because external callers often hardcoded this
            to True, ignoring the config.
            Re-training should be possible.
            """)
            binarize_mask_from_pts_for_mem_enc = False

        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc

    self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
    self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval


def init_sam_runtime_config(
    self,
    *,
    use_mask_input_as_output_without_sam,
    multimask_output_in_sam,
    multimask_min_pt_num,
    multimask_max_pt_num,
    multimask_output_for_tracking,
    use_multimask_token_for_obj_ptr,
    use_best_iou_mask_for_mem_enc,
    iou_prediction_use_sigmoid,
    object_score_logit_threshold,
    stability_score_attentuation,
    iter_use_prev_mask_pred,
    prob_to_use_pt_input_for_train,
    prob_to_use_pt_input_for_eval,
):
    self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
    self.multimask_output_in_sam = multimask_output_in_sam
    self.multimask_min_pt_num = multimask_min_pt_num
    self.multimask_max_pt_num = multimask_max_pt_num
    self.multimask_output_for_tracking = multimask_output_for_tracking
    self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
    self.use_best_iou_mask_for_mem_enc = use_best_iou_mask_for_mem_enc
    self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
    self.object_score_logit_threshold = object_score_logit_threshold
    self.stability_score_attentuation = stability_score_attentuation
    if iter_use_prev_mask_pred:
        if min(prob_to_use_pt_input_for_train, prob_to_use_pt_input_for_eval) < 1:
            assert use_mask_input_as_output_without_sam
    self.iter_use_prev_mask_pred = iter_use_prev_mask_pred


def init_image_runtime_config(
    self,
    *,
    image_size,
    backbone_stride,
    forward_backbone_per_frame_for_eval,
    offload_output_to_cpu_for_eval,
    trim_past_non_cond_mem_for_eval,
    num_frames_to_correct_for_eval,
):
    self.image_size = image_size
    self.backbone_stride = backbone_stride
    self.low_res_mask_size = self.image_size // self.backbone_stride * 4
    self.input_mask_size = self.low_res_mask_size * 4
    self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
    self.offload_output_to_cpu_for_eval = offload_output_to_cpu_for_eval
    if trim_past_non_cond_mem_for_eval:
        assert num_frames_to_correct_for_eval <= 1, (
            "trim_past_non_cond_mem_for_eval=True requires that only the first frame "
            "receives prompts"
        )
    self.trim_past_non_cond_mem_for_eval = trim_past_non_cond_mem_for_eval


def init_decoder_config(
    self,
    *,
    sam_mask_decoder_extra_args,
    interactive_sam_mask_decoder_extra_args,
    num_multimask_outputs,
    decode_mask_with_shared_tokens,
    decode_mask_attribute_with_shared_tokens,
    share_necks,
):
    self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
    self.interactive_sam_mask_decoder_extra_args = (
        interactive_sam_mask_decoder_extra_args
    )
    self.num_multimask_outputs = num_multimask_outputs
    self.decode_mask_with_shared_tokens = decode_mask_with_shared_tokens
    self.decode_mask_attribute_with_shared_tokens = (
        decode_mask_attribute_with_shared_tokens
    )
    self.share_necks = share_necks


def init_object_pointer_params(
    self,
    *,
    pred_obj_scores,
    pred_obj_scores_mlp,
    fixed_no_obj_ptr,
    use_no_obj_ptr,
    use_linear_no_obj_ptr,
    use_mlp_for_obj_ptr_proj,
    no_obj_embed_spatial,
):
    self.pred_obj_scores = pred_obj_scores
    self.pred_obj_scores_mlp = pred_obj_scores_mlp
    self.fixed_no_obj_ptr = fixed_no_obj_ptr
    self.use_no_obj_ptr = use_no_obj_ptr
    self.use_linear_no_obj_ptr = use_linear_no_obj_ptr

    if self.fixed_no_obj_ptr:
        assert self.pred_obj_scores
        assert self.use_obj_ptrs_in_encoder
    if self.pred_obj_scores and self.use_obj_ptrs_in_encoder and self.use_no_obj_ptr:
        if self.use_linear_no_obj_ptr:
            self.no_obj_ptr_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.no_obj_ptr = torch.nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            trunc_normal_(self.no_obj_ptr, std=0.02)

    self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
    self.no_obj_embed_spatial = None
    if no_obj_embed_spatial:
        self.no_obj_embed_spatial = torch.nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        trunc_normal_(self.no_obj_embed_spatial, std=0.02)


def init_condition_embedding_params(
    self,
    *,
    add_output_suppression_embeddings,
    add_object_conditional_embeddings,
    add_object_unconditional_embeddings,
    condition_as_mask_input,
    condition_as_mask_input_fg,
    condition_as_mask_input_bg,
):
    self.add_output_suppression_embeddings = add_output_suppression_embeddings
    if self.add_output_suppression_embeddings:
        self.output_valid_embed = torch.nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        self.output_invalid_embed = torch.nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        trunc_normal_(self.output_valid_embed, std=0.02)
        trunc_normal_(self.output_invalid_embed, std=0.02)

    self.add_object_conditional_embeddings = add_object_conditional_embeddings
    if add_object_unconditional_embeddings is None:
        add_object_unconditional_embeddings = add_object_conditional_embeddings
    self.add_object_unconditional_embeddings = add_object_unconditional_embeddings
    if add_object_unconditional_embeddings:
        assert add_object_conditional_embeddings
    if self.add_object_conditional_embeddings:
        self.obj_cond_embed = torch.nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        trunc_normal_(self.obj_cond_embed, std=0.02)
        if self.add_object_unconditional_embeddings:
            self.obj_non_cond_embed = torch.nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            trunc_normal_(self.obj_non_cond_embed, std=0.02)

    self.condition_as_mask_input = condition_as_mask_input
    self.condition_as_mask_input_fg = condition_as_mask_input_fg
    self.condition_as_mask_input_bg = condition_as_mask_input_bg


def init_prompt_sampling_config(
    self,
    *,
    prob_to_use_pt_input_for_train,
    prob_to_use_box_input_for_train,
    prob_to_use_pt_input_for_eval,
    prob_to_use_box_input_for_eval,
    num_frames_to_correct_for_train,
    num_frames_to_correct_for_eval,
    rand_frames_to_correct_for_train,
    rand_frames_to_correct_for_eval,
    prob_correct_all_objects_for_train,
    ratio_of_objects_to_correct_for_train,
    rand_objects_to_correct_for_train,
    force_correct_all_for_conditional_inputs,
    num_init_cond_frames_for_train,
    num_init_cond_frames_for_eval,
    rand_init_cond_frames_for_train,
    rand_init_cond_frames_for_eval,
    max_cond_frames_in_attn,
    keep_first_cond_frame,
    add_all_frames_to_correct_as_cond,
    num_correction_pt_per_frame,
    pt_sampling_for_eval,
    prob_to_sample_from_gt_for_train,
    randomness_fix,
):
    self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
    self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
    self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
    self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
    if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
        logging.info("Using points (sampled from masks) as inputs")
        assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
        assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

    self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
    self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
    self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
    self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
    self.prob_correct_all_objects_for_train = prob_correct_all_objects_for_train
    self.ratio_of_objects_to_correct_for_train = ratio_of_objects_to_correct_for_train
    self.rand_objects_to_correct_for_train = rand_objects_to_correct_for_train
    self.force_correct_all_for_conditional_inputs = (
        force_correct_all_for_conditional_inputs
    )
    self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
    self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
    self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
    self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
    self.max_cond_frames_in_attn = max_cond_frames_in_attn
    self.keep_first_cond_frame = keep_first_cond_frame
    self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
    self.num_correction_pt_per_frame = num_correction_pt_per_frame
    self.pt_sampling_for_eval = pt_sampling_for_eval
    self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
    self.rng = np.random.default_rng(seed=42)
    self.rng2 = np.random.default_rng(seed=42) if randomness_fix else self.rng


def init_tracking_runtime_config(
    self,
    *,
    use_memory_selection,
    mf_threshold,
    compile_all_components,
):
    self.use_memory_selection = use_memory_selection
    self.mf_threshold = mf_threshold
    self.compile_all_components = compile_all_components
    if self.compile_all_components:
        self._compile_all_components()
