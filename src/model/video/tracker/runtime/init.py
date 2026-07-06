from ....blocks.track_mgr import (
    init_condition_embedding_params,
    init_object_pointer_params,
)
from ....blocks.video_mem import init_memory_encoder
from ....blocks.video_track import init_backbone_and_transformer
from .params import (
    init_decoder_config,
    init_image_runtime_config,
    init_memory_mask_config,
    init_prompt_sampling_config,
    init_sam_runtime_config,
    init_temporal_pointer_config,
    init_tracking_runtime_config,
    prepare_sam_decoder_args,
)


def init_tracking_model(self, config):
    sam_args, interactive_args = prepare_sam_decoder_args(
        config["sam_mask_decoder_extra_args"]
    )
    init_tracking_backbone(self, config)
    init_tracking_memory(self, config)
    init_tracking_sam(self, config, sam_args, interactive_args)
    init_tracking_prompting(self, config)


def init_tracking_backbone(self, config):
    init_backbone_and_transformer(
        self,
        backbone=config["backbone"],
        transformer=config["transformer"],
        use_high_res_features_in_sam=config["use_high_res_features_in_sam"],
        use_obj_ptrs_in_encoder=config["use_obj_ptrs_in_encoder"],
        max_obj_ptrs_in_encoder=config["max_obj_ptrs_in_encoder"],
        multiplex_controller=config["multiplex_controller"],
        save_image_features=config["save_image_features"],
    )
    init_temporal_pointer_config(
        self,
        add_tpos_enc_to_obj_ptrs=config["add_tpos_enc_to_obj_ptrs"],
        proj_tpos_enc_in_obj_ptrs=config["proj_tpos_enc_in_obj_ptrs"],
        use_signed_tpos_enc_to_obj_ptrs=config["use_signed_tpos_enc_to_obj_ptrs"],
        only_obj_ptrs_in_the_past_for_eval=(
            config["only_obj_ptrs_in_the_past_for_eval"]
        ),
    )


def init_tracking_memory(self, config):
    init_memory_encoder(
        self,
        maskmem_backbone=config["maskmem_backbone"],
        num_maskmem=config["num_maskmem"],
        sincos_tpos_enc=config["sincos_tpos_enc"],
        use_maskmem_tpos_v2=config["use_maskmem_tpos_v2"],
        directly_add_no_mem_embed=config["directly_add_no_mem_embed"],
    )
    init_memory_mask_config(
        self,
        apply_sigmoid_to_mask_logits_for_mem_enc=(
            config["apply_sigmoid_to_mask_logits_for_mem_enc"]
        ),
        sigmoid_scale_for_mem_enc=config["sigmoid_scale_for_mem_enc"],
        sigmoid_bias_for_mem_enc=config["sigmoid_bias_for_mem_enc"],
        binarize_mask_from_pts_for_mem_enc=config["binarize_mask_from_pts_for_mem_enc"],
        non_overlap_masks_for_mem_enc=config["non_overlap_masks_for_mem_enc"],
        memory_temporal_stride_for_eval=config["memory_temporal_stride_for_eval"],
    )


def init_tracking_sam(self, config, sam_args, interactive_args):
    init_sam_runtime_config(
        self,
        use_mask_input_as_output_without_sam=(
            config["use_mask_input_as_output_without_sam"]
        ),
        multimask_output_in_sam=config["multimask_output_in_sam"],
        multimask_min_pt_num=config["multimask_min_pt_num"],
        multimask_max_pt_num=config["multimask_max_pt_num"],
        multimask_output_for_tracking=config["multimask_output_for_tracking"],
        use_multimask_token_for_obj_ptr=config["use_multimask_token_for_obj_ptr"],
        use_best_iou_mask_for_mem_enc=config["use_best_iou_mask_for_mem_enc"],
        iou_prediction_use_sigmoid=config["iou_prediction_use_sigmoid"],
        object_score_logit_threshold=config["object_score_logit_threshold"],
        stability_score_attentuation=config["stability_score_attentuation"],
        iter_use_prev_mask_pred=config["iter_use_prev_mask_pred"],
        prob_to_use_pt_input_for_train=config["prob_to_use_pt_input_for_train"],
        prob_to_use_pt_input_for_eval=config["prob_to_use_pt_input_for_eval"],
    )
    init_image_runtime_config(
        self,
        image_size=config["image_size"],
        backbone_stride=config["backbone_stride"],
        forward_backbone_per_frame_for_eval=(
            config["forward_backbone_per_frame_for_eval"]
        ),
        offload_output_to_cpu_for_eval=config["offload_output_to_cpu_for_eval"],
        trim_past_non_cond_mem_for_eval=config["trim_past_non_cond_mem_for_eval"],
        num_frames_to_correct_for_eval=config["num_frames_to_correct_for_eval"],
    )
    init_tracking_decoders(self, config, sam_args, interactive_args)


def init_tracking_decoders(self, config, sam_args, interactive_args):
    init_decoder_config(
        self,
        sam_mask_decoder_extra_args=sam_args,
        interactive_sam_mask_decoder_extra_args=interactive_args,
        num_multimask_outputs=config["num_multimask_outputs"],
        decode_mask_with_shared_tokens=config["decode_mask_with_shared_tokens"],
        decode_mask_attribute_with_shared_tokens=(
            config["decode_mask_attribute_with_shared_tokens"]
        ),
        share_necks=config["share_necks"],
    )
    init_object_pointer_params(
        self,
        pred_obj_scores=config["pred_obj_scores"],
        pred_obj_scores_mlp=config["pred_obj_scores_mlp"],
        fixed_no_obj_ptr=config["fixed_no_obj_ptr"],
        use_no_obj_ptr=config["use_no_obj_ptr"],
        use_linear_no_obj_ptr=config["use_linear_no_obj_ptr"],
        use_mlp_for_obj_ptr_proj=config["use_mlp_for_obj_ptr_proj"],
        no_obj_embed_spatial=config["no_obj_embed_spatial"],
    )
    init_condition_embedding_params(
        self,
        add_output_suppression_embeddings=config["add_output_suppression_embeddings"],
        add_object_conditional_embeddings=config["add_object_conditional_embeddings"],
        add_object_unconditional_embeddings=(
            config["add_object_unconditional_embeddings"]
        ),
        condition_as_mask_input=config["condition_as_mask_input"],
        condition_as_mask_input_fg=config["condition_as_mask_input_fg"],
        condition_as_mask_input_bg=config["condition_as_mask_input_bg"],
    )
    self.is_dynamic_model = config["is_dynamic_model"]
    self._build_sam_heads()


def init_tracking_prompting(self, config):
    init_prompt_sampling_config(
        self,
        prob_to_use_pt_input_for_train=config["prob_to_use_pt_input_for_train"],
        prob_to_use_box_input_for_train=config["prob_to_use_box_input_for_train"],
        prob_to_use_pt_input_for_eval=config["prob_to_use_pt_input_for_eval"],
        prob_to_use_box_input_for_eval=config["prob_to_use_box_input_for_eval"],
        num_frames_to_correct_for_train=config["num_frames_to_correct_for_train"],
        num_frames_to_correct_for_eval=config["num_frames_to_correct_for_eval"],
        rand_frames_to_correct_for_train=config["rand_frames_to_correct_for_train"],
        rand_frames_to_correct_for_eval=config["rand_frames_to_correct_for_eval"],
        prob_correct_all_objects_for_train=config["prob_correct_all_objects_for_train"],
        ratio_of_objects_to_correct_for_train=(
            config["ratio_of_objects_to_correct_for_train"]
        ),
        rand_objects_to_correct_for_train=config["rand_objects_to_correct_for_train"],
        force_correct_all_for_conditional_inputs=(
            config["force_correct_all_for_conditional_inputs"]
        ),
        num_init_cond_frames_for_train=config["num_init_cond_frames_for_train"],
        num_init_cond_frames_for_eval=config["num_init_cond_frames_for_eval"],
        rand_init_cond_frames_for_train=config["rand_init_cond_frames_for_train"],
        rand_init_cond_frames_for_eval=config["rand_init_cond_frames_for_eval"],
        max_cond_frames_in_attn=config["max_cond_frames_in_attn"],
        keep_first_cond_frame=config["keep_first_cond_frame"],
        add_all_frames_to_correct_as_cond=config["add_all_frames_to_correct_as_cond"],
        num_correction_pt_per_frame=config["num_correction_pt_per_frame"],
        pt_sampling_for_eval=config["pt_sampling_for_eval"],
        prob_to_sample_from_gt_for_train=config["prob_to_sample_from_gt_for_train"],
        randomness_fix=config["randomness_fix"],
    )
    init_tracking_runtime_config(
        self,
        use_memory_selection=config["use_memory_selection"],
        mf_threshold=config["mf_threshold"],
        compile_all_components=config["compile_all_components"],
    )
