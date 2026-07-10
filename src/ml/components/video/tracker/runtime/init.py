from copy import deepcopy

from ...init_parts import (
    init_backbone_and_transformer,
    init_condition_embedding_params,
    init_memory_encoder,
    init_object_pointer_params,
)


def init_tracking_model(model, config):
    init_backbone_and_transformer(
        model,
        backbone=config["backbone"],
        transformer=config["transformer"],
        use_high_res_features_in_sam=config["use_high_res_features_in_sam"],
        use_obj_ptrs_in_encoder=config["use_obj_ptrs_in_encoder"],
        max_obj_ptrs_in_encoder=config["max_obj_ptrs_in_encoder"],
        multiplex_controller=config["multiplex_controller"],
        save_image_features=config["save_image_features"],
    )
    init_memory_encoder(
        model,
        maskmem_backbone=config["maskmem_backbone"],
        num_maskmem=config["num_maskmem"],
        sincos_tpos_enc=config["sincos_tpos_enc"],
        use_maskmem_tpos_v2=config["use_maskmem_tpos_v2"],
        directly_add_no_mem_embed=config["directly_add_no_mem_embed"],
    )
    init_object_pointer_params(
        model,
        pred_obj_scores=config["pred_obj_scores"],
        pred_obj_scores_mlp=config["pred_obj_scores_mlp"],
        fixed_no_obj_ptr=config["fixed_no_obj_ptr"],
        use_no_obj_ptr=config["use_no_obj_ptr"],
        use_linear_no_obj_ptr=config["use_linear_no_obj_ptr"],
        use_mlp_for_obj_ptr_proj=config["use_mlp_for_obj_ptr_proj"],
        no_obj_embed_spatial=config["no_obj_embed_spatial"],
    )
    init_condition_embedding_params(
        model,
        add_output_suppression_embeddings=config["add_output_suppression_embeddings"],
        add_object_conditional_embeddings=config["add_object_conditional_embeddings"],
        add_object_unconditional_embeddings=config[
            "add_object_unconditional_embeddings"
        ],
        condition_as_mask_input=config["condition_as_mask_input"],
        condition_as_mask_input_fg=config["condition_as_mask_input_fg"],
        condition_as_mask_input_bg=config["condition_as_mask_input_bg"],
    )
    init_inference_values(model, config)
    model._build_sam_heads()


def init_inference_values(model, config):
    model.image_size = config["image_size"]
    model.backbone_stride = config["backbone_stride"]
    model.low_res_mask_size = model.image_size // model.backbone_stride * 4
    model.input_mask_size = model.low_res_mask_size * 4

    model.add_tpos_enc_to_obj_ptrs = config["add_tpos_enc_to_obj_ptrs"]
    model.proj_tpos_enc_in_obj_ptrs = config["proj_tpos_enc_in_obj_ptrs"]
    model.use_signed_tpos_enc_to_obj_ptrs = config["use_signed_tpos_enc_to_obj_ptrs"]
    model.only_obj_ptrs_in_the_past_for_eval = config[
        "only_obj_ptrs_in_the_past_for_eval"
    ]

    model.apply_sigmoid_to_mask_logits_for_mem_enc = config[
        "apply_sigmoid_to_mask_logits_for_mem_enc"
    ]
    model.sigmoid_scale_for_mem_enc = config["sigmoid_scale_for_mem_enc"]
    model.sigmoid_bias_for_mem_enc = config["sigmoid_bias_for_mem_enc"]
    model.binarize_mask_from_pts_for_mem_enc = False
    model.non_overlap_masks_for_mem_enc = config["non_overlap_masks_for_mem_enc"]
    model.memory_temporal_stride_for_eval = config["memory_temporal_stride_for_eval"]

    model.use_mask_input_as_output_without_sam = config[
        "use_mask_input_as_output_without_sam"
    ]
    model.multimask_output_in_sam = config["multimask_output_in_sam"]
    model.multimask_min_pt_num = config["multimask_min_pt_num"]
    model.multimask_max_pt_num = config["multimask_max_pt_num"]
    model.multimask_output_for_tracking = config["multimask_output_for_tracking"]
    model.use_multimask_token_for_obj_ptr = config["use_multimask_token_for_obj_ptr"]
    model.use_best_iou_mask_for_mem_enc = config["use_best_iou_mask_for_mem_enc"]
    model.iou_prediction_use_sigmoid = config["iou_prediction_use_sigmoid"]
    model.object_score_logit_threshold = config["object_score_logit_threshold"]
    model.stability_score_attentuation = config["stability_score_attentuation"]
    model.iter_use_prev_mask_pred = False

    interactive_args = deepcopy(config["sam_mask_decoder_extra_args"])
    propagation_args = deepcopy(interactive_args)
    propagation_args["dynamic_multimask_via_stability"] = False
    model.sam_mask_decoder_extra_args = propagation_args
    model.interactive_sam_mask_decoder_extra_args = interactive_args
    model.num_multimask_outputs = config["num_multimask_outputs"]
    model.decode_mask_with_shared_tokens = config["decode_mask_with_shared_tokens"]
    model.decode_mask_attribute_with_shared_tokens = config[
        "decode_mask_attribute_with_shared_tokens"
    ]
    model.share_necks = config["share_necks"]

    model.offload_output_to_cpu_for_eval = False
    model.trim_past_non_cond_mem_for_eval = False
    model.max_cond_frames_in_attn = config["max_cond_frames_in_attn"]
    model.keep_first_cond_frame = config["keep_first_cond_frame"]
    model.add_all_frames_to_correct_as_cond = config[
        "add_all_frames_to_correct_as_cond"
    ]
    model.use_memory_selection = config["use_memory_selection"]
    model.mf_threshold = config["mf_threshold"]
    model.is_dynamic_model = True
