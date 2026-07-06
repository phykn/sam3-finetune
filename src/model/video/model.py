from ..blocks.video_feat import VideoFeat
from ..components.backbone.create import create_vision_backbone
from ..components.backbone.neck import Sam3TriViTDetNeck
from ..components.backbone.vit import ViT
from ..components.video.create import (
    BACKBONE_STRIDE,
    create_maskmem_backbone,
    create_transformer,
    IMAGE_SIZE,
    MULTIMASK_OUTPUTS,
)
from .tracker.model import Sam3VideoTrackingMultiplexDemo
from .tracker.multiplex.state import MultiplexController


def make_tracking_kwargs() -> dict:
    return {
        "image_size": IMAGE_SIZE,
        "backbone_stride": BACKBONE_STRIDE,
        "num_maskmem": 7,
        "use_high_res_features_in_sam": True,
        "use_obj_ptrs_in_encoder": True,
        "max_obj_ptrs_in_encoder": 16,
        "add_tpos_enc_to_obj_ptrs": True,
        "proj_tpos_enc_in_obj_ptrs": True,
        "use_mlp_for_obj_ptr_proj": True,
        "pred_obj_scores": True,
        "pred_obj_scores_mlp": True,
        "fixed_no_obj_ptr": True,
        "use_no_obj_ptr": True,
        "use_linear_no_obj_ptr": True,
        "no_obj_embed_spatial": True,
        "sincos_tpos_enc": True,
        "multimask_output_in_sam": True,
        "multimask_output_for_tracking": True,
        "multimask_min_pt_num": 0,
        "multimask_max_pt_num": 1,
        "use_multimask_token_for_obj_ptr": True,
        "num_multimask_outputs": MULTIMASK_OUTPUTS,
        "apply_sigmoid_to_mask_logits_for_mem_enc": True,
        "sigmoid_scale_for_mem_enc": 2.0,
        "sigmoid_bias_for_mem_enc": -1.0,
        "non_overlap_masks_for_mem_enc": False,
        "add_output_suppression_embeddings": True,
        "add_object_conditional_embeddings": False,
        "condition_as_mask_input": True,
        "condition_as_mask_input_fg": 1.0,
        "condition_as_mask_input_bg": 0.0,
        "use_maskmem_tpos_v2": True,
        "save_image_features": True,
        "randomness_fix": True,
        "use_mask_input_as_output_without_sam": True,
        "directly_add_no_mem_embed": True,
        "iou_prediction_use_sigmoid": False,
        "forward_backbone_per_frame_for_eval": True,
        "offload_output_to_cpu_for_eval": False,
        "trim_past_non_cond_mem_for_eval": False,
        "max_cond_frames_in_attn": 4,
        "is_dynamic_model": True,
        "sam_mask_decoder_extra_args": {
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        "compile_all_components": False,
        "use_memory_selection": False,
    }


def create_video_memory_model(
    trunk: ViT | None = None,
    vision_backbone: Sam3TriViTDetNeck | None = None,
    multiplex_count: int = 16,
    max_num_objects: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = False,
) -> Sam3VideoTrackingMultiplexDemo:
    maskmem_backbone = create_maskmem_backbone(multiplex_count=multiplex_count)
    transformer = create_transformer(
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    if vision_backbone is None:
        vision_backbone = create_vision_backbone(
            trunk=trunk,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
        )

    backbone = VideoFeat(vision_backbone=vision_backbone, scalp=0)

    multiplex_controller = MultiplexController(
        multiplex_count=multiplex_count,
        eval_multiplex_count=multiplex_count,
    )

    model = Sam3VideoTrackingMultiplexDemo(
        backbone=backbone,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multiplex_controller=multiplex_controller,
        **make_tracking_kwargs(),
    )

    model.max_num_objects = max_num_objects
    return model
