from torch import nn

from ..components.backbone.create import create_vision_backbone
from ..components.backbone.neck import Sam3TriViTDetNeck
from ..components.backbone.vit import ViT
from ..components.nn.position import PositionEmbeddingSine
from ..components.transformer.decoder import (
    DecoupledTransformerDecoderLayerv2,
    SimpleRoPEAttention,
    TransformerEncoderDecoupledCrossAttention,
)
from ..components.transformer.wrapper import TransformerWrapper
from .tracker.memory.encoder import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from .tracker.model import Sam3VideoTrackingMultiplexDemo
from .tracker.multiplex.state import MultiplexController

D_MODEL = 256
IMAGE_SIZE = 1008
BACKBONE_STRIDE = 14
MULTIMASK_OUTPUTS = 3


class TriHeadVisionOnly(nn.Module):
    def __init__(self, visual: Sam3TriViTDetNeck, n_features: int, scalp: int = 0):
        super().__init__()
        self.vision_backbone = visual
        self.n_features = n_features
        self.scalp = scalp

    def forward_image(
        self,
        samples,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ):
        (
            sam3_features,
            sam3_pos,
            interactive_features,
            interactive_pos,
            propagation_features,
            propagation_pos,
        ) = self.vision_backbone(
            samples,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )

        if self.scalp > 0:
            sam3_features = sam3_features[: -self.scalp]
            sam3_pos = sam3_pos[: -self.scalp]
            interactive_features = interactive_features[: -self.scalp]
            interactive_pos = interactive_pos[: -self.scalp]
            propagation_features = propagation_features[: -self.scalp]
            propagation_pos = propagation_pos[: -self.scalp]

        output = {}
        if need_sam3_out:
            sam3_last = sam3_features[-1]
            output.update(
                {
                    "vision_features": sam3_last.tensors,
                    "vision_mask": sam3_last.mask,
                    "vision_pos_enc": sam3_pos,
                    "backbone_fpn": sam3_features,
                }
            )
        if need_interactive_out:
            interactive_last = interactive_features[-1]
            output["interactive"] = {
                "vision_features": interactive_last.tensors,
                "vision_mask": interactive_last.mask,
                "vision_pos_enc": interactive_pos,
                "backbone_fpn": interactive_features,
            }
        if need_propagation_out:
            propagation_last = propagation_features[-1]
            output["sam2_backbone_out"] = {
                "vision_features": propagation_last.tensors,
                "vision_mask": propagation_last.mask,
                "vision_pos_enc": propagation_pos,
                "backbone_fpn": propagation_features,
            }
        return output


def create_maskmem_backbone(multiplex_count: int = 16):
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=D_MODEL,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=IMAGE_SIZE,
    )

    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3,
        stride=2,
        padding=1,
        interpol_size=[1152, 1152],
        multiplex_count=multiplex_count,
        starting_out_chan=4,
        input_channel_multiplier=2,
    )

    cx_block_layer = CXBlock(
        dim=D_MODEL,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    return SimpleMaskEncoder(
        out_dim=D_MODEL,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


def create_transformer(
    use_fa3: bool = False,
    use_rope_real: bool = False,
):
    self_attention_rope = SimpleRoPEAttention(
        d_model=D_MODEL,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    cross_attention_rope = SimpleRoPEAttention(
        d_model=D_MODEL,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    encoder_layer = DecoupledTransformerDecoderLayerv2(
        activation="gelu",
        d_model=D_MODEL,
        num_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pos_enc_at_attn=False,
        pre_norm=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention_rope=self_attention_rope,
        cross_attention_rope=cross_attention_rope,
    )
    encoder = TransformerEncoderDecoupledCrossAttention(
        d_model=D_MODEL,
        frozen=False,
        pos_enc_at_input=True,
        use_image_in_output=False,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
        batch_first=True,
    )
    return TransformerWrapper(encoder=encoder, decoder=None, d_model=D_MODEL)


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

    backbone = TriHeadVisionOnly(visual=vision_backbone, n_features=256, scalp=0)

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
