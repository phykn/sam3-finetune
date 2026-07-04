from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .checkpoint import LoadReport
from .decoder_memory import (
    DecoupledTransformerDecoderLayerv2,
    SimpleRoPEAttention,
    TransformerEncoderDecoupledCrossAttention,
)
from .memory import CXBlock, SimpleFuser, SimpleMaskDownSampler, SimpleMaskEncoder
from .model_misc import TransformerWrapper
from .multiplex_utils import MultiplexController
from .neck import Sam3TriViTDetNeck
from .position_encoding import PositionEmbeddingSine
from .video_checkpoint import load_video_weights
from .video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo
from .vit import ViT


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


def _create_position_encoding(precompute_resolution=None):
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(use_fa3: bool = False, use_rope_real: bool = False):
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=None,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )


def _create_multiplex_tri_backbone(
    use_fa3: bool = False,
    use_rope_real: bool = False,
):
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone = _create_vit_backbone(
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    return Sam3TriViTDetNeck(
        trunk=vit_backbone,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0],
    )


def _create_multiplex_maskmem_backbone(multiplex_count: int = 16):
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
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
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)
    return SimpleMaskEncoder(
        out_dim=256,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


def _create_multiplex_transformer(
    use_fa3: bool = False,
    use_rope_real: bool = False,
):
    self_attention_rope = SimpleRoPEAttention(
        d_model=256,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    cross_attention_rope = SimpleRoPEAttention(
        d_model=256,
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
        d_model=256,
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
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        use_image_in_output=False,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
        batch_first=True,
    )
    return TransformerWrapper(encoder=encoder, decoder=None, d_model=256)


def build_video_memory_model(
    checkpoint_path: str | Path | None = None,
    device: torch.device | str = "cuda",
    multiplex_count: int = 16,
    max_num_objects: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = False,
) -> tuple[Sam3VideoTrackingMultiplexDemo, LoadReport | None] | Sam3VideoTrackingMultiplexDemo:
    maskmem_backbone = _create_multiplex_maskmem_backbone(
        multiplex_count=multiplex_count
    )
    transformer = _create_multiplex_transformer(
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    tri_neck = _create_multiplex_tri_backbone(
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    backbone = TriHeadVisionOnly(visual=tri_neck, n_features=256, scalp=0)
    multiplex_controller = MultiplexController(
        multiplex_count=multiplex_count,
        eval_multiplex_count=multiplex_count,
    )

    model = Sam3VideoTrackingMultiplexDemo(
        backbone=backbone,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multiplex_controller=multiplex_controller,
        image_size=1008,
        backbone_stride=14,
        num_maskmem=7,
        use_high_res_features_in_sam=True,
        use_obj_ptrs_in_encoder=True,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_mlp_for_obj_ptr_proj=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        use_no_obj_ptr=True,
        use_linear_no_obj_ptr=True,
        no_obj_embed_spatial=True,
        sincos_tpos_enc=True,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_multimask_token_for_obj_ptr=True,
        num_multimask_outputs=3,
        apply_sigmoid_to_mask_logits_for_mem_enc=True,
        sigmoid_scale_for_mem_enc=2.0,
        sigmoid_bias_for_mem_enc=-1.0,
        non_overlap_masks_for_mem_enc=False,
        add_output_suppression_embeddings=True,
        add_object_conditional_embeddings=False,
        condition_as_mask_input=True,
        condition_as_mask_input_fg=1.0,
        condition_as_mask_input_bg=0.0,
        use_maskmem_tpos_v2=True,
        save_image_features=True,
        randomness_fix=True,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        iou_prediction_use_sigmoid=False,
        forward_backbone_per_frame_for_eval=True,
        offload_output_to_cpu_for_eval=False,
        trim_past_non_cond_mem_for_eval=False,
        max_cond_frames_in_attn=4,
        is_dynamic_model=True,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        compile_all_components=False,
        use_memory_selection=False,
    )
    model.max_num_objects = max_num_objects
    model.to(device=device)
    model.eval()

    if checkpoint_path is None:
        return model
    report = load_video_weights(model, checkpoint_path, strict=False)
    model.to(device=device)
    model.eval()
    return model, report
