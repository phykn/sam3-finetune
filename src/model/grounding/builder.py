from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

from ...checkpoint import load_local_checkpoint
from ...types import LoadReport
from ..backbone.neck import Sam3DualViTDetNeck
from ..backbone.vit import ViT
from ..nn.decoder import TransformerDecoder, TransformerDecoderLayer
from ..nn.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from ..nn.layers import MLP
from ..nn.modules import MultiheadAttentionWrapper as MultiheadAttention
from ..nn.position import PositionEmbeddingSine
from ..nn.scoring import DotProductScoring
from ..nn.transformer import TransformerWrapper
from .geometry import SequenceGeometryEncoder
from .model import GroundingImageModel, GroundingVisionBackbone
from .segmentation import PixelDecoder, UniversalSegmentationHead


def _create_position_encoding(precompute_resolution=None):
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone() -> ViT:
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
        use_fa3=False,
        use_rope_real=False,
    )


def _create_vision_backbone() -> GroundingVisionBackbone:
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone = _create_vit_backbone()
    neck = Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=False,
    )
    return GroundingVisionBackbone(visual=neck, scalp=1)


def _create_transformer_encoder() -> TransformerEncoderFusion:
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
            use_fa3=False,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
            use_fa3=False,
        ),
    )
    return TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )


def _create_transformer_decoder() -> TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            use_fa3=False,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )
    return TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )


def _create_transformer() -> TransformerWrapper:
    return TransformerWrapper(
        encoder=_create_transformer_encoder(),
        decoder=_create_transformer_decoder(),
        d_model=256,
    )


def _create_dot_product_scoring() -> DotProductScoring:
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head() -> UniversalSegmentationHead:
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=None,
    )
    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
        use_fa3=False,
    )
    return UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )


def _create_geometry_encoder() -> SequenceGeometryEncoder:
    geo_pos_enc = _create_position_encoding()
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )
    return SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )


def build_grounding_model(
    checkpoint_path: str | Path | None = None,
    device: torch.device | str = "cuda",
) -> GroundingImageModel | tuple[GroundingImageModel, LoadReport]:
    model = GroundingImageModel(
        backbone=_create_vision_backbone(),
        transformer=_create_transformer(),
        input_geometry_encoder=_create_geometry_encoder(),
        segmentation_head=_create_segmentation_head(),
        dot_prod_scoring=_create_dot_product_scoring(),
        num_feature_levels=1,
        o2m_mask_predict=True,
    ).to(device)
    model.eval()
    if checkpoint_path is None:
        return model
    report = load_grounding_weights(model, checkpoint_path, strict=False)
    model.to(device)
    model.eval()
    return model, report


def filter_and_remap_grounding_state_dict(
    checkpoint: Mapping,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    state = (
        checkpoint["model"]
        if "model" in checkpoint and isinstance(checkpoint["model"], Mapping)
        else checkpoint
    )
    remapped: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    skip_prefixes = (
        "detector.backbone.language_backbone.",
        "tracker.",
    )
    for key, value in state.items():
        if key.startswith(skip_prefixes):
            ignored.append(key)
            continue
        if key.startswith("detector."):
            remapped[key[len("detector.") :]] = value
        else:
            ignored.append(key)
    return remapped, ignored


def load_grounding_weights(
    model: torch.nn.Module,
    path: str | Path,
    strict: bool = False,
) -> LoadReport:
    checkpoint_path = Path(path)
    checkpoint = load_local_checkpoint(checkpoint_path)
    remapped, ignored = filter_and_remap_grounding_state_dict(checkpoint)
    result = model.load_state_dict(remapped, strict=strict)
    return LoadReport(
        checkpoint_path=checkpoint_path,
        loaded_keys=len(remapped),
        ignored_keys=len(ignored),
        missing_keys=list(result.missing_keys),
        unexpected_keys=list(result.unexpected_keys),
        ignored_key_examples=ignored[:20],
    )
