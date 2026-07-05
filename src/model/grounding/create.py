import torch.nn as nn

from ..components.backbone.create import create_vit
from ..components.backbone.neck import Sam3DualViTDetNeck
from ..components.backbone.vit import ViT
from ..components.nn.layers import MLP
from ..components.nn.modules import MultiheadAttentionWrapper as MultiheadAttention
from ..components.nn.position import PositionEmbeddingSine
from ..components.transformer.decoder import TransformerDecoder, TransformerDecoderLayer
from ..components.transformer.encoder import (
    TransformerEncoderFusion,
    TransformerEncoderLayer,
)
from ..components.transformer.wrapper import TransformerWrapper
from .backbone import GroundingVisionBackbone
from .encoder import SequenceGeometryEncoder
from .model import GroundingImageModel
from .pixel import PixelDecoder
from .scoring import DotProductScorer
from .segmentation import UniversalSegmentationHead

D_MODEL = 256
FFN_DIM = 2048
NUM_HEADS = 8
DROPOUT = 0.1
IMAGE_RESOLUTION = 1008
VIT_STRIDE = 14


def create_attention(*, batch_first: bool, dropout: float = DROPOUT):
    return MultiheadAttention(
        num_heads=NUM_HEADS,
        dropout=dropout,
        embed_dim=D_MODEL,
        batch_first=batch_first,
        use_fa3=False,
    )


def create_position_encoding(precompute_resolution=None):
    return PositionEmbeddingSine(
        num_pos_feats=D_MODEL,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def create_backbone(trunk: ViT | None = None) -> GroundingVisionBackbone:
    position_encoding = create_position_encoding(precompute_resolution=IMAGE_RESOLUTION)
    if trunk is None:
        trunk = create_vit()
    neck = Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=D_MODEL,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=trunk,
        add_sam2_neck=False,
    )
    return GroundingVisionBackbone(visual=neck, scalp=1)


def create_transformer_encoder() -> TransformerEncoderFusion:
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=D_MODEL,
        dim_feedforward=FFN_DIM,
        dropout=DROPOUT,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=create_attention(batch_first=True),
        cross_attention=create_attention(batch_first=True),
    )
    return TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=D_MODEL,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )


def create_transformer_decoder() -> TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=D_MODEL,
        dim_feedforward=FFN_DIM,
        dropout=DROPOUT,
        cross_attention=create_attention(batch_first=False),
        n_heads=NUM_HEADS,
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
        d_model=D_MODEL,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=IMAGE_RESOLUTION,
        stride=VIT_STRIDE,
        use_act_checkpoint=True,
        presence_token=True,
    )


def create_transformer() -> TransformerWrapper:
    return TransformerWrapper(
        encoder=create_transformer_encoder(),
        decoder=create_transformer_decoder(),
        d_model=D_MODEL,
    )


def create_dot_product_scorer() -> DotProductScorer:
    prompt_mlp = MLP(
        input_dim=D_MODEL,
        hidden_dim=FFN_DIM,
        output_dim=D_MODEL,
        num_layers=2,
        dropout=DROPOUT,
        residual=True,
        out_norm=nn.LayerNorm(D_MODEL),
    )
    return DotProductScorer(d_model=D_MODEL, d_proj=D_MODEL, prompt_mlp=prompt_mlp)


def create_segmentation_head() -> UniversalSegmentationHead:
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=D_MODEL,
        compile_mode=None,
    )
    cross_attend_prompt = create_attention(batch_first=False, dropout=0)
    return UniversalSegmentationHead(
        hidden_dim=D_MODEL,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )


def create_geometry_encoder() -> SequenceGeometryEncoder:
    geo_pos_enc = create_position_encoding()
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=D_MODEL,
        dim_feedforward=FFN_DIM,
        dropout=DROPOUT,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=create_attention(batch_first=False),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=create_attention(batch_first=False),
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
        d_model=D_MODEL,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )


def create_grounding_model(
    trunk: ViT | None = None,
    vision_backbone: nn.Module | None = None,
) -> GroundingImageModel:
    backbone = (
        GroundingVisionBackbone(visual=vision_backbone, scalp=0)
        if vision_backbone
        else create_backbone(trunk)
    )
    return GroundingImageModel(
        backbone=backbone,
        transformer=create_transformer(),
        input_geometry_encoder=create_geometry_encoder(),
        segmentation_head=create_segmentation_head(),
        dot_prod_scoring=create_dot_product_scorer(),
        num_feature_levels=1,
        o2m_mask_predict=True,
    )
