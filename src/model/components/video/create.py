from ..nn.position import PositionEmbeddingSine
from ..sam.transformer import TwoWayTransformer
from ..transformer.decoder import (
    DecoupledTransformerDecoderLayerv2,
    SimpleRoPEAttention,
    TransformerEncoderDecoupledCrossAttention,
)
from ..transformer.wrapper import TransformerWrapper
from .memory import CXBlock, SimpleFuser, SimpleMaskDownSampler, SimpleMaskEncoder

D_MODEL = 256
IMAGE_SIZE = 1008
BACKBONE_STRIDE = 14
MULTIMASK_OUTPUTS = 3


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
    fuser = SimpleFuser(
        layer=CXBlock(
            dim=D_MODEL,
            kernel_size=7,
            padding=3,
            layer_scale_init_value=1.0e-06,
            use_dwconv=True,
        ),
        num_layers=2,
    )
    return SimpleMaskEncoder(
        out_dim=D_MODEL,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


def create_transformer(use_fa3: bool = False, use_rope_real: bool = False):
    self_attn = SimpleRoPEAttention(
        d_model=D_MODEL,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    cross_attn = SimpleRoPEAttention(
        d_model=D_MODEL,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    layer = DecoupledTransformerDecoderLayerv2(
        activation="gelu",
        d_model=D_MODEL,
        num_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pos_enc_at_attn=False,
        pre_norm=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention_rope=self_attn,
        cross_attention_rope=cross_attn,
    )
    encoder = TransformerEncoderDecoupledCrossAttention(
        d_model=D_MODEL,
        frozen=False,
        pos_enc_at_input=True,
        use_image_in_output=False,
        layer=layer,
        num_layers=4,
        use_act_checkpoint=False,
        batch_first=True,
    )
    return TransformerWrapper(encoder=encoder, decoder=None, d_model=D_MODEL)


def make_two_way_transformer(embed_dim):
    return TwoWayTransformer(
        depth=2,
        embedding_dim=embed_dim,
        mlp_dim=2048,
        num_heads=8,
    )
