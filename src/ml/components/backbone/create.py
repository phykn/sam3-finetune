from ..nn.position import PositionEmbeddingSine
from .neck import Sam3TriViTDetNeck
from .vit import ViT


def create_vit(
    use_fa3: bool = False,
    use_rope_real: bool = False,
) -> ViT:
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


def create_vision_backbone(
    trunk: ViT | None = None,
    use_fa3: bool = False,
    use_rope_real: bool = False,
) -> Sam3TriViTDetNeck:
    if trunk is None:
        trunk = create_vit(use_fa3=use_fa3, use_rope_real=use_rope_real)
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )
    return Sam3TriViTDetNeck(
        trunk=trunk,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0],
    )
