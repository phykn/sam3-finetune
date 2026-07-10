import torch
from torch import nn

from ..components.backbone.neck import Sam3TriViTDetNeck
from ..components.backbone.vit import ViT
from ..components.nn.position import PositionEmbeddingSine


def _make_vit(use_rope_real: bool = False) -> ViT:
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
        use_rope_real=use_rope_real,
    )


def make_vision_backbone(
    trunk: ViT | None = None,
    use_rope_real: bool = False,
) -> Sam3TriViTDetNeck:
    if trunk is None:
        trunk = _make_vit(use_rope_real=use_rope_real)
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )
    return Sam3TriViTDetNeck(
        trunk=trunk,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0],
    )


class VisionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_backbone = make_vision_backbone()

    def load_weights(self, ckpt):
        ckpt.load_block("image.vision", self)
        return self

    def forward(
        self,
        images: torch.Tensor,
        need_sam3: bool = True,
        need_interactive: bool = True,
        need_propagation: bool = True,
    ) -> dict[str, dict[str, object] | None]:
        (
            sam3_features,
            sam3_pos,
            interactive_features,
            interactive_pos,
            propagation_features,
            propagation_pos,
        ) = self.vision_backbone(
            images,
            need_sam3_out=need_sam3,
            need_interactive_out=need_interactive,
            need_propagation_out=need_propagation,
        )

        return {
            "sam3": self.branch(
                "sam3",
                sam3_features,
                sam3_pos,
                need_sam3,
            ),
            "interactive": self.branch(
                "interactive",
                interactive_features,
                interactive_pos,
                need_interactive,
            ),
            "propagation": self.branch(
                "propagation",
                propagation_features,
                propagation_pos,
                need_propagation,
            ),
        }

    @staticmethod
    def branch(name: str, features, pos_enc, required: bool):
        if not required:
            return None
        if not features:
            raise RuntimeError(f"{name} vision branch returned no feature maps")
        if len(features) != len(pos_enc):
            raise RuntimeError(
                f"{name} vision branch returned {len(features)} feature maps "
                f"but {len(pos_enc)} position encodings"
            )

        last = features[-1]
        return {
            "vision_features": getattr(last, "tensors", last),
            "vision_mask": getattr(last, "mask", None),
            "vision_pos_enc": tuple(pos_enc),
            "backbone_fpn": tuple(features),
        }
