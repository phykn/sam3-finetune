import torch
import torch.nn as nn

from ...checkpoint import load_weights
from ...types import LoadReport
from ..backbone.image_encoder import InteractiveImageEncoder
from ..backbone.neck import Sam3TriViTDetNeck
from ..backbone.vit import ViT
from ..nn.position import PositionEmbeddingSine
from ..sam.mask_decoder import MaskDecoder
from ..sam.prompt_encoder import PromptEncoder
from ..sam.transformer import TwoWayTransformer


class Sam3PromptedSegmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_size = 1008
        self.backbone_stride = 14
        self.hidden_dim = 256
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        self.interactivity_no_mem_embed = nn.Parameter(
            torch.zeros(1, 1, self.hidden_dim)
        )

        position_encoding = PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=1008,
        )
        trunk = ViT(
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
        tri_neck = Sam3TriViTDetNeck(
            trunk=trunk,
            position_encoding=position_encoding,
            d_model=256,
            scale_factors=[4.0, 2.0, 1.0],
        )
        self.image_encoder = InteractiveImageEncoder(tri_neck)
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=False,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
        )

    def encode_image(self, images: torch.Tensor) -> dict[str, object]:
        return self.image_encoder(
            images,
            self.mask_decoder,
            interactivity_no_mem_embed=self.interactivity_no_mem_embed,
        )


def build_model(
    checkpoint_path: str | None = None,
    device: torch.device | str = "cuda",
) -> tuple[Sam3PromptedSegmenter, LoadReport | None] | Sam3PromptedSegmenter:
    model = Sam3PromptedSegmenter().to(device)
    model.eval()
    if checkpoint_path is None:
        return model
    report = load_weights(model, checkpoint_path, strict=False)
    model.to(device)
    model.eval()
    return model, report
