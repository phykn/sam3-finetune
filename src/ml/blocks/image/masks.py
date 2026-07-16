import torch
from torch import nn

from ...components.sam.mask_decoder import MaskDecoder
from ...components.sam.transformer import TwoWayTransformer


class ImageMaskDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
            dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
        )

    def load_weights(self, ckpt):
        ckpt.load_block("image.masks", self)
        return self

    def forward(
        self,
        image_embed: torch.Tensor,
        high_res,
        prompt,
        image_pe: torch.Tensor,
        multimask=True,
        repeat_image=False,
        mix: torch.Tensor | None = None,
    ):
        sparse, dense = prompt
        return self.mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
            repeat_image=repeat_image,
            high_res_features=list(high_res),
            mix=mix,
        )
