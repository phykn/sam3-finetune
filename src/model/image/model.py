import torch
import torch.nn as nn

from ..components.backbone.create import create_vision_backbone
from ..components.backbone.encoder import ImageEncoder
from ..components.sam.mask_decoder import MaskDecoder
from ..components.sam.prompt_encoder import PromptEncoder
from ..components.sam.transformer import TwoWayTransformer


class Sam3ImageModel(nn.Module):
    def __init__(
        self,
        vision_backbone: nn.Module | None = None,
        prompt_encoder: nn.Module | None = None,
        mask_decoder: nn.Module | None = None,
        interactivity_no_mem_embed: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        self.image_size = 1008
        self.backbone_stride = 14
        self.hidden_dim = 256
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        if interactivity_no_mem_embed is None:
            interactivity_no_mem_embed = nn.Parameter(
                torch.zeros(1, 1, self.hidden_dim)
            )
        self.interactivity_no_mem_embed = interactivity_no_mem_embed

        if vision_backbone is None:
            vision_backbone = create_vision_backbone()
        self.image_encoder = ImageEncoder(vision_backbone)
        self.prompt_encoder = prompt_encoder or PromptEncoder(
            embed_dim=256,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = mask_decoder or MaskDecoder(
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
