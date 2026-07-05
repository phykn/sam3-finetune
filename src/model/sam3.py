from torch import nn

from .components.backbone.create import create_vision_backbone
from .grounding.create import create_grounding_model
from src.model.image.model import Sam3ImageModel
from src.model.video.model import create_video_memory_model


class Sam3Model(nn.Module):
    def __init__(
        self,
        multiplex_count: int = 16,
        max_num_objects: int = 16,
        use_fa3: bool = False,
        use_rope_real: bool = False,
    ) -> None:
        super().__init__()
        vision_backbone = create_vision_backbone(
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
        )

        self.video = create_video_memory_model(
            vision_backbone=vision_backbone,
            multiplex_count=multiplex_count,
            max_num_objects=max_num_objects,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
        )
        self.image = Sam3ImageModel(
            vision_backbone=vision_backbone,
            prompt_encoder=self.video.interactive_sam_prompt_encoder,
            mask_decoder=self.video.interactive_sam_mask_decoder,
            interactivity_no_mem_embed=self.video.interactivity_no_mem_embed,
        )
        self.grounding = create_grounding_model(vision_backbone=vision_backbone)
        self.share()

    def share(self):
        self.image.image_encoder.vision_backbone = self.video.backbone.vision_backbone
        self.grounding.backbone.vision_backbone = self.video.backbone.vision_backbone
        self.image.prompt_encoder = self.video.interactive_sam_prompt_encoder
        self.image.mask_decoder = self.video.interactive_sam_mask_decoder
        self.image.interactivity_no_mem_embed = self.video.interactivity_no_mem_embed
        return self
