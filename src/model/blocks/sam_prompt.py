from torch import nn

from ..components.sam.prompt_encoder import PromptEncoder


class SamPrompt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(72, 72),
            input_image_size=(1008, 1008),
            mask_in_chans=16,
        )

    def from_ckpt(self, ckpt, strict=False):
        self.load_state_dict(ckpt.block_state("image.sam_prompt"), strict=strict)
        return self

    def forward(self, points=None, boxes=None, masks=None):
        return self.prompt_encoder(points=points, boxes=boxes, masks=masks)
