from torch import nn

from ...components.sam.prompt_encoder import PromptEncoder


class ImagePromptEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(72, 72),
            input_image_size=(1008, 1008),
            mask_in_chans=16,
        )

    def load_weights(self, ckpt):
        ckpt.load_block("image.prompt", self)
        return self

    def forward(self, points=None, boxes=None, masks=None):
        return self.prompt_encoder(points=points, boxes=boxes, masks=masks)
