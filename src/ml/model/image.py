import torch
from torch import nn

from ...io.checkpoint import Checkpoint
from ..blocks.image.features import ImageFeatures
from ..blocks.image.masks import ImageMaskDecoder
from ..blocks.image.prompt import ImagePromptEncoder
from ..blocks.vision import VisionEncoder


class Sam3ImageModel(nn.Module):
    def __init__(self, path=None) -> None:
        super().__init__()
        self.vision = VisionEncoder()
        self.sam_image = ImageFeatures()
        self.sam_prompt = ImagePromptEncoder()
        self.sam_mask = ImageMaskDecoder()
        if path is not None:
            self.load_weights(Checkpoint.load(path))

    def load_weights(self, ckpt):
        self.vision.load_weights(ckpt)
        self.sam_image.load_weights(ckpt)
        self.sam_prompt.load_weights(ckpt)
        self.sam_mask.load_weights(ckpt)
        return self

    def encode_image(self, images: torch.Tensor) -> dict[str, object]:
        features = self.vision(
            images,
            need_sam3=False,
            need_interactive=True,
            need_propagation=False,
        )
        if features["interactive"] is None:
            raise RuntimeError("image model expected interactive vision features")

        image_embed, high_res = self.sam_image(features["interactive"])
        return {
            "image_embed": image_embed,
            "high_res_features": list(high_res),
        }

    def encode_prompt(self, points=None, boxes=None, masks=None):
        return self.sam_prompt(points=points, boxes=boxes, masks=masks)

    def decode_masks(
        self,
        image_embed: torch.Tensor,
        high_res_features: tuple[torch.Tensor, ...],
        prompt,
        image_pe: torch.Tensor,
        multimask: bool = True,
        repeat_image: bool = False,
        mix: torch.Tensor | None = None,
        cond=None,
        prompt_type=None,
    ):
        return self.sam_mask(
            image_embed,
            high_res_features,
            prompt,
            image_pe,
            multimask,
            repeat_image,
            mix=mix,
        )

    @property
    def mask_input_size(self):
        return self.sam_prompt.prompt_encoder.mask_input_size

    def get_image_position_encoding(self, device=None):
        pe = self.sam_prompt.prompt_encoder.get_dense_pe()
        return pe if device is None else pe.to(device)
