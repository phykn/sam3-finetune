import torch
from torch import nn

from ...io.checkpoint import Checkpoint, load_visual
from ..blocks.grounding.decoder import GroundingDecoder
from ..blocks.grounding.image import GroundingImage
from ..blocks.grounding.prompt import GroundingPromptEncoder
from ..blocks.grounding.tokens import VisualTokens
from ..blocks.vision import VisionEncoder


class Sam3GroundingModel(nn.Module):
    def __init__(self, path=None, visual_path=None) -> None:
        super().__init__()
        self.vision = VisionEncoder()
        self.cond = VisualTokens()
        if visual_path is not None:
            self.cond.load_weights(load_visual(visual_path))
        self.ground_image = GroundingImage()
        self.ground_prompt = GroundingPromptEncoder()
        self.ground_dec = GroundingDecoder()
        if path is not None:
            self.load_weights(Checkpoint.load(path))

    def load_weights(self, ckpt):
        self.vision.load_weights(ckpt)
        self.ground_prompt.load_weights(ckpt)
        self.ground_dec.load_weights(ckpt)
        return self

    def encode_image(self, images: torch.Tensor) -> dict[str, object]:
        features = self.vision(
            images,
            need_sam3=True,
            need_interactive=False,
            need_propagation=False,
        )
        if features["sam3"] is None:
            raise RuntimeError("grounding model expected sam3 vision features")
        return self.ground_image(features["sam3"])

    def encode_prompt(
        self,
        image,
        prompt=None,
        boxes=None,
        box_labels=None,
        box_mask=None,
        points=None,
        point_labels=None,
        point_mask=None,
        masks=None,
        mask_labels=None,
        mask_mask=None,
    ):
        return self.ground_prompt(
            image,
            prompt=prompt,
            boxes=boxes,
            box_labels=box_labels,
            box_mask=box_mask,
            points=points,
            point_labels=point_labels,
            point_mask=point_mask,
            masks=masks,
            mask_labels=mask_labels,
            mask_mask=mask_mask,
        )

    def encode_box_prompts(self, image, boxes, labels, box_mask):
        image = GroundingImage.expand(image, boxes.shape[1])
        return self.ground_prompt(
            image,
            boxes=boxes,
            box_labels=labels,
            box_mask=box_mask,
        )

    def decode(self, image, prompt):
        image = GroundingImage.expand(image, prompt["features"].shape[1])
        return self.ground_dec(image, self.cond(), prompt)

    def forward(
        self,
        images: torch.Tensor,
        prompt=None,
        boxes=None,
        box_labels=None,
        box_mask=None,
        points=None,
        point_labels=None,
        point_mask=None,
        masks=None,
        mask_labels=None,
        mask_mask=None,
    ):
        image = self.encode_image(images)
        prompt = self.encode_prompt(
            image,
            prompt=prompt,
            boxes=boxes,
            box_labels=box_labels,
            box_mask=box_mask,
            points=points,
            point_labels=point_labels,
            point_mask=point_mask,
            masks=masks,
            mask_labels=mask_labels,
            mask_mask=mask_mask,
        )
        return self.decode(image, prompt)
