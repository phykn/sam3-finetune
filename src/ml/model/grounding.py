import torch
from torch import nn

from ...io.checkpoint import Checkpoint, load_visual
from ..blocks.grounding.decoder import GroundingDecoder
from ..blocks.grounding.image import GroundingImage
from ..blocks.grounding.prompt import GroundingPromptEncoder
from ..blocks.grounding.tokens import VisualTokens
from ..blocks.image.features import ImageFeatures
from ..blocks.image.masks import ImageMaskDecoder
from ..blocks.image.prompt import ImagePromptEncoder
from ..blocks.vision import VisionEncoder


class Sam3GroundingModel(nn.Module):
    def __init__(self, path=None, visual_path=None, use_sam_masks=False) -> None:
        super().__init__()
        self.use_sam_masks = bool(use_sam_masks)
        self.vision = VisionEncoder()
        self.cond = VisualTokens()
        if visual_path is not None:
            self.cond.load_weights(load_visual(visual_path))
        self.ground_image = GroundingImage()
        self.ground_prompt = GroundingPromptEncoder()
        self.ground_dec = GroundingDecoder()
        if self.use_sam_masks:
            self.sam_image = ImageFeatures()
            self.sam_prompt = ImagePromptEncoder()
            self.sam_mask = ImageMaskDecoder()
        if path is not None:
            self.load_weights(Checkpoint.load(path))

    def load_weights(self, ckpt):
        self.vision.load_weights(ckpt)
        self.ground_prompt.load_weights(ckpt)
        self.ground_dec.load_weights(ckpt)
        if self.use_sam_masks:
            self.sam_image.load_weights(ckpt)
            self.sam_prompt.load_weights(ckpt)
            self.sam_mask.load_weights(ckpt)
        return self

    def encode_image(self, images: torch.Tensor) -> dict[str, object]:
        features = self.vision(
            images,
            need_sam3=True,
            need_interactive=self.use_sam_masks,
            need_propagation=False,
        )
        if features["sam3"] is None:
            raise RuntimeError("grounding model expected sam3 vision features")
        out = self.ground_image(features["sam3"])
        if self.use_sam_masks:
            interactive = features["interactive"]
            if interactive is None:
                raise RuntimeError(
                    "grounding model expected interactive vision features"
                )
            image_embed, high_res = self.sam_image(interactive)
            out["mask_image_embed"] = image_embed
            out["mask_high_res"] = tuple(high_res)
        return out

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

    @property
    def mask_input_size(self):
        return self.sam_prompt.prompt_encoder.mask_input_size

    def encode_mask_prompts(self, points, masks=None):
        return self.sam_prompt(points=points, boxes=None, masks=masks)

    def decode_sam_mask(self, image, prompt):
        image_pe = self.sam_prompt.prompt_encoder.get_dense_pe().to(
            image["mask_image_embed"]
        )
        return self.sam_mask(
            image["mask_image_embed"],
            image["mask_high_res"],
            prompt,
            image_pe,
            multimask=False,
            repeat_image=True,
        )

    def mask_stability(self, masks):
        return self.sam_mask.mask_decoder.stability_scores(masks)

    def decode_point(self, image, prompt):
        image_pe = self.sam_prompt.prompt_encoder.get_dense_pe().to(
            image["mask_image_embed"]
        )
        return self.sam_mask(
            image["mask_image_embed"],
            image["mask_high_res"],
            prompt,
            image_pe,
            multimask=True,
            repeat_image=False,
        )

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
