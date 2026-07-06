import torch

from . import prompt, transform
from .result import ImageEmbed, SingleResult


class SinglePredictor:
    def __init__(self, model, config: dict | None = None) -> None:
        config = {} if config is None else config
        device = config.get("device", "cuda")
        image_size = config.get("image_size", 1008)
        mask_threshold = config.get("mask_threshold", 0.0)

        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.mask_threshold = float(mask_threshold)
        self.model = model.to(self.device).eval()
        self._image_pe = None

    def encode(self, image) -> ImageEmbed:
        tensor, orig_hw = transform.image_tensor(image, self.image_size, self.device)
        out = self.model.encode_image(tensor)
        return ImageEmbed(
            image_embed=out["image_embed"],
            high_res=tuple(out["high_res_features"]),
            orig_hw=orig_hw,
        )

    @torch.inference_mode()
    def predict(
        self,
        image,
        point_coords=None,
        point_labels=None,
        box=None,
        mask=None,
        multimask: bool = True,
    ) -> SingleResult:
        embed = self.encode(image)
        return self.predict_embed(
            embed,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask=mask,
            multimask=multimask,
        )

    @torch.inference_mode()
    def predict_embed(
        self,
        embed: ImageEmbed,
        point_coords=None,
        point_labels=None,
        box=None,
        mask=None,
        multimask: bool = True,
    ) -> SingleResult:
        sam_prompt = prompt.sam_prompt(
            prompt_encoder=self.model.prompt_encoder,
            embed=embed,
            image_size=self.image_size,
            device=self.device,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask=mask,
        )
        encoded_prompt = self.model.encode_prompt(
            points=sam_prompt[0],
            boxes=None,
            masks=sam_prompt[1],
        )
        masks, scores, *_ = self.model.decode_masks(
            embed.image_embed,
            embed.high_res,
            encoded_prompt,
            self.image_pe(),
            multimask,
            True,
        )
        return self.out(masks, scores, embed.orig_hw)

    def image_pe(self):
        if self._image_pe is None or self._image_pe.device != self.device:
            self._image_pe = self.model.prompt_encoder.get_dense_pe().to(self.device)
        return self._image_pe

    def out(self, masks: torch.Tensor, scores: torch.Tensor, orig_hw) -> SingleResult:
        resized = transform.resize_masks(masks, orig_hw, self.mask_threshold)
        masks = torch.clamp(masks, -32.0, 32.0).float()
        return SingleResult(
            masks=resized.squeeze(0).detach().cpu().numpy(),
            scores=scores.squeeze(0).float().detach().cpu().numpy(),
            logits=masks.squeeze(0).detach().cpu().numpy(),
        )
