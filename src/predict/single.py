from contextlib import nullcontext

import torch

from ..data import image as image_data, prompt
from .mask import format as mask_format


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

    def encode(self, image):
        tensor, orig_hw = image_data.make_tensor(image, self.image_size, self.device)
        with self.autocast():
            out = self.model.encode_image(tensor)
        return {
            "image_embed": out["image_embed"],
            "high_res": tuple(out["high_res_features"]),
            "orig_hw": orig_hw,
        }

    @torch.inference_mode()
    def predict(
        self,
        image,
        point_coords=None,
        point_labels=None,
        box=None,
        mask=None,
        multimask: bool = True,
    ):
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
    def refine(self, image, logit):
        return self.predict(image, mask=logit, multimask=False)

    @torch.inference_mode()
    def refine_low(self, embed, logit, point_coords=None, point_labels=None):
        return self.predict_embed_low(
            embed,
            point_coords=point_coords,
            point_labels=point_labels,
            mask=logit,
            multimask=False,
        )

    @torch.inference_mode()
    def predict_embed(
        self,
        embed,
        point_coords=None,
        point_labels=None,
        box=None,
        mask=None,
        multimask: bool = True,
    ):
        masks, scores = self._decode(
            embed,
            point_coords,
            point_labels,
            box,
            mask,
            multimask,
        )
        return mask_format.make_full(
            masks, scores, embed["orig_hw"], self.mask_threshold
        )

    @torch.inference_mode()
    def predict_embed_low(
        self,
        embed,
        point_coords=None,
        point_labels=None,
        box=None,
        mask=None,
        multimask: bool = True,
    ):
        masks, scores = self._decode(
            embed,
            point_coords,
            point_labels,
            box,
            mask,
            multimask,
        )
        return mask_format.make_low(masks, scores, self.mask_threshold)

    def _decode(self, embed, point_coords, point_labels, box, mask, multimask):
        sam_prompt = self._make_prompt(embed, point_coords, point_labels, box, mask)
        with self.autocast():
            encoded_prompt = self.model.encode_prompt(
                points=sam_prompt[0],
                boxes=None,
                masks=sam_prompt[1],
            )
            masks, scores, *_ = self.model.decode_masks(
                embed["image_embed"],
                embed["high_res"],
                encoded_prompt,
                self.get_image_pe(),
                multimask,
                True,
            )
        return masks, scores

    def autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def get_image_pe(self):
        if self._image_pe is None or self._image_pe.device != self.device:
            self._image_pe = self.model.prompt_encoder.get_dense_pe().to(self.device)
        return self._image_pe

    def _make_prompt(self, embed, point_coords, point_labels, box, mask):
        point_prompt = self._merge_prompt(
            prompt.build_box(box, embed["orig_hw"], self.image_size, self.device),
            prompt.build_points(
                point_coords,
                point_labels,
                embed["orig_hw"],
                self.image_size,
                self.device,
            ),
        )
        mask_prompt = prompt.build_mask(
            mask,
            self.model.prompt_encoder.mask_input_size,
            self.device,
        )

        if point_prompt is None and mask_prompt is None:
            raise ValueError("prompt is required")
        if point_prompt is None:
            point_prompt = self._make_dummy_prompt(mask_prompt.shape[0])
        return point_prompt, mask_prompt

    def _merge_prompt(self, first, second):
        if first is None:
            return second
        if second is None:
            return first
        return torch.cat([first[0], second[0]], dim=1), torch.cat(
            [first[1], second[1]],
            dim=1,
        )

    def _make_dummy_prompt(self, batch_size: int):
        return (
            torch.zeros(batch_size, 1, 2, device=self.device),
            -torch.ones(batch_size, 1, dtype=torch.int, device=self.device),
        )
