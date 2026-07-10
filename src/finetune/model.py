from typing import Any

import torch
from torch import nn

from ..ml.model import Sam3ImageModel
from .adapter import FeatureAdapter, LoraLinear
from .prompt import build_prompt
from .router import Router

LORA_NAMES = {"q_proj", "k_proj", "v_proj", "out_proj", "lin1", "lin2"}


class FinetuneModel(nn.Module):
    def __init__(
        self,
        model: Sam3ImageModel,
        num_conditions: int,
        num_experts: int,
        num_labels: int,
        lora_rank: int = 8,
        feature_rank: int = 16,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")
        self.model = model
        self.size = 1008
        self.router = Router(
            image_dim=256,
            num_conditions=num_conditions,
            num_experts=num_experts,
        )
        self.image_adapter = FeatureAdapter(256, feature_rank, num_experts)
        self.high_adapter0 = FeatureAdapter(32, feature_rank, num_experts)
        self.high_adapter1 = FeatureAdapter(64, feature_rank, num_experts)
        self.class_head = nn.Linear(256, num_labels)
        self._freeze_model()
        self._wrap_decoder_linear(lora_rank, num_experts)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        image = batch["image"]
        cond = batch["cond"].to(device=image.device, dtype=torch.long)
        prompts = batch["prompt"]

        encoded = self.encode_image(image)
        image_embed = encoded["image_embed"]
        high_res = tuple(encoded["high_res_features"])

        decoded = []
        image_pe = self.get_image_position_encoding(image.device)
        for index, prompt in enumerate(prompts):
            decoded.append(
                self._decode_prompt(
                    prompt,
                    image_embed[index : index + 1],
                    (
                        high_res[0][index : index + 1],
                        high_res[1][index : index + 1],
                    ),
                    image_pe,
                    cond[index : index + 1],
                    prompt["type"],
                    image.device,
                )
            )

        masks, ious, classes = zip(*decoded)
        return {
            "mask_logits": torch.cat(masks, dim=0),
            "iou_scores": torch.cat(ious, dim=0),
            "class_logits": torch.cat(classes, dim=0),
        }

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]

    @property
    def mask_input_size(self):
        return self.model.mask_input_size

    def get_image_position_encoding(self, device=None):
        return self.model.get_image_position_encoding(device)

    def encode_image(self, images: torch.Tensor) -> dict[str, object]:
        with torch.no_grad():
            return self.model.encode_image(images)

    def encode_prompt(self, points=None, boxes=None, masks=None):
        with torch.no_grad():
            return self.model.encode_prompt(points=points, boxes=boxes, masks=masks)

    def decode_masks(
        self,
        image_embed: torch.Tensor,
        high_res_features: tuple[torch.Tensor, ...],
        prompt,
        image_pe: torch.Tensor,
        multimask: bool = True,
        repeat_image: bool = False,
        cond: int | torch.Tensor | None = None,
        prompt_type: str | list[str] | tuple[str, ...] | torch.Tensor | None = None,
    ):
        mix = self._make_mix(image_embed, cond, prompt_type)
        image_embed, high_res = self._adapt_features(
            image_embed,
            (high_res_features[0], high_res_features[1]),
            mix,
        )
        masks, ious, tokens, objects = self.model.decode_masks(
            image_embed,
            high_res,
            prompt,
            image_pe,
            multimask,
            repeat_image,
            mix=mix,
        )
        class_logits = self.class_head(tokens)
        if class_logits.shape[:2] != masks.shape[:2]:
            raise ValueError("class tokens must align with masks")
        return masks, ious, tokens, objects, class_logits

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def _freeze_model(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def _adapt_features(
        self,
        image_embed: torch.Tensor,
        high_res: tuple[torch.Tensor, torch.Tensor],
        mix: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.image_adapter(image_embed, mix), (
            self.high_adapter0(high_res[0], mix),
            self.high_adapter1(high_res[1], mix),
        )

    def _make_mix(
        self,
        image_embed: torch.Tensor,
        cond: int | torch.Tensor | None,
        prompt_type: str | list[str] | tuple[str, ...] | torch.Tensor | None,
    ) -> torch.Tensor:
        batch = image_embed.shape[0]
        cond = self._make_cond(cond, batch, image_embed.device)
        prompt_type = self._make_prompt_type(prompt_type, batch)
        return self.router(image_embed, cond, prompt_type)

    def _make_cond(
        self,
        cond: int | torch.Tensor | None,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        if cond is None:
            return torch.zeros(batch, dtype=torch.long, device=device)
        if isinstance(cond, int):
            return torch.full((batch,), cond, dtype=torch.long, device=device)

        cond = torch.as_tensor(cond, dtype=torch.long, device=device).flatten()
        if cond.numel() == batch:
            return cond
        if cond.numel() == 1:
            return cond.repeat(batch)
        if batch == 1:
            return cond[:1]
        raise ValueError("cond length must match image batch")

    def _make_prompt_type(
        self,
        prompt_type: str | list[str] | tuple[str, ...] | torch.Tensor | None,
        batch: int,
    ) -> list[str] | torch.Tensor:
        if prompt_type is None:
            return ["point"] * batch
        if isinstance(prompt_type, str):
            return [prompt_type] * batch
        if isinstance(prompt_type, torch.Tensor):
            prompt_type = prompt_type.flatten()
            if prompt_type.numel() == batch:
                return prompt_type
            if prompt_type.numel() == 1:
                return prompt_type.repeat(batch)
            if batch == 1:
                return prompt_type[:1]
            raise ValueError("prompt_type length must match image batch")

        prompt_type = list(prompt_type)
        if len(prompt_type) == batch:
            return prompt_type
        if len(prompt_type) == 1:
            return prompt_type * batch
        if batch == 1:
            return prompt_type[:1]
        raise ValueError("prompt_type length must match image batch")

    def _decode_prompt(
        self,
        prompt: dict[str, Any],
        image_embed: torch.Tensor,
        high_res: tuple[torch.Tensor, torch.Tensor],
        image_pe: torch.Tensor,
        cond: torch.Tensor,
        prompt_type: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point_prompt, mask_prompt = build_prompt(
            prompt,
            self.size,
            self.model.mask_input_size,
            device,
        )
        encoded_prompt = self.encode_prompt(
            points=point_prompt,
            boxes=None,
            masks=mask_prompt,
        )
        mask, iou, _token, _obj, class_logits = self.decode_masks(
            image_embed,
            high_res,
            encoded_prompt,
            image_pe,
            multimask=False,
            repeat_image=False,
            cond=cond,
            prompt_type=prompt_type,
        )
        return mask, iou, class_logits

    def _wrap_decoder_linear(
        self,
        rank: int,
        num_experts: int,
    ) -> None:
        transformer = self.model.sam_mask.mask_decoder.transformer
        for module in transformer.modules():
            for name in LORA_NAMES:
                child = getattr(module, name, None)
                if not isinstance(child, nn.Linear):
                    continue
                wrapped = LoraLinear(child, rank=rank, num_experts=num_experts)
                setattr(module, name, wrapped)
