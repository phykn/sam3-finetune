import torch


class GroundingVisionBackbone(torch.nn.Module):
    def __init__(
        self,
        visual: torch.nn.Module,
        scalp: int = 1,
        language_backbone: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.vision_backbone = visual
        self.language_backbone = language_backbone
        self.scalp = scalp

    def forward_image(self, samples: torch.Tensor) -> dict[str, object]:
        if hasattr(self.vision_backbone, "interactive_convs"):
            output = self.vision_backbone(
                samples,
                need_sam3_out=True,
                need_interactive_out=False,
                need_propagation_out=False,
            )
        else:
            output = self.vision_backbone(samples)

        if len(output) == 6:
            features, pos, *_ = output
            features = [getattr(feature, "tensors", feature) for feature in features]
        else:
            features, pos, *_ = output

        if self.scalp > 0:
            features = features[: -self.scalp]
            pos = pos[: -self.scalp]

        output = {
            "vision_features": features[-1],
            "vision_mask": None,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output

    def forward_text(
        self,
        captions,
        input_boxes=None,
        additional_text=None,
        device="cuda",
    ) -> dict[str, torch.Tensor]:
        if self.language_backbone is None:
            raise RuntimeError("language_backbone is required for visual prompts")

        text_to_encode = list(captions)
        if additional_text is not None:
            text_to_encode += list(additional_text)

        text_attention_mask, text_memory, text_embeds = self.language_backbone(
            text_to_encode,
            input_boxes=input_boxes,
            device=device,
        )
        output = {}
        if additional_text is not None:
            output["additional_text_features"] = text_memory[:, -len(additional_text) :]
            output["additional_text_mask"] = text_attention_mask[
                -len(additional_text) :
            ]
        text_memory = text_memory[:, : len(captions)]
        text_attention_mask = text_attention_mask[: len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = text_embeds
        return output
