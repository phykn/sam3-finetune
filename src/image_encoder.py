from __future__ import annotations

import torch
from torch import nn

from .neck import Sam3TriViTDetNeck


class InteractiveImageEncoder(nn.Module):
    def __init__(self, vision_backbone: Sam3TriViTDetNeck) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone

    def forward(
        self,
        images: torch.Tensor,
        mask_decoder: nn.Module,
        interactivity_no_mem_embed: torch.Tensor | None = None,
    ) -> dict[str, object]:
        _, _, interactive_features, _interactive_pos, _, _ = self.vision_backbone(
            images,
            need_sam3_out=False,
            need_interactive_out=True,
            need_propagation_out=False,
        )

        if len(interactive_features) < 3:
            raise RuntimeError("Interactive encoder expected three feature levels")

        interactive_features[0].tensors = mask_decoder.conv_s0(
            interactive_features[0].tensors
        )
        interactive_features[1].tensors = mask_decoder.conv_s1(
            interactive_features[1].tensors
        )

        image_embed = interactive_features[-1].tensors
        if interactivity_no_mem_embed is not None:
            no_mem_embed = interactivity_no_mem_embed.view(1, -1, 1, 1).to(
                device=image_embed.device,
                dtype=image_embed.dtype,
            )
            image_embed = image_embed + no_mem_embed

        return {
            "image_embed": image_embed,
            "high_res_features": [
                interactive_features[0].tensors,
                interactive_features[1].tensors,
            ],
        }
