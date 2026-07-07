import torch
from torch import nn

from ..components.backbone.create import create_vision_backbone


class VisionCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_backbone = create_vision_backbone()

    def from_ckpt(self, ckpt, strict: bool = False):
        self.load_state_dict(ckpt.block_state("image.vision"), strict=strict)
        return self

    def forward(
        self,
        images: torch.Tensor,
        need_sam3: bool = True,
        need_interactive: bool = True,
        need_propagation: bool = True,
    ) -> dict[str, dict[str, object] | None]:
        (
            sam3_features,
            sam3_pos,
            interactive_features,
            interactive_pos,
            propagation_features,
            propagation_pos,
        ) = self.vision_backbone(
            images,
            need_sam3_out=need_sam3,
            need_interactive_out=need_interactive,
            need_propagation_out=need_propagation,
        )

        return {
            "sam3": self.branch(
                "sam3",
                sam3_features,
                sam3_pos,
                need_sam3,
            ),
            "interactive": self.branch(
                "interactive",
                interactive_features,
                interactive_pos,
                need_interactive,
            ),
            "propagation": self.branch(
                "propagation",
                propagation_features,
                propagation_pos,
                need_propagation,
            ),
        }

    @staticmethod
    def branch(name: str, features, pos_enc, required: bool):
        if not required:
            return None
        if not features:
            raise RuntimeError(f"{name} vision branch returned no feature maps")
        if len(features) != len(pos_enc):
            raise RuntimeError(
                f"{name} vision branch returned {len(features)} feature maps "
                f"but {len(pos_enc)} position encodings"
            )

        last = features[-1]
        return {
            "vision_features": getattr(last, "tensors", last),
            "vision_mask": getattr(last, "mask", None),
            "vision_pos_enc": tuple(pos_enc),
            "backbone_fpn": tuple(features),
        }
