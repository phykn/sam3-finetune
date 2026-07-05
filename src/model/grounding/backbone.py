import torch


class GroundingVisionBackbone(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module, scalp: int = 1) -> None:
        super().__init__()
        self.vision_backbone = visual
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
