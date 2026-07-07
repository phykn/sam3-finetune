import torch
from torch import nn


class VideoFeat(nn.Module):
    def __init__(self, vision_backbone=None, scalp: int = 0) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone
        self.scalp = scalp

    def forward_image(
        self,
        samples,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ):
        if self.vision_backbone is None:
            raise RuntimeError("video feature block has no vision backbone")

        (
            sam3_features,
            sam3_pos,
            interactive_features,
            interactive_pos,
            propagation_features,
            propagation_pos,
        ) = self.vision_backbone(
            samples,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )

        sam3_features, sam3_pos = self.scalp_levels(sam3_features, sam3_pos)
        interactive_features, interactive_pos = self.scalp_levels(
            interactive_features,
            interactive_pos,
        )
        propagation_features, propagation_pos = self.scalp_levels(
            propagation_features,
            propagation_pos,
        )

        out = {}
        if need_sam3_out:
            out.update(self.out(sam3_features, sam3_pos))
        if need_interactive_out:
            out["interactive"] = self.out(interactive_features, interactive_pos)
        if need_propagation_out:
            out["sam2_backbone_out"] = self.out(
                propagation_features,
                propagation_pos,
            )
        return out

    def forward(self, features) -> dict[str, object]:
        fpn = tuple(features["backbone_fpn"])
        if not fpn:
            raise RuntimeError("video feature block expected feature levels")

        return {
            "vision_features": self.tensor(fpn[-1]),
            "vision_mask": getattr(fpn[-1], "mask", features.get("vision_mask")),
            "vision_pos_enc": tuple(features["vision_pos_enc"]),
            "backbone_fpn": fpn,
            "feat_sizes": tuple(self.size(level) for level in fpn),
        }

    @staticmethod
    def tensor(value) -> torch.Tensor:
        return getattr(value, "tensors", value)

    @staticmethod
    def size(value) -> tuple[int, int]:
        tensor = VideoFeat.tensor(value)
        return int(tensor.shape[-2]), int(tensor.shape[-1])

    def scalp_levels(self, features, pos):
        if self.scalp <= 0:
            return features, pos
        return features[: -self.scalp], pos[: -self.scalp]

    @classmethod
    def out(cls, features, pos):
        last = features[-1]
        return {
            "vision_features": cls.tensor(last),
            "vision_mask": getattr(last, "mask", None),
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
