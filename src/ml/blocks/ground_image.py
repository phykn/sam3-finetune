import torch
from torch import nn


class GroundImage(nn.Module):
    def forward(self, features) -> dict[str, object]:
        fpn = tuple(features["backbone_fpn"])
        if not fpn:
            raise RuntimeError("grounding image block expected feature levels")

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
        tensor = GroundImage.tensor(value)
        return int(tensor.shape[-2]), int(tensor.shape[-1])
