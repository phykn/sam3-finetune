import torch
from torch import nn

from ...structures import NestedTensor


class GroundingImage(nn.Module):
    def forward(self, features) -> dict[str, object]:
        fpn = tuple(features["backbone_fpn"])
        if not fpn:
            raise RuntimeError("grounding image block expected feature levels")

        return {
            "vision_features": self.unwrap_tensor(fpn[-1]),
            "vision_mask": getattr(fpn[-1], "mask", features.get("vision_mask")),
            "vision_pos_enc": tuple(features["vision_pos_enc"]),
            "backbone_fpn": fpn,
            "feat_sizes": tuple(self.size(level) for level in fpn),
        }

    @staticmethod
    def unwrap_tensor(value) -> torch.Tensor:
        return getattr(value, "tensors", value)

    @staticmethod
    def size(value) -> tuple[int, int]:
        tensor = GroundingImage.unwrap_tensor(value)
        return int(tensor.shape[-2]), int(tensor.shape[-1])

    @classmethod
    def expand(cls, image: dict[str, object], batch_size: int) -> dict[str, object]:
        current = image["vision_features"].shape[0]
        if current == batch_size:
            return image
        if current != 1:
            raise ValueError("encoded image batch does not match prompt batch")
        return {
            "vision_features": cls._expand(image["vision_features"], batch_size),
            "vision_mask": cls._expand(image["vision_mask"], batch_size),
            "vision_pos_enc": tuple(
                cls._expand(value, batch_size) for value in image["vision_pos_enc"]
            ),
            "backbone_fpn": tuple(
                cls._expand(value, batch_size) for value in image["backbone_fpn"]
            ),
            "feat_sizes": image["feat_sizes"],
        }

    @classmethod
    def _expand(cls, value, batch_size):
        if value is None:
            return None
        if isinstance(value, NestedTensor):
            return NestedTensor(
                cls._expand(value.tensors, batch_size),
                cls._expand(value.mask, batch_size),
            )
        shape = (batch_size, *value.shape[1:])
        return value.expand(shape)
