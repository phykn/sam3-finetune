import torch
from torch import nn

from ..vision import make_vision_backbone


class VideoFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_backbone = make_vision_backbone()

    def forward_image(
        self,
        samples,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ):
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

        out = {}
        if need_sam3_out:
            out.update(self._branch(sam3_features, sam3_pos))
        if need_interactive_out:
            out["interactive"] = self._branch(
                interactive_features,
                interactive_pos,
            )
        if need_propagation_out:
            out["sam2_backbone_out"] = self._branch(
                propagation_features,
                propagation_pos,
            )
        return out

    def forward(self, features) -> dict[str, object]:
        fpn = tuple(features["backbone_fpn"])
        if not fpn:
            raise RuntimeError("video feature block expected feature levels")

        return {
            "vision_features": self._tensor(fpn[-1]),
            "vision_mask": getattr(fpn[-1], "mask", features.get("vision_mask")),
            "vision_pos_enc": tuple(features["vision_pos_enc"]),
            "backbone_fpn": fpn,
            "feat_sizes": tuple(self._size(level) for level in fpn),
        }

    @staticmethod
    def _tensor(value) -> torch.Tensor:
        return getattr(value, "tensors", value)

    @classmethod
    def _size(cls, value) -> tuple[int, int]:
        tensor = cls._tensor(value)
        return int(tensor.shape[-2]), int(tensor.shape[-1])

    @classmethod
    def _branch(cls, features, pos):
        last = features[-1]
        return {
            "vision_features": cls._tensor(last),
            "vision_mask": getattr(last, "mask", None),
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
