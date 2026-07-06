import torch
from torch import nn

from ..components.grounding.create import create_geometry_encoder
from ..components.grounding.prompt import Prompt


class GroundPrompt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = create_geometry_encoder()

    def from_ckpt(self, ckpt, strict=False):
        self.encoder.load_state_dict(
            ckpt.block_state("grounding.geometry_encoder"),
            strict=strict,
        )
        return self

    def forward(
        self,
        image,
        prompt: Prompt | None = None,
        boxes: torch.Tensor | None = None,
        box_labels: torch.Tensor | None = None,
        box_mask: torch.Tensor | None = None,
        points: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        mask_labels: torch.Tensor | None = None,
        mask_mask: torch.Tensor | None = None,
    ) -> dict[str, object]:
        if prompt is None:
            prompt = self.build_prompt(
                image,
                boxes,
                box_labels,
                box_mask,
                points,
                point_labels,
                point_mask,
                masks,
                mask_labels,
                mask_mask,
            )

        out = self.encoder(
            geo_prompt=prompt,
            img_feats=[self.seq(level) for level in image["backbone_fpn"]],
            img_sizes=list(image["feat_sizes"]),
            img_pos_embeds=[self.seq(pos) for pos in image["vision_pos_enc"]],
        )
        features, prompt_mask = self.out(out)
        return {"features": features, "mask": prompt_mask, "prompt": prompt}

    @staticmethod
    def build_prompt(
        image,
        boxes,
        box_labels,
        box_mask,
        points,
        point_labels,
        point_mask,
        masks,
        mask_labels,
        mask_mask,
    ) -> Prompt:
        if boxes is None and points is None and masks is None:
            batch_size = image["vision_features"].shape[0]
            device = image["vision_features"].device
            boxes = torch.zeros(0, batch_size, 4, device=device)

        return Prompt(
            box_embeddings=boxes,
            box_mask=box_mask,
            point_embeddings=points,
            point_mask=point_mask,
            box_labels=box_labels,
            point_labels=point_labels,
            mask_embeddings=masks,
            mask_mask=mask_mask,
            mask_labels=mask_labels,
        )

    @staticmethod
    def seq(value) -> torch.Tensor:
        tensor = getattr(value, "tensors", value)
        if tensor.dim() == 4:
            return tensor.flatten(2).permute(2, 0, 1)
        if tensor.dim() == 3:
            return tensor
        raise RuntimeError(
            f"expected 3D or 4D image feature, got {tuple(tensor.shape)}"
        )

    @staticmethod
    def out(out) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(out, dict):
            features = out.get("features", out.get("prompt_features"))
            mask = out.get("mask", out.get("prompt_mask"))
            if features is not None and mask is not None:
                return features, mask
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return out[0], out[1]
        raise RuntimeError("grounding prompt encoder must return features and mask")
