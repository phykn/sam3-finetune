import torch
from torch import nn

from ...components.grounding.geometry import SequenceGeometryEncoder
from ...components.grounding.prompt import Prompt
from ...components.nn.attention import MultiheadAttention
from ...components.nn.position import PositionEmbeddingSine
from ...components.transformer.encoder import TransformerEncoderLayer

MODEL_DIM = 256
FEEDFORWARD_DIM = 2048
HEADS = 8
DROPOUT = 0.1


def _make_attention(batch_first: bool) -> MultiheadAttention:
    return MultiheadAttention(
        num_heads=HEADS,
        dropout=DROPOUT,
        embed_dim=MODEL_DIM,
        batch_first=batch_first,
    )


def _make_position_encoding() -> PositionEmbeddingSine:
    return PositionEmbeddingSine(
        num_pos_feats=MODEL_DIM,
        normalize=True,
        scale=None,
        temperature=10000,
    )


def _make_geometry_encoder() -> SequenceGeometryEncoder:
    layer = TransformerEncoderLayer(
        activation="relu",
        d_model=MODEL_DIM,
        dim_feedforward=FEEDFORWARD_DIM,
        dropout=DROPOUT,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=_make_attention(batch_first=False),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=_make_attention(batch_first=False),
    )
    return SequenceGeometryEncoder(
        pos_enc=_make_position_encoding(),
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=MODEL_DIM,
        num_layers=3,
        layer=layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )


class GroundingPromptEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _make_geometry_encoder()

    def load_weights(self, ckpt):
        ckpt.load_block("grounding.prompt", self)
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
            img_feats=[self.flatten_spatial(level) for level in image["backbone_fpn"]],
            img_sizes=list(image["feat_sizes"]),
            img_pos_embeds=[
                self.flatten_spatial(pos) for pos in image["vision_pos_enc"]
            ],
        )
        features, prompt_mask = self.unpack_output(out)
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
    def flatten_spatial(value) -> torch.Tensor:
        tensor = getattr(value, "tensors", value)
        if tensor.dim() == 4:
            return tensor.flatten(2).permute(2, 0, 1)
        if tensor.dim() == 3:
            return tensor
        raise RuntimeError(
            f"expected 3D or 4D image feature, got {tuple(tensor.shape)}"
        )

    @staticmethod
    def unpack_output(out) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(out, dict):
            features = out.get("features", out.get("prompt_features"))
            mask = out.get("mask", out.get("prompt_mask"))
            if features is not None and mask is not None:
                return features, mask
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return out[0], out[1]
        raise RuntimeError("grounding prompt encoder must return features and mask")
