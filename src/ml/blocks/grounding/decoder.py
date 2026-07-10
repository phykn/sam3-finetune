import torch
from torch import nn

from ....ops.box import cxcywh_to_xyxy
from ....ops.tensor import inverse_sigmoid
from ...components.grounding.box_out import write_box_outputs
from ...components.grounding.pixel import PixelDecoder
from ...components.grounding.scoring import DotProductScorer
from ...components.grounding.segmentation import UniversalSegmentationHead
from ...components.nn.attention import MultiheadAttention
from ...components.nn.layers import MLP
from ...components.transformer.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from ...components.transformer.encoder import (
    TransformerEncoderFusion,
    TransformerEncoderLayer,
)
from ...components.transformer.model import Transformer
from ...runtime.checkpointing import activation_ckpt_wrapper

MODEL_DIM = 256
FEEDFORWARD_DIM = 2048
HEADS = 8
DROPOUT = 0.1
IMAGE_SIZE = 1008
VIT_STRIDE = 14


def _make_attention(
    batch_first: bool,
    dropout: float = DROPOUT,
) -> MultiheadAttention:
    return MultiheadAttention(
        num_heads=HEADS,
        dropout=dropout,
        embed_dim=MODEL_DIM,
        batch_first=batch_first,
    )


def _make_transformer_encoder() -> TransformerEncoderFusion:
    layer = TransformerEncoderLayer(
        activation="relu",
        d_model=MODEL_DIM,
        dim_feedforward=FEEDFORWARD_DIM,
        dropout=DROPOUT,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=_make_attention(batch_first=True),
        cross_attention=_make_attention(batch_first=True),
    )
    return TransformerEncoderFusion(
        layer=layer,
        num_layers=6,
        d_model=MODEL_DIM,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )


def _make_transformer_decoder() -> TransformerDecoder:
    layer = TransformerDecoderLayer(
        activation="relu",
        d_model=MODEL_DIM,
        dim_feedforward=FEEDFORWARD_DIM,
        dropout=DROPOUT,
        cross_attention=_make_attention(batch_first=False),
        n_heads=HEADS,
        use_text_cross_attention=True,
    )
    return TransformerDecoder(
        layer=layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=MODEL_DIM,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=IMAGE_SIZE,
        stride=VIT_STRIDE,
        use_act_checkpoint=True,
        presence_token=True,
    )


def _make_transformer() -> Transformer:
    return Transformer(
        encoder=_make_transformer_encoder(),
        decoder=_make_transformer_decoder(),
        d_model=MODEL_DIM,
    )


def _make_scorer() -> DotProductScorer:
    prompt_mlp = MLP(
        input_dim=MODEL_DIM,
        hidden_dim=FEEDFORWARD_DIM,
        output_dim=MODEL_DIM,
        num_layers=2,
        dropout=DROPOUT,
        residual=True,
        out_norm=nn.LayerNorm(MODEL_DIM),
    )
    return DotProductScorer(
        d_model=MODEL_DIM,
        d_proj=MODEL_DIM,
        prompt_mlp=prompt_mlp,
    )


def _make_segmentation_head() -> UniversalSegmentationHead:
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=MODEL_DIM,
        compile_mode=None,
    )
    return UniversalSegmentationHead(
        hidden_dim=MODEL_DIM,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=_make_attention(batch_first=False, dropout=0),
        pixel_decoder=pixel_decoder,
    )


def count_queries(hs, apply_dac):
    num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
    return num_o2o, hs.size(2) - num_o2o


class GroundingDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _make_transformer()
        self.scorer = _make_scorer()
        self.seg_head = _make_segmentation_head()

    def load_weights(self, ckpt):
        ckpt.load_block("grounding.decoder", self)
        return self

    def forward(self, image, cond: dict, prompt) -> dict[str, object]:
        image_features, image_pos, sizes = self.image_inputs(image)
        prompt_features, prompt_mask = self.prompt_inputs(cond, prompt)
        enc = self.encode(
            image_features, image_pos, sizes, prompt_features, prompt_mask
        )
        out, hs = self.decode(enc, prompt_features, prompt_mask)
        self.predict_masks(out, image, hs, prompt_features, prompt_mask)
        return {
            "pred_logits": out["pred_logits"],
            "pred_boxes": out["pred_boxes"],
            "pred_masks": out["pred_masks"],
            "raw": out,
        }

    def encode(self, image_features, image_pos, sizes, prompt, prompt_mask):
        memory = self.transformer.encoder(
            src=image_features.copy(),
            src_key_padding_mask=None,
            src_pos=image_pos.copy(),
            prompt=prompt,
            prompt_pos=torch.zeros_like(prompt),
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=sizes,
            encoder_extra_kwargs=None,
        )
        return {
            "memory": memory["memory"],
            "pos": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "sizes": sizes,
        }

    def decode(self, enc, prompt, prompt_mask):
        batch_size = enc["memory"].shape[1]
        query = self.transformer.decoder.query_embed.weight
        tgt = query.unsqueeze(1).repeat(1, batch_size, 1)
        apply_dac = self.transformer.decoder.dac and self.training
        hs, refs, presence_logits, presence_feats = self.transformer.decoder(
            tgt=tgt,
            memory=enc["memory"],
            memory_key_padding_mask=enc["padding_mask"],
            pos=enc["pos"],
            reference_boxes=None,
            level_start_index=enc["level_start_index"],
            spatial_shapes=enc["spatial_shapes"],
            valid_ratios=enc["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=apply_dac,
        )
        hs = hs.transpose(1, 2)
        refs = refs.transpose(1, 2)
        if presence_logits is not None:
            presence_logits = presence_logits.transpose(1, 2)
        out = {
            "encoder_hidden_states": enc["memory"],
            "presence_feats": presence_feats,
        }
        self.predict_detections(out, hs, refs, prompt, prompt_mask, presence_logits)
        return out, hs

    def predict_detections(
        self,
        out,
        hs,
        refs,
        prompt,
        prompt_mask,
        presence_logits=None,
    ):
        apply_dac = self.transformer.decoder.dac and self.training
        num_o2o, num_o2m = count_queries(hs, apply_dac)
        out["queries"] = hs[-1][:, :num_o2o]

        scores = self.scorer(hs, prompt, prompt_mask)
        if presence_logits is not None:
            scores = inverse_sigmoid(
                scores.sigmoid() * presence_logits.sigmoid().unsqueeze(2)
            ).clamp(min=-10.0, max=10.0)
        boxes = (
            inverse_sigmoid(refs) + self.transformer.decoder.bbox_embed(hs)
        ).sigmoid()
        write_box_outputs(
            out=out,
            scores=scores,
            boxes=boxes,
            boxes_xyxy=cxcywh_to_xyxy(boxes),
            num_o2o=num_o2o,
            num_o2m=num_o2m,
            training=self.training,
        )

    def predict_masks(self, out, image, hs, prompt, prompt_mask):
        apply_dac = self.transformer.decoder.dac and self.training
        num_o2o, _ = count_queries(hs, apply_dac)
        seg = activation_ckpt_wrapper(self.seg_head)(
            backbone_feats=image["backbone_fpn"],
            obj_queries=hs,
            image_ids=torch.arange(hs.shape[1], device=hs.device),
            encoder_hidden_states=out["encoder_hidden_states"],
            act_ckpt_enable=self.training,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        for key, val in seg.items():
            out[key] = val[:, :num_o2o] if key in self.seg_head.instance_keys else val

    @staticmethod
    def image_inputs(image):
        features = image["backbone_fpn"][-1:]
        pos = image["vision_pos_enc"][-1:]
        sizes = [x.shape[-2:] for x in pos]
        return (
            [GroundingDecoder.flatten_spatial(x) for x in features],
            [GroundingDecoder.flatten_spatial(x) for x in pos],
            sizes,
        )

    @staticmethod
    def prompt_inputs(cond, prompt):
        features = torch.cat([cond["language_features"], prompt["features"]], dim=0)
        mask = torch.cat([cond["language_mask"], prompt["mask"]], dim=1)
        return features, mask

    @staticmethod
    def flatten_spatial(value):
        tensor = getattr(value, "tensors", value)
        return tensor.flatten(2).permute(2, 0, 1)
