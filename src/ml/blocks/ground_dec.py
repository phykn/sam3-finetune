import torch
from torch import nn

from ...ops.box import convert_to_xyxy
from ...ops.tensor import invert_sigmoid
from ..components.grounding.box_out import write_box_outputs
from ..components.grounding.create import (
    create_dot_product_scorer,
    create_segmentation_head,
    create_transformer,
)
from ..runtime.checkpointing import activation_ckpt_wrapper


def count_queries(hs, apply_dac):
    num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
    return num_o2o, hs.size(2) - num_o2o


class GroundDec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = create_transformer()
        self.scorer = create_dot_product_scorer()
        self.seg_head = create_segmentation_head()

    def from_ckpt(self, ckpt, strict=False):
        self.transformer.load_state_dict(
            ckpt.block_state("grounding.transformer"),
            strict=strict,
        )
        self.scorer.load_state_dict(
            ckpt.block_state("grounding.dot_prod_scoring"),
            strict=strict,
        )
        self.seg_head.load_state_dict(
            ckpt.block_state("grounding.segmentation_head"),
            strict=strict,
        )
        return self

    def forward(self, image, cond: dict, prompt) -> dict[str, object]:
        image_features, image_pos, sizes = self.image_inputs(image)
        prompt_features, prompt_mask = self.prompt_inputs(cond, prompt)
        enc = self.encode(
            image_features, image_pos, sizes, prompt_features, prompt_mask
        )
        out, hs = self.decode(enc, prompt_features, prompt_mask)
        self.segment(out, image, hs, prompt_features, prompt_mask)
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
        self.score_and_box(out, hs, refs, prompt, prompt_mask, presence_logits)
        return out, hs

    def score_and_box(self, out, hs, refs, prompt, prompt_mask, presence_logits=None):
        apply_dac = self.transformer.decoder.dac and self.training
        num_o2o, num_o2m = count_queries(hs, apply_dac)
        out["queries"] = hs[-1][:, :num_o2o]

        scores = self.scorer(hs, prompt, prompt_mask)
        if presence_logits is not None:
            scores = invert_sigmoid(
                scores.sigmoid() * presence_logits.sigmoid().unsqueeze(2)
            ).clamp(min=-10.0, max=10.0)
        boxes = (
            invert_sigmoid(refs) + self.transformer.decoder.bbox_embed(hs)
        ).sigmoid()
        write_box_outputs(
            out=out,
            scores=scores,
            boxes=boxes,
            boxes_xyxy=convert_to_xyxy(boxes),
            num_o2o=num_o2o,
            num_o2m=num_o2m,
            training=self.training,
        )

    def segment(self, out, image, hs, prompt, prompt_mask):
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
            [GroundDec.seq(x) for x in features],
            [GroundDec.seq(x) for x in pos],
            sizes,
        )

    @staticmethod
    def prompt_inputs(cond, prompt):
        features = torch.cat([cond["language_features"], prompt["features"]], dim=0)
        mask = torch.cat([cond["language_mask"], prompt["mask"]], dim=1)
        return features, mask

    @staticmethod
    def seq(value):
        tensor = getattr(value, "tensors", value)
        return tensor.flatten(2).permute(2, 0, 1)
