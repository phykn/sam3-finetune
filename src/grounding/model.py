from copy import deepcopy
from typing import Dict, Tuple

import torch

from ..nn.activation_checkpoint import activation_ckpt_wrapper
from ..ops.box import box_cxcywh_to_xyxy
from ..data.structures import FindStage
from ..nn.output import SAM3Output
from ..nn.utils import inverse_sigmoid
from .geometry import Prompt


def _update_out(out, out_name, out_value, auxiliary=True, update_aux=True):
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value


class GroundingVisionBackbone(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module, scalp: int = 1) -> None:
        super().__init__()
        self.vision_backbone = visual
        self.scalp = scalp

    def forward_image(self, samples: torch.Tensor) -> dict[str, object]:
        features, pos, _sam2_features, _sam2_pos = self.vision_backbone(samples)
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


class GroundingImageModel(torch.nn.Module):
    def __init__(
        self,
        *,
        backbone: GroundingVisionBackbone,
        transformer: torch.nn.Module,
        input_geometry_encoder: torch.nn.Module,
        segmentation_head: torch.nn.Module,
        dot_prod_scoring: torch.nn.Module,
        num_feature_levels: int = 1,
        o2m_mask_predict: bool = True,
        use_act_checkpoint_seg_head: bool = True,
        use_dot_prod_scoring: bool = True,
        supervise_joint_box_scores: bool = False,
        detach_presence_in_joint_score: bool = False,
        separate_scorer_for_instance: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = segmentation_head
        self.o2m_mask_predict = o2m_mask_predict
        self.dot_prod_scoring = dot_prod_scoring
        self.use_act_checkpoint_seg_head = use_act_checkpoint_seg_head
        self.use_dot_prod_scoring = use_dot_prod_scoring
        self.supervise_joint_box_scores = supervise_joint_box_scores
        self.detach_presence_in_joint_score = detach_presence_in_joint_score

        if self.use_dot_prod_scoring:
            assert dot_prod_scoring is not None
            self.instance_dot_prod_scoring = None
            if separate_scorer_for_instance:
                self.instance_dot_prod_scoring = deepcopy(dot_prod_scoring)
        else:
            self.class_embed = torch.nn.Linear(self.hidden_dim, 1)
            self.instance_class_embed = None
            if separate_scorer_for_instance:
                self.instance_class_embed = deepcopy(self.class_embed)

        num_o2o_static = self.transformer.decoder.num_queries
        num_o2m_static = self.transformer.decoder.num_o2m_queries
        assert num_o2m_static == (num_o2o_static if self.transformer.decoder.dac else 0)
        self.dac = self.transformer.decoder.dac

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        self._device = None
        return super().to(*args, **kwargs)

    def _get_img_feats(self, backbone_out, img_ids):
        if "backbone_fpn" not in backbone_out:
            raise RuntimeError("GroundingImageModel expects precomputed backbone_fpn")
        if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
            img_ids = backbone_out["id_mapping"][img_ids]
            torch._assert_async((img_ids >= 0).all())

        vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
        img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
        return backbone_out, img_feats, img_pos_embeds, vis_feat_sizes

    def _encode_prompt(
        self,
        backbone_out,
        find_input,
        geometric_prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
    ):
        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]

        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros(
                (0, *geo_feats.shape[1:]), device=geo_feats.device
            )
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )

        prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
        prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
        return prompt, prompt_mask, backbone_out

    def _run_encoder(
        self,
        backbone_out: Dict,
        find_input: FindStage,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> Tuple[Dict, Dict, Tuple]:
        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple
        prompt_pos_embed = torch.zeros_like(prompt)
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=None,
        )
        encoder_out = {
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        return backbone_out, encoder_out, feat_tuple

    def _run_decoder(
        self,
        pos_embed,
        memory,
        src_mask,
        out,
        prompt,
        prompt_mask,
        encoder_out,
    ):
        bs = memory.shape[1]
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        apply_dac = self.transformer.decoder.dac and self.training
        hs, reference_boxes, dec_presence_out, dec_presence_feats = (
            self.transformer.decoder(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=src_mask,
                pos=pos_embed,
                reference_boxes=None,
                level_start_index=encoder_out["level_start_index"],
                spatial_shapes=encoder_out["spatial_shapes"],
                valid_ratios=encoder_out["valid_ratios"],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=apply_dac,
            )
        )
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)
        if dec_presence_out is not None:
            dec_presence_out = dec_presence_out.transpose(1, 2)

        out["presence_feats"] = dec_presence_feats
        self._update_scores_and_boxes(
            out,
            hs,
            reference_boxes,
            prompt,
            prompt_mask,
            dec_presence_out=dec_presence_out,
        )
        return out, hs

    def _update_scores_and_boxes(
        self,
        out,
        hs,
        reference_boxes,
        prompt,
        prompt_mask,
        dec_presence_out=None,
        is_instance_prompt=False,
    ):
        apply_dac = self.transformer.decoder.dac and self.training
        num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
        num_o2m = hs.size(2) - num_o2o
        out["queries"] = hs[-1][:, :num_o2o]
        if self.use_dot_prod_scoring:
            dot_prod_scoring_head = self.dot_prod_scoring
            if is_instance_prompt and self.instance_dot_prod_scoring is not None:
                dot_prod_scoring_head = self.instance_dot_prod_scoring
            outputs_class = dot_prod_scoring_head(hs, prompt, prompt_mask)
        else:
            class_embed_head = self.class_embed
            if is_instance_prompt and self.instance_class_embed is not None:
                class_embed_head = self.instance_class_embed
            outputs_class = class_embed_head(hs)

        box_head = self.transformer.decoder.bbox_embed
        anchor_box_offsets = box_head(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)

        if dec_presence_out is not None:
            _update_out(
                out, "presence_logit_dec", dec_presence_out, update_aux=self.training
            )

        if self.supervise_joint_box_scores:
            assert dec_presence_out is not None
            prob_dec_presence_out = dec_presence_out.clone().sigmoid()
            if self.detach_presence_in_joint_score:
                prob_dec_presence_out = prob_dec_presence_out.detach()
            outputs_class = inverse_sigmoid(
                outputs_class.sigmoid() * prob_dec_presence_out.unsqueeze(2)
            ).clamp(min=-10.0, max=10.0)

        _update_out(
            out, "pred_logits", outputs_class[:, :, :num_o2o], update_aux=self.training
        )
        _update_out(
            out, "pred_boxes", outputs_coord[:, :, :num_o2o], update_aux=self.training
        )
        _update_out(
            out,
            "pred_boxes_xyxy",
            outputs_boxes_xyxy[:, :, :num_o2o],
            update_aux=self.training,
        )
        if num_o2m > 0 and self.training:
            _update_out(
                out,
                "pred_logits_o2m",
                outputs_class[:, :, num_o2o:],
                update_aux=self.training,
            )
            _update_out(
                out,
                "pred_boxes_o2m",
                outputs_coord[:, :, num_o2o:],
                update_aux=self.training,
            )
            _update_out(
                out,
                "pred_boxes_xyxy_o2m",
                outputs_boxes_xyxy[:, :, num_o2o:],
                update_aux=self.training,
            )

    def _run_segmentation_heads(
        self,
        out,
        backbone_out,
        img_ids,
        vis_feat_sizes,
        encoder_hidden_states,
        prompt,
        prompt_mask,
        hs,
    ):
        if self.segmentation_head is None:
            backbone_out.pop("backbone_fpn", None)
            return
        apply_dac = self.transformer.decoder.dac and self.training
        num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
        num_o2m = hs.size(2) - num_o2o
        obj_queries = hs if self.o2m_mask_predict else hs[:, :, :num_o2o]
        seg_head_outputs = activation_ckpt_wrapper(self.segmentation_head)(
            backbone_feats=backbone_out["backbone_fpn"],
            obj_queries=obj_queries,
            image_ids=img_ids,
            encoder_hidden_states=encoder_hidden_states,
            act_ckpt_enable=self.training and self.use_act_checkpoint_seg_head,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        aux_masks = False
        for key, value in seg_head_outputs.items():
            if key in self.segmentation_head.instance_keys:
                _update_out(out, key, value[:, :num_o2o], auxiliary=aux_masks)
                if self.o2m_mask_predict and num_o2m > 0:
                    _update_out(
                        out, f"{key}_o2m", value[:, num_o2o:], auxiliary=aux_masks
                    )
            else:
                out[key] = value

    def forward_grounding(
        self,
        backbone_out,
        find_input: FindStage,
        geometric_prompt: Prompt,
    ):
        with torch.profiler.record_function("GroundingImageModel._encode_prompt"):
            prompt, prompt_mask, backbone_out = self._encode_prompt(
                backbone_out, find_input, geometric_prompt
            )
        with torch.profiler.record_function("GroundingImageModel._run_encoder"):
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }
        with torch.profiler.record_function("GroundingImageModel._run_decoder"):
            out, hs = self._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )
        with torch.profiler.record_function(
            "GroundingImageModel._run_segmentation_heads"
        ):
            seg_img_ids = find_input.img_ids
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                seg_img_ids = backbone_out["id_mapping"][seg_img_ids]
            self._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=seg_img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )
        return out

    def get_dummy_prompt(self, num_prompts: int = 1) -> Prompt:
        device = self.device
        return Prompt(
            box_embeddings=torch.zeros(0, num_prompts, 4, device=device),
            box_mask=torch.zeros(num_prompts, 0, device=device, dtype=torch.bool),
        )
