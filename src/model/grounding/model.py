from copy import deepcopy

import torch

from ...ops.box import convert_to_xyxy
from ...ops.tensor import invert_sigmoid
from ..runtime.checkpointing import activation_ckpt_wrapper
from ..structures import FindStage
from .backbone import GroundingVisionBackbone
from .output import write_box_outputs, write_output
from .prompt import Prompt


def map_image_ids(backbone_out, img_ids):
    id_mapping = backbone_out.get("id_mapping")
    if id_mapping is None:
        return img_ids

    img_ids = id_mapping[img_ids]
    torch._assert_async((img_ids >= 0).all())
    return img_ids


def count_queries(hs, apply_dac):
    num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
    return num_o2o, hs.size(2) - num_o2o


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

    def _select_image_features(self, backbone_out, img_ids):
        if "backbone_fpn" not in backbone_out:
            raise RuntimeError("GroundingImageModel expects precomputed backbone_fpn")

        features = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        sizes = [pos.shape[-2:] for pos in pos_embeds]

        image_features = [
            feat[img_ids].flatten(2).permute(2, 0, 1) for feat in features
        ]
        image_pos_embeds = [
            pos[img_ids].flatten(2).permute(2, 0, 1) for pos in pos_embeds
        ]
        return image_features, image_pos_embeds, sizes

    def _encode_prompt(
        self,
        backbone_out,
        find_input,
        geometric_prompt,
        image_features,
        image_pos_embeds,
        image_sizes,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
    ):
        text_ids = find_input.text_ids
        text_features = backbone_out["language_features"][:, text_ids]
        text_mask = backbone_out["language_mask"][text_ids]

        geometry_features, geometry_mask = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=image_features,
            img_sizes=image_sizes,
            img_pos_embeds=image_pos_embeds,
        )
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros(
                (0, *geometry_features.shape[1:]), device=geometry_features.device
            )
            visual_prompt_mask = torch.zeros(
                (*geometry_mask.shape[:-1], 0),
                device=geometry_mask.device,
                dtype=geometry_mask.dtype,
            )

        prompt = torch.cat(
            [text_features, geometry_features, visual_prompt_embed],
            dim=0,
        )
        prompt_mask = torch.cat(
            [text_mask, geometry_mask, visual_prompt_mask],
            dim=1,
        )
        return prompt, prompt_mask

    def _run_encoder(
        self,
        image_features: list[torch.Tensor],
        image_pos_embeds: list[torch.Tensor],
        image_sizes,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> dict:
        prompt_pos_embed = torch.zeros_like(prompt)
        memory = self.transformer.encoder(
            src=image_features.copy(),
            src_key_padding_mask=None,
            src_pos=image_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=image_sizes,
            encoder_extra_kwargs=None,
        )
        encoder_out = {
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": image_sizes,
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        return encoder_out

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
        batch_size = memory.shape[1]
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
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

    def _score_queries(self, hs, prompt, prompt_mask, is_instance_prompt):
        if self.use_dot_prod_scoring:
            head = self.dot_prod_scoring
            if is_instance_prompt and self.instance_dot_prod_scoring is not None:
                head = self.instance_dot_prod_scoring
            return head(hs, prompt, prompt_mask)

        head = self.class_embed
        if is_instance_prompt and self.instance_class_embed is not None:
            head = self.instance_class_embed
        return head(hs)

    def _predict_boxes(self, hs, reference_boxes):
        box_head = self.transformer.decoder.bbox_embed
        box_offsets = box_head(hs)
        boxes = (invert_sigmoid(reference_boxes) + box_offsets).sigmoid()
        return boxes, convert_to_xyxy(boxes)

    def _apply_joint_box_scores(self, scores, dec_presence_out):
        if not self.supervise_joint_box_scores:
            return scores

        assert dec_presence_out is not None
        presence_prob = dec_presence_out.clone().sigmoid()
        if self.detach_presence_in_joint_score:
            presence_prob = presence_prob.detach()
        scores = scores.sigmoid() * presence_prob.unsqueeze(2)
        return invert_sigmoid(scores).clamp(min=-10.0, max=10.0)

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
        num_o2o, num_o2m = count_queries(hs, apply_dac)
        out["queries"] = hs[-1][:, :num_o2o]

        scores = self._score_queries(hs, prompt, prompt_mask, is_instance_prompt)
        boxes, boxes_xyxy = self._predict_boxes(hs, reference_boxes)

        if dec_presence_out is not None:
            write_output(
                out,
                "presence_logit_dec",
                dec_presence_out,
                update_aux=self.training,
            )

        scores = self._apply_joint_box_scores(scores, dec_presence_out)
        write_box_outputs(
            out=out,
            scores=scores,
            boxes=boxes,
            boxes_xyxy=boxes_xyxy,
            num_o2o=num_o2o,
            num_o2m=num_o2m,
            training=self.training,
        )

    def _run_segmentation_heads(
        self,
        out,
        backbone_out,
        img_ids,
        encoder_hidden_states,
        prompt,
        prompt_mask,
        hs,
    ):
        if self.segmentation_head is None:
            backbone_out.pop("backbone_fpn", None)
            return
        apply_dac = self.transformer.decoder.dac and self.training
        num_o2o, num_o2m = count_queries(hs, apply_dac)
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
                write_output(out, key, value[:, :num_o2o], auxiliary=aux_masks)
                if self.o2m_mask_predict and num_o2m > 0:
                    write_output(
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
        img_ids = map_image_ids(backbone_out, find_input.img_ids)
        image_features, image_pos_embeds, image_sizes = self._select_image_features(
            backbone_out,
            img_ids,
        )

        with torch.profiler.record_function("GroundingImageModel._encode_prompt"):
            prompt, prompt_mask = self._encode_prompt(
                backbone_out=backbone_out,
                find_input=find_input,
                geometric_prompt=geometric_prompt,
                image_features=image_features,
                image_pos_embeds=image_pos_embeds,
                image_sizes=image_sizes,
            )

        with torch.profiler.record_function("GroundingImageModel._run_encoder"):
            encoder_out = self._run_encoder(
                image_features=image_features,
                image_pos_embeds=image_pos_embeds,
                image_sizes=image_sizes,
                prompt=prompt,
                prompt_mask=prompt_mask,
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
            self._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=img_ids,
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
