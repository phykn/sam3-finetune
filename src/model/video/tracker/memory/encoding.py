from collections.abc import Iterable
from typing import Optional

import torch

from ..multiplex.state import MultiplexState
from .encoder import SimpleMaskEncoder


def encode_new_memory(
    self,
    image,
    current_vision_feats,
    feat_sizes,
    pred_masks_high_res,
    object_score_logits,
    is_mask_from_pts,
    *,
    conditioning_objects: Optional[Iterable[int]] = None,
    multiplex_state: MultiplexState,
):
    batch_size = current_vision_feats[-1].size(1)
    channels = self.hidden_dim
    height, width = feat_sizes[-1]
    # Shape: (HW)BC -> BCHW.
    pix_feat = current_vision_feats[-1].permute(1, 2, 0)
    pix_feat = pix_feat.view(batch_size, channels, height, width)

    if self.non_overlap_masks_for_mem_enc and not self.training:
        pred_masks_high_res = self._apply_non_overlapping_constraints(
            pred_masks_high_res
        )

    mask_for_mem = prepare_memory_mask(self, pred_masks_high_res, is_mask_from_pts)
    conditioning_objects, unconditioning_objects = split_conditioning_objects(
        self,
        conditioning_objects,
        multiplex_state,
    )
    mux_mask_for_mem = mux_memory_mask(
        self,
        mask_for_mem,
        conditioning_objects,
        multiplex_state,
    )
    maskmem_features, maskmem_pos_enc = run_maskmem_backbone(
        self,
        image,
        pix_feat,
        mux_mask_for_mem,
    )
    maskmem_features = add_no_object_embedding(
        self,
        maskmem_features,
        object_score_logits,
        multiplex_state,
    )
    maskmem_features = add_condition_embeddings(
        self,
        maskmem_features,
        unconditioning_objects,
        multiplex_state,
    )
    return demux_memory_output(maskmem_features, maskmem_pos_enc, multiplex_state)


def prepare_memory_mask(self, pred_masks_high_res, is_mask_from_pts):
    if not self.apply_sigmoid_to_mask_logits_for_mem_enc:
        return pred_masks_high_res

    assert (
        not self.binarize_mask_from_pts_for_mem_enc
    ), "haven't been trained this way; beware of hardcoded config override"
    binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
    if binarize and not self.training:
        mask_for_mem = (pred_masks_high_res > 0).float()
    else:
        mask_for_mem = torch.sigmoid(pred_masks_high_res)

    if self.sigmoid_scale_for_mem_enc != 1.0:
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
    if self.sigmoid_bias_for_mem_enc != 0.0:
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
    return mask_for_mem


def split_conditioning_objects(self, conditioning_objects, multiplex_state):
    if not self.add_object_conditional_embeddings and not self.condition_as_mask_input:
        return None, None

    if conditioning_objects is None:
        return [], sorted(list(multiplex_state.get_all_valid_object_idx()))

    conditioning_objects = sorted(list(conditioning_objects))
    all_objects_idx = multiplex_state.get_all_valid_object_idx()
    unconditioning_objects = sorted(
        [index for index in all_objects_idx if index not in conditioning_objects]
    )
    return conditioning_objects, unconditioning_objects


def mux_memory_mask(self, mask_for_mem, conditioning_objects, multiplex_state):
    mux_mask_for_mem = multiplex_state.mux(mask_for_mem).squeeze(2)
    if not self.condition_as_mask_input:
        return mux_mask_for_mem

    num_objects = mask_for_mem.shape[0]
    cond_values = torch.full(
        (num_objects,),
        self.condition_as_mask_input_bg,
        device=mask_for_mem.device,
        dtype=mask_for_mem.dtype,
    )
    if len(conditioning_objects) > 0:
        cond_values[conditioning_objects] = self.condition_as_mask_input_fg

    # Shape: [N] -> [N, 1, H, W].
    embedded_conditions = cond_values.view(-1, 1, 1, 1).expand_as(mask_for_mem)
    embedded_conditions = multiplex_state.mux(embedded_conditions).squeeze(2)
    return torch.cat([mux_mask_for_mem, embedded_conditions], dim=1)


def run_maskmem_backbone(self, image, pix_feat, mux_mask_for_mem):
    if isinstance(self.maskmem_backbone, SimpleMaskEncoder):
        maskmem_out = self.maskmem_backbone(
            pix_feat,
            mux_mask_for_mem,
            skip_mask_sigmoid=True,
        )
    else:
        maskmem_out = self.maskmem_backbone(image, pix_feat, mux_mask_for_mem)

    maskmem_features = self._maybe_clone(maskmem_out["vision_features"])
    maskmem_pos_enc = [self._maybe_clone(pos) for pos in maskmem_out["vision_pos_enc"]]
    return maskmem_features, maskmem_pos_enc


def add_no_object_embedding(
    self,
    maskmem_features,
    object_score_logits,
    multiplex_state,
):
    if self.no_obj_embed_spatial is None:
        return maskmem_features

    no_obj_embed_spatial = self.no_obj_embed_spatial.unsqueeze(0).repeat(
        multiplex_state.num_buckets, 1, 1
    )
    object_score_logits = align_object_scores(object_score_logits, multiplex_state)
    object_score_logits = multiplex_state.mux(object_score_logits)
    is_obj_appearing = (object_score_logits > self.object_score_logit_threshold).float()

    no_obj_embed = ((1 - is_obj_appearing) * no_obj_embed_spatial).sum(dim=1)
    return maskmem_features + no_obj_embed[..., None, None].expand_as(maskmem_features)


def align_object_scores(object_score_logits, multiplex_state):
    if object_score_logits is None:
        return object_score_logits

    expected = multiplex_state.total_valid_entries
    current = object_score_logits.shape[0]
    if current == expected:
        return object_score_logits

    if current > expected:
        return object_score_logits[:expected]

    pad_shape = (expected - current, *object_score_logits.shape[1:])
    padding = object_score_logits.new_zeros(pad_shape)
    return torch.cat([object_score_logits, padding], dim=0)


def add_condition_embeddings(
    self,
    maskmem_features,
    unconditioning_objects,
    multiplex_state,
):
    if not self.add_object_conditional_embeddings:
        return maskmem_features

    obj_cond_embed = self.obj_cond_embed.unsqueeze(0).repeat(
        multiplex_state.num_buckets, 1, 1
    )
    obj_merged_embed = multiplex_state.demux(obj_cond_embed)

    if self.add_object_unconditional_embeddings:
        obj_non_cond_embed = self.obj_non_cond_embed.unsqueeze(0).repeat(
            multiplex_state.num_buckets, 1, 1
        )
        obj_non_cond_embed = multiplex_state.demux(obj_non_cond_embed)
        if self.training:
            obj_merged_embed = obj_merged_embed.clone()
        obj_merged_embed[unconditioning_objects] = obj_non_cond_embed[
            unconditioning_objects
        ]

    obj_merged_embed = multiplex_state.mux(obj_merged_embed).sum(dim=1)
    return maskmem_features + obj_merged_embed[..., None, None].expand_as(
        maskmem_features
    )


def demux_memory_output(maskmem_features, maskmem_pos_enc, multiplex_state):
    if maskmem_features.dim() == 5:
        maskmem_features = multiplex_state.demux(maskmem_features).contiguous()

    demuxed_pos_enc = []
    for pos_enc in maskmem_pos_enc:
        if pos_enc is not None and pos_enc.dim() == 5:
            pos_enc = multiplex_state.demux(pos_enc).contiguous()
        demuxed_pos_enc.append(pos_enc)
    return maskmem_features, demuxed_pos_enc
