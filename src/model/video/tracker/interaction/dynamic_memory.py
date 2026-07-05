def existing_obj_ptrs(self, prev_output, multiplex_state):
    if not self.use_obj_ptrs_in_encoder:
        return None
    return multiplex_state.demux(prev_output["obj_ptr"])


def encode_mask_memory(
    self,
    *,
    prev_output,
    propagation_vision_feats,
    propagation_feat_sizes,
    multiplex_state,
    add_mask_to_memory,
    are_masks_from_pts,
):
    if not add_mask_to_memory:
        return

    assert (
        prev_output["pred_masks_high_res"].shape[0]
        == multiplex_state.total_valid_entries
    )
    maskmem_features, maskmem_pos_enc = self._encode_new_memory(
        image=None,
        current_vision_feats=propagation_vision_feats,
        feat_sizes=propagation_feat_sizes,
        pred_masks_high_res=prev_output["pred_masks_high_res"],
        object_score_logits=prev_output["object_score_logits"],
        conditioning_objects=prev_output["conditioning_objects"],
        is_mask_from_pts=are_masks_from_pts,
        multiplex_state=multiplex_state,
    )
    prev_output["maskmem_features"] = maskmem_features
    prev_output["maskmem_pos_enc"] = maskmem_pos_enc
    if self.save_image_features:
        assert "image_features" in prev_output
        assert "image_pos_enc" in prev_output
