import torch


def update_existing_frame_masks(
    self,
    *,
    inference_state,
    output_dict,
    frame_idx,
    is_init_cond_frame,
    mask_inputs,
    run_mem_encoder,
    backbone_features_interactive,
    backbone_features_propagation,
    reconditioning,
    new_obj_idxs,
    new_obj_ids,
    allow_new_buckets,
    prefer_new_buckets,
):
    with torch.profiler.record_function(
        "VideoTrackingMultiplexDemo.add_new_masks_to_existing_state"
    ):
        existing_out = _find_existing_output(output_dict, frame_idx)
        interactive_pix_feat, interactive_high_res_features = _get_interactive_features(
            self, backbone_features_interactive
        )
        propagation_vision_feats, propagation_feat_sizes = _get_propagation_features(
            backbone_features_propagation, run_mem_encoder
        )

        if reconditioning:
            recondition_existing_masks(
                self,
                inference_state,
                existing_out,
                interactive_pix_feat,
                interactive_high_res_features,
                propagation_vision_feats,
                propagation_feat_sizes,
                mask_inputs,
                new_obj_idxs,
                new_obj_ids,
                run_mem_encoder,
            )
            return existing_out

        add_masks_to_existing_frame(
            self,
            inference_state,
            existing_out,
            is_init_cond_frame=is_init_cond_frame,
            mask_inputs=mask_inputs,
            interactive_pix_feat=interactive_pix_feat,
            interactive_high_res_features=interactive_high_res_features,
            propagation_vision_feats=propagation_vision_feats,
            propagation_feat_sizes=propagation_feat_sizes,
            new_obj_idxs=new_obj_idxs,
            obj_ids_in_mask=new_obj_ids,
            add_mask_to_memory=run_mem_encoder,
            allow_new_buckets=allow_new_buckets,
            prefer_new_buckets=prefer_new_buckets,
        )
        return existing_out


def recondition_existing_masks(
    self,
    inference_state,
    existing_out,
    interactive_pix_feat,
    interactive_high_res_features,
    propagation_vision_feats,
    propagation_feat_sizes,
    mask_inputs,
    new_obj_idxs,
    new_obj_ids,
    run_mem_encoder,
):
    self.recondition_masks_in_existing_state(
        interactive_pix_feat=interactive_pix_feat,
        interactive_high_res_features=interactive_high_res_features,
        propagation_vision_feats=propagation_vision_feats,
        propagation_feat_sizes=propagation_feat_sizes,
        new_masks=mask_inputs,
        obj_idxs_in_mask=new_obj_idxs,
        obj_ids_in_mask=new_obj_ids,
        prev_output=existing_out,
        multiplex_state=inference_state["multiplex_state"],
        add_mask_to_memory=run_mem_encoder,
    )


def add_masks_to_existing_frame(
    self,
    inference_state,
    existing_out,
    *,
    is_init_cond_frame,
    mask_inputs,
    interactive_pix_feat,
    interactive_high_res_features,
    propagation_vision_feats,
    propagation_feat_sizes,
    new_obj_idxs,
    obj_ids_in_mask,
    add_mask_to_memory,
    allow_new_buckets,
    prefer_new_buckets,
):
    if mask_inputs is None:
        raise ValueError("mask inputs are required for object updates")
    self.add_new_masks_to_existing_state(
        interactive_pix_feat=interactive_pix_feat,
        interactive_high_res_features=interactive_high_res_features,
        propagation_vision_feats=propagation_vision_feats,
        propagation_feat_sizes=propagation_feat_sizes,
        new_masks=mask_inputs,
        obj_idxs_in_mask=new_obj_idxs,
        obj_ids_in_mask=obj_ids_in_mask,
        prev_output=existing_out,
        multiplex_state=inference_state["multiplex_state"],
        add_mask_to_memory=add_mask_to_memory,
        are_masks_from_pts=False,
        allow_new_buckets=allow_new_buckets,
        prefer_new_buckets=prefer_new_buckets,
    )


def _find_existing_output(output_dict, frame_idx):
    existing_out = output_dict["cond_frame_outputs"].get(frame_idx)
    if existing_out is None:
        existing_out = output_dict["non_cond_frame_outputs"].get(frame_idx)
    if existing_out is None:
        raise RuntimeError(
            f"No existing output found for frame {frame_idx} in either storage"
        )
    return existing_out


def _get_interactive_features(self, backbone_features_interactive):
    interactive_pix_feat = self._get_interactive_pix_mem(
        backbone_features_interactive["vision_feats"],
        backbone_features_interactive["feat_sizes"],
    )

    # Shape: (HW)BC -> BCHW.
    high_res_features = [
        x.permute(1, 2, 0).view(x.size(1), x.size(2), *size)
        for x, size in zip(
            backbone_features_interactive["vision_feats"][:-1],
            backbone_features_interactive["feat_sizes"][:-1],
        )
    ]
    return interactive_pix_feat, high_res_features


def _get_propagation_features(backbone_features_propagation, run_mem_encoder):
    if not run_mem_encoder:
        return None, None
    return (
        backbone_features_propagation["vision_feats"],
        backbone_features_propagation["feat_sizes"],
    )
