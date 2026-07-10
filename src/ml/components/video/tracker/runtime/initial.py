def make_current_out(mask_inputs):
    return {
        "conditioning_objects": set(),
        "mask_inputs": mask_inputs,
    }


def determine_mode(mask_inputs):
    return "mask_as_output" if mask_inputs is not None else "propagation"


def make_high_res_features(vision_feats, feat_sizes):
    if len(vision_feats) <= 1:
        return None
    return [
        feat.permute(1, 2, 0).view(feat.size(1), feat.size(2), *size)
        for feat, size in zip(vision_feats[:-1], feat_sizes[:-1])
    ]


def get_interactive_features(backbone_features):
    if backbone_features is None:
        return None, None, None
    vision_feats = backbone_features["vision_feats"]
    feat_sizes = backbone_features["feat_sizes"]
    return vision_feats, feat_sizes, make_high_res_features(vision_feats, feat_sizes)


def get_propagation_features(backbone_features):
    if backbone_features is None:
        return None, None, None, None, None
    vision_feats = backbone_features["vision_feats"]
    feat_sizes = backbone_features["feat_sizes"]
    return (
        vision_feats,
        backbone_features["vision_masks"],
        backbone_features["vision_pos_embeds"],
        feat_sizes,
        make_high_res_features(vision_feats, feat_sizes),
    )


def get_step_features(backbone_features_interactive, backbone_features_propagation):
    interactive_feats, interactive_sizes, interactive_high_res = (
        get_interactive_features(backbone_features_interactive)
    )
    propagation = get_propagation_features(backbone_features_propagation)
    return {
        "interactive_vision_feats": interactive_feats,
        "interactive_feat_sizes": interactive_sizes,
        "interactive_high_res_features": interactive_high_res,
        "propagation_vision_feats": propagation[0],
        "propagation_vision_masks": propagation[1],
        "propagation_vision_pos_embeds": propagation[2],
        "propagation_feat_sizes": propagation[3],
        "propagation_high_res_features": propagation[4],
    }


def run_mask_as_output(model, features, mask_inputs, multiplex_state):
    interactive_pix_feat = model._get_interactive_pix_mem(
        features["interactive_vision_feats"],
        features["interactive_feat_sizes"],
    )
    output = model._use_mask_as_output(
        backbone_features=interactive_pix_feat,
        high_res_features=features["interactive_high_res_features"],
        mask_inputs=mask_inputs,
        multiplex_state=multiplex_state,
    )
    return output


def run_propagation(
    model,
    *,
    frame_idx,
    is_init_cond_frame,
    features,
    output_dict,
    num_frames,
    multiplex_state,
):
    pix_feat_with_mem = model._prepare_memory_conditioned_features(
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        current_vision_feats=features["propagation_vision_feats"][-1:],
        current_vision_masks=features["propagation_vision_masks"][-1:],
        current_vision_pos_embeds=features["propagation_vision_pos_embeds"][-1:],
        feat_sizes=features["propagation_feat_sizes"][-1:],
        output_dict=output_dict,
        num_frames=num_frames,
        multiplex_state=multiplex_state,
    )
    return model._forward_sam_heads(
        backbone_features=pix_feat_with_mem,
        propagation_high_res_features=features["propagation_high_res_features"],
        multimask_output=model._use_multimask(is_init_cond_frame),
        objects_to_interact=list(range(multiplex_state.total_valid_entries)),
        multiplex_state=multiplex_state,
    )


def run_initial_step(
    model,
    *,
    mode,
    frame_idx,
    is_init_cond_frame,
    mask_inputs,
    output_dict,
    num_frames,
    multiplex_state,
    features,
):
    if mode == "mask_as_output":
        return run_mask_as_output(model, features, mask_inputs, multiplex_state)
    return run_propagation(
        model,
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        features=features,
        output_dict=output_dict,
        num_frames=num_frames,
        multiplex_state=multiplex_state,
    )


def update_conditioning_objects(current_out, mode, mask_inputs):
    if mode == "mask_as_output":
        current_out["conditioning_objects"].update(range(mask_inputs.shape[0]))
