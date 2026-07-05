import torch


def make_current_out(point_inputs, mask_inputs):
    return {
        "conditioning_objects": set(),
        "point_inputs": point_inputs,
        "mask_inputs": mask_inputs,
    }


def determine_mode(
    model,
    *,
    is_init_cond_frame,
    point_inputs,
    mask_inputs,
    prev_sam_mask_logits,
    objects_to_interact,
):
    if mask_inputs is not None:
        return "mask_as_output"
    if point_inputs is None:
        return "propagation_only"
    if prev_sam_mask_logits is not None:
        assert (
            objects_to_interact is not None
        ), "objects_to_interact must be specified when refining with prev_sam_mask_logits"
        return "interaction_only"
    if is_init_cond_frame:
        return "interaction_only"
    if objects_to_interact is not None:
        assert not model.training
        return "propagation_and_interaction"

    raise ValueError(
        f"Unable to determine tracking case. "
        f"mask_inputs={mask_inputs is not None}, "
        f"point_inputs={point_inputs is not None}, "
        f"prev_sam_mask_logits={prev_sam_mask_logits is not None}, "
        f"objects_to_interact={objects_to_interact}, "
        f"is_init_cond_frame={is_init_cond_frame}"
    )


def make_high_res_features(vision_feats, feat_sizes):
    if len(vision_feats) <= 1:
        return None

    return [
        x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
        for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
    ]


def get_interactive_features(mode, backbone_features):
    if backbone_features is None:
        assert mode not in ["interaction_only", "propagation_and_interaction"]
        return None, None, None

    vision_feats = backbone_features["vision_feats"]
    feat_sizes = backbone_features["feat_sizes"]
    return vision_feats, feat_sizes, make_high_res_features(vision_feats, feat_sizes)


def get_propagation_features(mode, backbone_features, run_mem_encoder):
    if backbone_features is None:
        assert mode not in ["propagation_only", "propagation_and_interaction"]
        assert not run_mem_encoder
        return None, None, None, None, None

    vision_feats = backbone_features["vision_feats"]
    vision_masks = backbone_features["vision_masks"]
    vision_pos_embeds = backbone_features["vision_pos_embeds"]
    feat_sizes = backbone_features["feat_sizes"]
    high_res_features = make_high_res_features(vision_feats, feat_sizes)

    return (
        vision_feats,
        vision_masks,
        vision_pos_embeds,
        feat_sizes,
        high_res_features,
    )


def get_step_features(
    mode, backbone_features_interactive, backbone_features_propagation, run_mem_encoder
):
    interactive_vision_feats, interactive_feat_sizes, interactive_high_res_features = (
        get_interactive_features(mode, backbone_features_interactive)
    )
    (
        propagation_vision_feats,
        propagation_vision_masks,
        propagation_vision_pos_embeds,
        propagation_feat_sizes,
        propagation_high_res_features,
    ) = get_propagation_features(mode, backbone_features_propagation, run_mem_encoder)

    return {
        "interactive_vision_feats": interactive_vision_feats,
        "interactive_feat_sizes": interactive_feat_sizes,
        "interactive_high_res_features": interactive_high_res_features,
        "propagation_vision_feats": propagation_vision_feats,
        "propagation_vision_masks": propagation_vision_masks,
        "propagation_vision_pos_embeds": propagation_vision_pos_embeds,
        "propagation_feat_sizes": propagation_feat_sizes,
        "propagation_high_res_features": propagation_high_res_features,
    }


def run_mask_as_output(
    model,
    *,
    interactive_vision_feats,
    interactive_feat_sizes,
    interactive_high_res_features,
    mask_inputs,
    multiplex_state,
):
    assert model.use_mask_input_as_output_without_sam
    interactive_pix_feat = model._get_interactive_pix_mem(
        interactive_vision_feats, interactive_feat_sizes
    )
    sam_outputs = model._use_mask_as_output(
        backbone_features=interactive_pix_feat,
        high_res_features=interactive_high_res_features,
        mask_inputs=mask_inputs,
        multiplex_state=multiplex_state,
    )
    return interactive_pix_feat, sam_outputs


def run_propagation(
    model,
    *,
    frame_idx,
    is_init_cond_frame,
    propagation_vision_feats,
    propagation_vision_masks,
    propagation_vision_pos_embeds,
    propagation_feat_sizes,
    propagation_high_res_features,
    output_dict,
    num_frames,
    track_in_reverse,
    multiplex_state,
):
    assert propagation_vision_feats is not None
    assert propagation_vision_masks is not None
    assert propagation_vision_pos_embeds is not None
    assert propagation_feat_sizes is not None

    pix_feat_with_mem = model._prepare_memory_conditioned_features(
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        current_vision_feats=propagation_vision_feats[-1:],
        current_vision_masks=propagation_vision_masks[-1:],
        current_vision_pos_embeds=propagation_vision_pos_embeds[-1:],
        feat_sizes=propagation_feat_sizes[-1:],
        output_dict=output_dict,
        num_frames=num_frames,
        track_in_reverse=track_in_reverse,
        multiplex_state=multiplex_state,
    )

    multimask_output = model._use_multimask(is_init_cond_frame, point_inputs=None)
    return model._forward_sam_heads(
        backbone_features=pix_feat_with_mem,
        propagation_high_res_features=propagation_high_res_features,
        multimask_output=multimask_output,
        objects_to_interact=list(range(multiplex_state.total_valid_entries)),
        multiplex_state=multiplex_state,
    )


def get_interaction_mask_inputs(
    model,
    *,
    mode,
    mask_inputs,
    point_inputs,
    prev_sam_mask_logits,
    propagation_out,
    objects_to_interact,
):
    assert mask_inputs is None and point_inputs is not None
    if prev_sam_mask_logits is not None:
        assert objects_to_interact is not None
        assert model.iter_use_prev_mask_pred
        assert mode != "propagation_and_interaction"
        return prev_sam_mask_logits[objects_to_interact]

    if mode == "propagation_and_interaction":
        assert objects_to_interact is not None
        assert propagation_out is not None
        return propagation_out["low_res_masks"][objects_to_interact]

    return mask_inputs


def check_objects_to_interact(point_inputs, objects_to_interact):
    if objects_to_interact is None:
        return

    assert point_inputs["point_coords"].shape[0] == len(objects_to_interact)
    assert point_inputs["point_labels"].shape[0] == len(objects_to_interact)


def run_interaction(
    model,
    *,
    mode,
    is_init_cond_frame,
    interactive_vision_feats,
    interactive_feat_sizes,
    interactive_high_res_features,
    point_inputs,
    mask_inputs,
    prev_sam_mask_logits,
    propagation_out,
    objects_to_interact,
    multiplex_state,
):
    assert interactive_vision_feats is not None
    assert interactive_feat_sizes is not None

    interactive_pix_feat = model._get_interactive_pix_mem(
        interactive_vision_feats, interactive_feat_sizes
    )
    mask_inputs = get_interaction_mask_inputs(
        model,
        mode=mode,
        mask_inputs=mask_inputs,
        point_inputs=point_inputs,
        prev_sam_mask_logits=prev_sam_mask_logits,
        propagation_out=propagation_out,
        objects_to_interact=objects_to_interact,
    )
    check_objects_to_interact(point_inputs, objects_to_interact)

    multimask_output = model._use_multimask(
        is_init_cond_frame, point_inputs=point_inputs
    )
    interaction_out = model._forward_sam_heads(
        backbone_features=interactive_pix_feat,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        interactive_high_res_features=interactive_high_res_features,
        multimask_output=multimask_output,
        objects_to_interact=(
            objects_to_interact
            if objects_to_interact is not None
            else list(range(multiplex_state.total_valid_entries))
        ),
        multiplex_state=multiplex_state,
    )
    return interactive_pix_feat, mask_inputs, interaction_out


def run_initial_step(
    model,
    *,
    mode,
    frame_idx,
    is_init_cond_frame,
    point_inputs,
    mask_inputs,
    prev_sam_mask_logits,
    objects_to_interact,
    output_dict,
    num_frames,
    track_in_reverse,
    multiplex_state,
    features,
):
    interactive_pix_feat = None
    if mode == "mask_as_output":
        interactive_pix_feat, sam_outputs = run_mask_as_output(
            model,
            interactive_vision_feats=features["interactive_vision_feats"],
            interactive_feat_sizes=features["interactive_feat_sizes"],
            interactive_high_res_features=features["interactive_high_res_features"],
            mask_inputs=mask_inputs,
            multiplex_state=multiplex_state,
        )
        return interactive_pix_feat, sam_outputs, mask_inputs

    propagation_out = None
    if mode in ["propagation_only", "propagation_and_interaction"]:
        propagation_out = run_propagation(
            model,
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            propagation_vision_feats=features["propagation_vision_feats"],
            propagation_vision_masks=features["propagation_vision_masks"],
            propagation_vision_pos_embeds=features["propagation_vision_pos_embeds"],
            propagation_feat_sizes=features["propagation_feat_sizes"],
            propagation_high_res_features=features["propagation_high_res_features"],
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            multiplex_state=multiplex_state,
        )

    interaction_out = None
    if mode in ["interaction_only", "propagation_and_interaction"]:
        interactive_pix_feat, mask_inputs, interaction_out = run_interaction(
            model,
            mode=mode,
            is_init_cond_frame=is_init_cond_frame,
            interactive_vision_feats=features["interactive_vision_feats"],
            interactive_feat_sizes=features["interactive_feat_sizes"],
            interactive_high_res_features=features["interactive_high_res_features"],
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            prev_sam_mask_logits=prev_sam_mask_logits,
            propagation_out=propagation_out,
            objects_to_interact=objects_to_interact,
            multiplex_state=multiplex_state,
        )

    sam_outputs = choose_sam_outputs(
        propagation_out, interaction_out, objects_to_interact
    )
    return interactive_pix_feat, sam_outputs, mask_inputs


def update_conditioning_objects(
    current_out, mode, mask_inputs, objects, multiplex_state
):
    if mode == "mask_as_output":
        current_out["conditioning_objects"].update(range(mask_inputs.shape[0]))
    elif mode in ["interaction_only", "propagation_and_interaction"]:
        if objects is None:
            current_out["conditioning_objects"].update(
                multiplex_state.get_all_valid_object_idx()
            )
        else:
            current_out["conditioning_objects"].update(objects)


def merge_interaction_into_propagation(propagation_out, interaction_out, objects):
    keys_to_merge = [
        "low_res_multimasks",
        "high_res_multimasks",
        "low_res_masks",
        "high_res_masks",
        "ious",
        "object_score_logits",
        "obj_ptr",
    ]
    for key in keys_to_merge:
        src = interaction_out[key]
        dst = propagation_out[key]
        if torch.is_tensor(src) and torch.is_tensor(dst):
            if torch.is_floating_point(src) and src.dtype != dst.dtype:
                src = src.to(dtype=dst.dtype)
        propagation_out[key][objects] = src
    return propagation_out


def choose_sam_outputs(propagation_out, interaction_out, objects_to_interact):
    if propagation_out is None:
        assert interaction_out is not None
        return interaction_out
    if interaction_out is None:
        return propagation_out

    assert objects_to_interact is not None
    return merge_interaction_into_propagation(
        propagation_out,
        interaction_out,
        objects_to_interact,
    )
