from ..multiplex.state import MultiplexState
from ..outputs import StageOutput
from .correction import apply_correction_points
from .initial import (
    determine_mode,
    get_step_features,
    make_current_out,
    run_initial_step,
    update_conditioning_objects,
)
from .output import (
    encode_memory,
    get_sam_values,
    make_aux_output,
    save_image_features,
    write_final_outputs,
    write_initial_multistep,
)


def run_track_step_aux(
    model,
    *,
    frame_idx,
    is_init_cond_frame,
    backbone_features_interactive,
    backbone_features_propagation,
    image,
    point_inputs,
    mask_inputs,
    gt_masks,
    frames_to_add_correction_pt,
    output_dict,
    num_frames,
    track_in_reverse=False,
    run_mem_encoder=True,
    prev_sam_mask_logits=None,
    multiplex_state: MultiplexState,
    objects_to_interact: list[int] | None = None,
    need_aux_output: bool = False,
) -> tuple[StageOutput, dict]:
    current_out, mode, features = prepare_step(
        model,
        is_init_cond_frame=is_init_cond_frame,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        prev_sam_mask_logits=prev_sam_mask_logits,
        objects_to_interact=objects_to_interact,
        backbone_features_interactive=backbone_features_interactive,
        backbone_features_propagation=backbone_features_propagation,
        run_mem_encoder=run_mem_encoder,
    )
    interactive_pix_feat, values, mask_inputs = run_initial_sam_step(
        model,
        mode=mode,
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        prev_sam_mask_logits=prev_sam_mask_logits,
        objects_to_interact=objects_to_interact,
        output_dict=output_dict,
        num_frames=num_frames,
        track_in_reverse=track_in_reverse,
        multiplex_state=multiplex_state,
        features=features,
    )
    write_initial_step_state(
        current_out,
        mode,
        mask_inputs,
        objects_to_interact,
        multiplex_state,
        values,
        point_inputs,
    )

    point_inputs, values = complete_track_step(
        model,
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        current_out=current_out,
        image=image,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        gt_masks=gt_masks,
        frames_to_add_correction_pt=frames_to_add_correction_pt,
        objects_to_interact=objects_to_interact,
        features=features,
        multiplex_state=multiplex_state,
        values=values,
        run_mem_encoder=run_mem_encoder,
    )
    return current_out, make_step_aux_output(
        model,
        need_aux_output=need_aux_output,
        interactive_pix_feat=interactive_pix_feat,
        features=features,
    )


def write_initial_step_state(
    current_out,
    mode,
    mask_inputs,
    objects_to_interact,
    multiplex_state,
    values,
    point_inputs,
):
    update_conditioning_objects(
        current_out, mode, mask_inputs, objects_to_interact, multiplex_state
    )
    write_initial_multistep(current_out, values, point_inputs)


def complete_track_step(
    model,
    *,
    frame_idx,
    is_init_cond_frame,
    current_out,
    image,
    point_inputs,
    mask_inputs,
    gt_masks,
    frames_to_add_correction_pt,
    objects_to_interact,
    features,
    multiplex_state,
    values,
    run_mem_encoder,
):
    point_inputs, values = apply_step_corrections(
        model,
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        current_out=current_out,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        gt_masks=gt_masks,
        frames_to_add_correction_pt=frames_to_add_correction_pt,
        objects_to_interact=objects_to_interact,
        features=features,
        multiplex_state=multiplex_state,
        values=values,
    )
    finish_track_step(
        model,
        current_out=current_out,
        image=image,
        features=features,
        point_inputs=point_inputs,
        values=values,
        run_mem_encoder=run_mem_encoder,
        multiplex_state=multiplex_state,
    )
    return point_inputs, values


def prepare_step(
    model,
    *,
    is_init_cond_frame,
    point_inputs,
    mask_inputs,
    prev_sam_mask_logits,
    objects_to_interact,
    backbone_features_interactive,
    backbone_features_propagation,
    run_mem_encoder,
):
    current_out: StageOutput = make_current_out(point_inputs, mask_inputs)
    mode = determine_mode(
        model,
        is_init_cond_frame=is_init_cond_frame,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        prev_sam_mask_logits=prev_sam_mask_logits,
        objects_to_interact=objects_to_interact,
    )
    features = get_step_features(
        mode,
        backbone_features_interactive,
        backbone_features_propagation,
        run_mem_encoder,
    )
    return current_out, mode, features


def run_initial_sam_step(
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
    interactive_pix_feat, sam_outputs, mask_inputs = run_initial_step(
        model,
        mode=mode,
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        prev_sam_mask_logits=prev_sam_mask_logits,
        objects_to_interact=objects_to_interact,
        output_dict=output_dict,
        num_frames=num_frames,
        track_in_reverse=track_in_reverse,
        multiplex_state=multiplex_state,
        features=features,
    )
    return interactive_pix_feat, get_sam_values(model, sam_outputs), mask_inputs


def apply_step_corrections(
    model,
    *,
    frame_idx,
    is_init_cond_frame,
    current_out,
    point_inputs,
    mask_inputs,
    gt_masks,
    frames_to_add_correction_pt,
    objects_to_interact,
    features,
    multiplex_state,
    values,
):
    point_inputs, _mask_inputs, _objects_to_interact, values = apply_correction_points(
        model,
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        current_out=current_out,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        gt_masks=gt_masks,
        frames_to_add_correction_pt=frames_to_add_correction_pt,
        objects_to_interact=objects_to_interact,
        interactive_vision_feats=features["interactive_vision_feats"],
        interactive_feat_sizes=features["interactive_feat_sizes"],
        interactive_high_res_features=features["interactive_high_res_features"],
        propagation_high_res_features=features["propagation_high_res_features"],
        multiplex_state=multiplex_state,
        values=values,
    )
    return point_inputs, values


def finish_track_step(
    model,
    *,
    current_out,
    image,
    features,
    point_inputs,
    values,
    run_mem_encoder,
    multiplex_state,
):
    write_final_outputs(model, current_out, values, multiplex_state)
    encode_memory(
        model,
        current_out=current_out,
        image=image,
        propagation_vision_feats=features["propagation_vision_feats"],
        propagation_feat_sizes=features["propagation_feat_sizes"],
        point_inputs=point_inputs,
        values=values,
        run_mem_encoder=run_mem_encoder,
        multiplex_state=multiplex_state,
    )
    save_image_features(
        model,
        current_out,
        features["propagation_vision_feats"],
        features["propagation_vision_pos_embeds"],
    )


def make_step_aux_output(
    model,
    *,
    need_aux_output,
    interactive_pix_feat,
    features,
):
    return make_aux_output(
        model,
        need_aux_output=need_aux_output,
        interactive_pix_feat=interactive_pix_feat,
        interactive_vision_feats=features["interactive_vision_feats"],
        interactive_feat_sizes=features["interactive_feat_sizes"],
        interactive_high_res_features=features["interactive_high_res_features"],
        propagation_vision_feats=features["propagation_vision_feats"],
        propagation_feat_sizes=features["propagation_feat_sizes"],
    )
