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
    save_image_features,
    write_final_outputs,
    write_initial_multistep,
)


def run_track_step(
    model,
    *,
    frame_idx,
    is_init_cond_frame,
    backbone_features_interactive,
    backbone_features_propagation,
    image,
    mask_inputs,
    output_dict,
    num_frames,
    run_mem_encoder,
    multiplex_state,
):
    mode = determine_mode(mask_inputs)
    current_out = make_current_out(mask_inputs)
    features = get_step_features(
        backbone_features_interactive,
        backbone_features_propagation,
    )
    output = run_initial_step(
        model,
        mode=mode,
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        mask_inputs=mask_inputs,
        output_dict=output_dict,
        num_frames=num_frames,
        multiplex_state=multiplex_state,
        features=features,
    )
    values = get_sam_values(model, output)
    update_conditioning_objects(current_out, mode, mask_inputs)
    write_initial_multistep(current_out, values)
    write_final_outputs(model, current_out, values, multiplex_state)
    encode_memory(
        model,
        current_out=current_out,
        image=image,
        features=features,
        values=values,
        run_mem_encoder=run_mem_encoder,
        multiplex_state=multiplex_state,
    )
    save_image_features(model, current_out, features)
    return current_out
