from .frame import (
    copy_frame_output,
    find_conditioning_objects,
    find_frame_output,
    get_target,
    new_output,
)
from .memory import encode_memory, write_reconstructed_scores
from .objects import merge_per_object_outputs


def consolidate_temp_output_across_obj(
    model,
    inference_state,
    frame_idx,
    is_cond,
    run_mem_encoder,
    consolidate_at_video_res=False,
):
    batch_size = model._get_obj_num(inference_state)
    max_obj_idx = max(
        [
            batch_size - 1,
            *inference_state["temp_output_dict_per_obj"].keys(),
            *inference_state["output_dict_per_obj"].keys(),
        ]
    )
    target = get_target(
        model, inference_state, run_mem_encoder, consolidate_at_video_res
    )
    object_count = max(max_obj_idx + 1, 0)
    out = new_output(inference_state, object_count, target)

    frame_out = find_frame_output(inference_state, frame_idx)
    reconstruct_from_objects = frame_out is None
    if reconstruct_from_objects:
        out["conditioning_objects"] = find_conditioning_objects(
            inference_state, frame_idx, batch_size
        )
    else:
        copy_frame_output(model, out, frame_out, target)

    obj_scores, iou_scores = merge_per_object_outputs(
        model=model,
        inference_state=inference_state,
        out=out,
        frame_idx=frame_idx,
        storage_key="cond_frame_outputs" if is_cond else "non_cond_frame_outputs",
        target=target,
        object_count=object_count,
        reconstruct_from_objects=reconstruct_from_objects,
    )

    if reconstruct_from_objects:
        run_mem_encoder = write_reconstructed_scores(
            model=model,
            inference_state=inference_state,
            out=out,
            batch_size=batch_size,
            obj_scores=obj_scores,
            iou_scores=iou_scores,
            run_mem_encoder=run_mem_encoder,
        )

    if run_mem_encoder:
        encode_memory(model, inference_state, frame_idx, batch_size, out)

    return out
