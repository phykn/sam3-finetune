from .mask_update import update_existing_frame_masks
from .output import compact_frame_output


def run_single_frame_inference(
    self,
    inference_state,
    output_dict,
    frame_idx,
    batch_size,
    is_init_cond_frame,
    mask_inputs,
    run_mem_encoder,
    add_to_existing_state=False,
    new_obj_idxs=None,
    new_obj_ids=None,
    allow_new_buckets=False,
    prefer_new_buckets=False,
    reconditioning=False,
):
    image, backbone_features = self._get_image_feature(
        inference_state,
        frame_idx,
        batch_size,
    )
    interactive = backbone_features["interactive"]
    propagation = backbone_features["sam2_backbone_out"]

    if add_to_existing_state or reconditioning:
        if new_obj_idxs is None or new_obj_ids is None:
            raise RuntimeError("existing-frame updates require object ids and indices")
        current_out = update_existing_frame_masks(
            self,
            inference_state=inference_state,
            output_dict=output_dict,
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            mask_inputs=mask_inputs,
            run_mem_encoder=run_mem_encoder,
            backbone_features_interactive=interactive,
            backbone_features_propagation=propagation,
            reconditioning=reconditioning,
            new_obj_idxs=new_obj_idxs,
            new_obj_ids=new_obj_ids,
            allow_new_buckets=allow_new_buckets,
            prefer_new_buckets=prefer_new_buckets,
        )
    else:
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            backbone_features_interactive=interactive,
            backbone_features_propagation=propagation,
            image=image,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            run_mem_encoder=run_mem_encoder,
            multiplex_state=inference_state["multiplex_state"],
        )
    return compact_frame_output(self, inference_state, current_out)
