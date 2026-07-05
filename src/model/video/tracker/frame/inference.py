from typing import Optional

import torch

from .mask_update import update_existing_frame_masks
from .output import compact_frame_output


def run_single_frame_inference(
    self,
    inference_state,
    output_dict,
    frame_idx,
    batch_size,
    is_init_cond_frame,
    point_inputs,
    mask_inputs,
    reverse,
    run_mem_encoder,
    prev_sam_mask_logits=None,
    add_to_existing_state: bool = False,
    new_obj_idxs: Optional[list[int]] = None,
    new_obj_ids: Optional[list[int]] = None,
    allow_new_buckets: bool = False,
    prefer_new_buckets: bool = False,
    reconditioning: bool = False,
    objects_to_interact: Optional[list[int]] = None,
):
    with torch.profiler.record_function(
        "VideoTrackingMultiplexDemo._get_image_feature"
    ):
        image, backbone_features = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )

    if add_to_existing_state or reconditioning:
        assert new_obj_idxs is not None
        assert new_obj_ids is not None

    backbone_features_interactive = backbone_features["interactive"]
    backbone_features_propagation = backbone_features["sam2_backbone_out"]

    if add_to_existing_state or reconditioning:
        current_out = update_existing_frame(
            self,
            inference_state,
            output_dict,
            frame_idx,
            is_init_cond_frame,
            point_inputs,
            mask_inputs,
            run_mem_encoder,
            backbone_features_interactive,
            backbone_features_propagation,
            reconditioning,
            new_obj_idxs,
            new_obj_ids,
            allow_new_buckets,
            prefer_new_buckets,
        )
    else:
        current_out = track_new_frame(
            self,
            inference_state,
            output_dict,
            frame_idx,
            is_init_cond_frame,
            point_inputs,
            mask_inputs,
            image,
            backbone_features_interactive,
            backbone_features_propagation,
            reverse,
            run_mem_encoder,
            prev_sam_mask_logits,
            objects_to_interact,
        )

    return compact_frame_output(self, inference_state, current_out)


def update_existing_frame(
    self,
    inference_state,
    output_dict,
    frame_idx,
    is_init_cond_frame,
    point_inputs,
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
    return update_existing_frame_masks(
        self,
        inference_state=inference_state,
        output_dict=output_dict,
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        run_mem_encoder=run_mem_encoder,
        backbone_features_interactive=backbone_features_interactive,
        backbone_features_propagation=backbone_features_propagation,
        reconditioning=reconditioning,
        new_obj_idxs=new_obj_idxs,
        new_obj_ids=new_obj_ids,
        allow_new_buckets=allow_new_buckets,
        prefer_new_buckets=prefer_new_buckets,
    )


def track_new_frame(
    self,
    inference_state,
    output_dict,
    frame_idx,
    is_init_cond_frame,
    point_inputs,
    mask_inputs,
    image,
    backbone_features_interactive,
    backbone_features_propagation,
    reverse,
    run_mem_encoder,
    prev_sam_mask_logits,
    objects_to_interact,
):
    assert point_inputs is None or mask_inputs is None
    with torch.profiler.record_function("VideoTrackingMultiplexDemo.track_step"):
        return self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            backbone_features_interactive=backbone_features_interactive,
            backbone_features_propagation=backbone_features_propagation,
            image=image,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            gt_masks=None,
            frames_to_add_correction_pt=[],
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            multiplex_state=inference_state["multiplex_state"],
            objects_to_interact=objects_to_interact,
        )
