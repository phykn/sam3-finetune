import torch

from ....io.load import load_frames
from .frame.features import get_image_feature, get_maskmem_pos_enc
from .frame.inference import run_single_frame_inference
from .frame.mask_cleanup import fill_holes_in_mask_scores
from .frame.memory import run_memory_encoder
from .interaction.masks import add_new_masks
from .interaction.objects import (
    clear_non_cond_mem_around_input,
    remove_object,
    remove_objects,
)
from .interaction.points import (
    add_new_points,
    clear_all_points_in_frame,
    clear_all_points_in_video,
)
from .interaction.propagation import (
    get_processing_order,
    propagate_in_video,
    propagate_in_video_preflight,
    propagate_sam3_in_video,
)
from .state import add_object_slot, create_inference_state
from .tracking import VideoTrackingDynamicMultiplex


class VideoTrackingMultiplexDemo(VideoTrackingDynamicMultiplex):
    def __init__(
        self,
        clear_non_cond_mem_around_input=False,
        clear_non_cond_mem_for_multi_obj=False,
        fill_hole_area=0,
        always_start_from_first_ann_frame=False,
        max_point_num_in_prompt_enc=16,
        non_overlap_masks_for_output=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.fill_hole_area = fill_hole_area
        self.always_start_from_first_ann_frame = always_start_from_first_ann_frame
        self.max_point_num_in_prompt_enc = max_point_num_in_prompt_enc
        self.non_overlap_masks_for_output = non_overlap_masks_for_output

    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu,
        offload_state_to_cpu,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
    ):
        if not self.apply_sigmoid_to_mask_logits_for_mem_enc:
            raise NotImplementedError(
                "Multi-object tracking requires sigmoid in memory encoder for non-overlapping constraints."
            )

        images, video_height, video_width = load_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
        )
        return create_inference_state(
            images=images,
            num_frames=len(images),
            video_height=video_height,
            video_width=video_width,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            track_user_refinement=True,
        )

    def _obj_id_to_idx(self, inference_state, obj_id, error_if_new=False):
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        if (
            self.is_dynamic_model or not inference_state["tracking_has_started"]
        ) and not error_if_new:
            return add_object_slot(inference_state, obj_id)
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id}. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

    def _get_obj_num(self, inference_state):
        return inference_state["multiplex_state"].total_valid_entries

    add_new_points = add_new_points
    add_new_masks = add_new_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks_for_output:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        if self.fill_hole_area > 0:
            video_res_masks = fill_holes_in_mask_scores(
                video_res_masks, self.fill_hole_area
            )
        return any_res_masks, video_res_masks

    propagate_in_video_preflight = propagate_in_video_preflight
    _get_processing_order = get_processing_order
    propagate_in_video = propagate_in_video

    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "pred_masks": current_out["pred_masks"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            if self.use_memory_selection:
                obj_out["iou_score"] = current_out["iou_score"][obj_slice]
            obj_output_dict[storage_key][frame_idx] = obj_out

    clear_all_points_in_frame = clear_all_points_in_frame
    clear_all_points_in_video = clear_all_points_in_video

    def _reset_tracking_results(self, inference_state):
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()
        inference_state["first_ann_frame_idx"] = None

    _get_image_feature = get_image_feature
    _run_single_frame_inference = run_single_frame_inference
    _run_memory_encoder = run_memory_encoder
    _get_maskmem_pos_enc = get_maskmem_pos_enc

    remove_object = remove_object
    remove_objects = remove_objects
    _clear_non_cond_mem_around_input = clear_non_cond_mem_around_input


class Sam3VideoTrackingMultiplexDemo(VideoTrackingMultiplexDemo):
    @torch.inference_mode()
    def init_state(
        self,
        video_height,
        video_width,
        num_frames,
        cached_features=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        if not self.apply_sigmoid_to_mask_logits_for_mem_enc:
            raise NotImplementedError(
                "Multi-object tracking requires sigmoid in memory encoder for non-overlapping constraints."
            )
        inference_state = create_inference_state(
            cached_features=cached_features,
            num_frames=num_frames,
            video_height=video_height,
            video_width=video_width,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        self.clear_all_points_in_video(inference_state)
        return inference_state

    propagate_in_video = propagate_sam3_in_video
