import torch


class VideoMemoryAdapter:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
    ) -> None:
        self.model = model
        self.device = torch.device(device)

    def run_single_frame_with_memory(
        self,
        *,
        inference_state: dict,
        frame_idx: int,
        obj_idx: int,
        point_inputs: dict[str, torch.Tensor],
    ) -> tuple[dict, torch.Tensor]:
        current_out, pred_masks = self.model._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=inference_state["output_dict"],
            frame_idx=frame_idx,
            batch_size=self.model._get_obj_num(inference_state),
            is_init_cond_frame=False,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=False,
            run_mem_encoder=False,
            objects_to_interact=[obj_idx],
        )
        _low_res_masks, video_res_masks = self.model._get_orig_video_res_output(
            inference_state,
            pred_masks,
        )
        return current_out, video_res_masks
