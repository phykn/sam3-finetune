import torch
from torch import nn

from ....io.checkpoint import Checkpoint
from ...structures import NestedTensor
from .runtime import create_runtime


class Sam3VideoModel(nn.Module):
    def __init__(self, path=None) -> None:
        super().__init__()
        self.runtime = create_runtime()
        if path is not None:
            self.load_weights(Checkpoint.load(path))

    def load_weights(self, ckpt):
        ckpt.load_block("video", self.runtime)
        return self

    @property
    def image_size(self):
        return self.runtime.image_size

    def init_state(
        self,
        video_height,
        video_width,
        num_frames,
        cached_features=None,
        device="cuda",
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        return self.runtime.init_state(
            video_height=video_height,
            video_width=video_width,
            num_frames=num_frames,
            cached_features=cached_features,
            device=device,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )

    def add_masks(
        self,
        state,
        frame_idx,
        obj_ids,
        masks,
        add_mask_to_memory=False,
        reconditioning=False,
    ):
        return self.runtime.add_masks(
            state,
            frame_idx,
            obj_ids,
            masks,
            add_mask_to_memory=add_mask_to_memory,
            reconditioning=reconditioning,
        )

    def remove_objects(
        self,
        state,
        obj_ids,
        strict=False,
        need_output=True,
        clear_user_refined_map=True,
    ):
        return self.runtime.remove_objects(
            state,
            obj_ids,
            strict=strict,
            need_output=need_output,
            clear_user_refined_map=clear_user_refined_map,
        )

    def propagate_in_video_preflight(self, state, run_mem_encoder=True):
        return self.runtime.propagate_in_video_preflight(
            state,
            run_mem_encoder=run_mem_encoder,
        )

    def propagate_in_video(
        self,
        state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        tqdm_disable=False,
        run_mem_encoder=True,
    ):
        return self.runtime.propagate_in_video(
            state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            tqdm_disable=tqdm_disable,
            run_mem_encoder=run_mem_encoder,
        )

    def forward_image(
        self,
        image,
        *,
        need_sam3_out=True,
        need_interactive_out=True,
        need_propagation_out=True,
    ):
        if isinstance(image, torch.Tensor):
            image = NestedTensor(image, None)
        return self.runtime.forward_image(
            image,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )
