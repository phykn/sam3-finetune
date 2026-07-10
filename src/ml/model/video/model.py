import torch
from torch import nn

from ....io.checkpoint import Checkpoint
from ...blocks.video_feat import VideoFeat
from ...blocks.video_mem import VideoMem
from ...blocks.video_track import VideoTrack
from ...structures import NestedTensor
from .runtime import create_runtime


class Sam3VideoModel(nn.Module):
    def __init__(self, path=None) -> None:
        super().__init__()
        self.video_feat = VideoFeat()
        self.video_mem = VideoMem()
        self.video_track = VideoTrack()
        self.runtime = create_runtime(
            self.video_feat,
            self.video_track.transformer,
            self.video_mem.encoder,
        )
        self.runtime.image_pe_layer = self.video_track.image_pe
        self.runtime.sam_mask_decoder = self.video_track.mask_decoder
        self.runtime.output_valid_embed = self.video_track.output_valid_embed
        self.runtime.output_invalid_embed = self.video_track.output_invalid_embed
        if path is not None:
            self.load_weights(Checkpoint.load(path))

    def load_weights(self, ckpt):
        ckpt.load_block("video", self.runtime)
        return self

    @property
    def image_size(self):
        return self.runtime.image_size

    def init_state(self, *args, **kwargs):
        return self.runtime.init_state(*args, **kwargs)

    def add_masks(self, *args, **kwargs):
        return self.runtime.add_masks(*args, **kwargs)

    def remove_objects(self, *args, **kwargs):
        return self.runtime.remove_objects(*args, **kwargs)

    def propagate_in_video_preflight(self, *args, **kwargs):
        return self.runtime.propagate_in_video_preflight(*args, **kwargs)

    def propagate_in_video(self, *args, **kwargs):
        return self.runtime.propagate_in_video(*args, **kwargs)

    def forward_image(self, image, *args, **kwargs):
        if isinstance(image, torch.Tensor):
            image = NestedTensor(image, None)
        return self.runtime.forward_image(image, *args, **kwargs)
