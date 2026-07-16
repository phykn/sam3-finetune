import torch
from torch import nn

from ....io.checkpoint import Checkpoint
from ...blocks.video.features import VideoFeatures
from ...blocks.video.memory import VideoMemory
from ...blocks.video.tracking import VideoTracking
from ...structures import NestedTensor
from .runtime import create_runtime


class Sam3VideoModel(nn.Module):
    def __init__(self, path=None) -> None:
        super().__init__()
        self.video_feat = VideoFeatures()
        self.video_mem = VideoMemory()
        self.video_track = VideoTracking()
        self.runtime = create_runtime(
            self.video_feat,
            self.video_mem,
            self.video_track,
        )
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
