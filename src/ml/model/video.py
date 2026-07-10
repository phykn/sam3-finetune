import torch
from torch import nn

from ...io.checkpoint import Checkpoint
from ..blocks.video_feat import VideoFeat
from ..blocks.video_mem import VideoMem
from ..blocks.video_track import VideoTrack
from ..components.video.tracking_model import create_tracking_model
from ..structures import NestedTensor


class Sam3VideoModel(nn.Module):
    def __init__(self, path=None) -> None:
        super().__init__()
        self.video_feat = VideoFeat()
        self.video_mem = VideoMem()
        self.video_track = VideoTrack()
        self.runtime = create_tracking_model(
            backbone=self.video_feat,
            maskmem_backbone=self.video_mem.encoder,
            transformer=self.video_track.transformer,
            image_pe=self.video_track.image_pe,
            mask_decoder=self.video_track.mask_decoder,
            output_valid_embed=self.video_track.output_valid_embed,
            output_invalid_embed=self.video_track.output_invalid_embed,
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

    def add_new_masks(self, *args, **kwargs):
        return self.runtime.add_new_masks(*args, **kwargs)

    def propagate_in_video_preflight(self, *args, **kwargs):
        return self.runtime.propagate_in_video_preflight(*args, **kwargs)

    def propagate_in_video(self, *args, **kwargs):
        return self.runtime.propagate_in_video(*args, **kwargs)

    def forward_image(self, image, *args, **kwargs):
        if isinstance(image, torch.Tensor):
            image = NestedTensor(image, None)
        return self.runtime.forward_image(image, *args, **kwargs)

    def forward(
        self,
        *args,
        **kwargs,
    ):
        return self.runtime(*args, **kwargs)
