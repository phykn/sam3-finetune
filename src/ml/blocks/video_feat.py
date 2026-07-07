from ..components.backbone.create import create_vision_backbone
from ..components.video.frame import VideoFeat as BaseVideoFeat


class VideoFeat(BaseVideoFeat):
    def __init__(self) -> None:
        super().__init__(vision_backbone=create_vision_backbone(), scalp=0)

    def from_ckpt(self, ckpt, strict=False):
        self.load_state_dict(ckpt.block_state("video.backbone"), strict=strict)
        return self
