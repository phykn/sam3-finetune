from torch import nn

from ...components.nn.position import PositionEmbeddingSine
from ...components.video.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)


def _make_memory_encoder(multiplex_count: int = 16):
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3,
        stride=2,
        padding=1,
        interpol_size=[1152, 1152],
        multiplex_count=multiplex_count,
        starting_out_chan=4,
        input_channel_multiplier=2,
    )
    fuser = SimpleFuser(
        layer=CXBlock(
            dim=256,
            kernel_size=7,
            padding=3,
            layer_scale_init_value=1.0e-06,
            use_dwconv=True,
        ),
        num_layers=2,
    )
    return SimpleMaskEncoder(
        out_dim=256,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


class VideoMemory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _make_memory_encoder()

    def forward(self, frame, reference_mask, obj_id: int | None = None):
        mask = self.memory_mask(reference_mask)
        out = self.encoder(self.pix(frame), mask, skip_mask_sigmoid=True)
        return {
            "video_memory": out["vision_features"],
            "object_memory": out["vision_features"],
            "memory_pos": tuple(out["vision_pos_enc"]),
            "object_state": {"obj_id": obj_id} if obj_id is not None else None,
            "raw": out,
        }

    @staticmethod
    def pix(frame):
        feature = frame["vision_features"]
        if feature.dim() == 4:
            return feature
        height, width = frame["feat_sizes"][-1]
        batch_size = feature.shape[1]
        channels = feature.shape[2]
        return feature.permute(1, 2, 0).view(batch_size, channels, height, width)

    @staticmethod
    def memory_mask(mask):
        if mask.dim() == 2:
            mask = mask[None]
        if mask.dim() == 3:
            mask = mask[:, None]
        if mask.dim() != 4:
            raise ValueError("video memory mask must have shape HxW, NxHxW, or Nx1xHxW")

        mask = mask.float()
        if mask.shape[1] == 32:
            return mask
        if mask.shape[1] != 1:
            raise ValueError("video memory mask must have 1 or 32 channels")

        batch, _, height, width = mask.shape
        out = mask.new_zeros(batch, 32, height, width)
        logits = mask
        if logits.min() >= 0 and logits.max() <= 1:
            logits = logits.mul(2).sub(1)
        out[:, 0:1] = logits
        out[:, 16:17] = 1
        return out
