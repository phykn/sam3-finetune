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
