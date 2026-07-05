import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath
except ModuleNotFoundError:
    from timm.models.layers import DropPath

from ....components.nn.layers import clone_modules, LayerNorm2d


class SimpleMaskDownSampler(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
        interpol_size=None,
        multiplex_count: int = 1,
        starting_out_chan: int = 1,
        input_channel_multiplier: int = 1,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        multiplex_count *= input_channel_multiplier
        assert stride**num_layers == total_stride

        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = multiplex_count, starting_out_chan
        for _ in range(num_layers):
            mask_out_chans = mask_out_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))
        self.multiplex_count = multiplex_count
        self.interpol_size = interpol_size

        if self.interpol_size is not None:
            assert isinstance(
                self.interpol_size, (list, tuple)
            ), f"Unsupported type {type(self.interpol_size)}. Should be a list or tuple."
            self.interpol_size = list(interpol_size)
            assert len(self.interpol_size) == 2

    def forward(self, x: torch.Tensor):
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            x = F.interpolate(
                x.float(),
                size=self.interpol_size,
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
        return self.encoder(x)


class CXBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)

        # Shape: NCHW -> NHWC for the linear channel projections.
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        # Shape: NHWC -> NCHW for convolutional memory features.
        x = x.permute(0, 3, 1, 2)

        x = residual + self.drop_path(x)
        return x


class SimpleFuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = clone_modules(layer, num_layers)

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class SimpleMaskEncoder(nn.Module):
    def __init__(
        self,
        out_dim,
        mask_downsampler,
        fuser,
        position_encoding,
        in_dim=256,
    ):
        super().__init__()

        self.mask_downsampler = mask_downsampler

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        pix_feat = pix_feat.to(masks.device)
        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}
