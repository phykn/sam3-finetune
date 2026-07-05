import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_upsampling_stages,
        interpolation_mode="nearest",
        shared_conv=False,
        compile_mode=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_upsampling_stages = num_upsampling_stages
        self.interpolation_mode = interpolation_mode

        num_convs = 1 if shared_conv else num_upsampling_stages
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1) for _ in range(num_convs)]
        )
        self.norms = nn.ModuleList(
            [nn.GroupNorm(8, hidden_dim) for _ in range(num_convs)]
        )
        self.shared_conv = shared_conv
        self.out_dim = self.conv_layers[-1].out_channels

        if compile_mode is not None:
            self.forward = torch.compile(
                self.forward, mode=compile_mode, dynamic=True, fullgraph=True
            )
            # Checkpointed modules can trip DDP graph optimization.
            torch._dynamo.config.optimize_ddp = False

    def forward(self, backbone_feats: list[torch.Tensor]):
        prev_fpn = backbone_feats[-1]
        fpn_feats = backbone_feats[:-1]

        for layer_idx, feat in enumerate(fpn_feats[::-1]):
            prev_fpn = feat + F.interpolate(
                prev_fpn,
                size=feat.shape[-2:],
                mode=self.interpolation_mode,
            )
            if self.shared_conv:
                layer_idx = 0
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = F.relu(self.norms[layer_idx](prev_fpn))

        return prev_fpn
