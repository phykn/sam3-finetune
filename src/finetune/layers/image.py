import torch
from torch import nn


class ImageAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        rank: int,
        num_experts: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.scale = alpha / rank
        self.down = nn.ModuleList(
            nn.Conv2d(channels, rank, kernel_size=1, bias=False)
            for _ in range(num_experts)
        )
        self.up = nn.ModuleList(
            nn.Conv2d(rank, channels, kernel_size=1, bias=False)
            for _ in range(num_experts)
        )
        self._init_adapter()

    def forward(self, x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        delta = 0.0
        for index, (down, up) in enumerate(zip(self.down, self.up)):
            weight = mix[:, index].view(mix.shape[0], 1, 1, 1)
            delta = delta + up(down(x)) * weight
        return x + delta * self.scale

    def _init_adapter(self) -> None:
        for up in self.up:
            nn.init.zeros_(up.weight)

