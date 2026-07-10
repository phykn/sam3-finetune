import math

import torch
from torch import nn


class FeatureAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        rank: int,
        num_experts: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if rank <= 0 or num_experts <= 0:
            raise ValueError("rank and num_experts must be positive")
        self.scale = alpha / rank
        self.down = nn.ModuleList(
            nn.Conv2d(channels, rank, kernel_size=1, bias=False)
            for _ in range(num_experts)
        )
        self.up = nn.ModuleList(
            nn.Conv2d(rank, channels, kernel_size=1, bias=False)
            for _ in range(num_experts)
        )
        for up in self.up:
            nn.init.zeros_(up.weight)

    def forward(self, x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(x)
        for index, (down, up) in enumerate(zip(self.down, self.up)):
            weight = mix[:, index, None, None, None]
            delta = delta + up(down(x)) * weight
        return x + delta * self.scale


class LoraLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        num_experts: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if rank <= 0 or num_experts <= 0:
            raise ValueError("rank and num_experts must be positive")
        self.base = base
        self.scale = alpha / rank
        self.down = nn.ModuleList(
            nn.Linear(base.in_features, rank, bias=False) for _ in range(num_experts)
        )
        self.up = nn.ModuleList(
            nn.Linear(rank, base.out_features, bias=False) for _ in range(num_experts)
        )
        for param in base.parameters():
            param.requires_grad = False
        for down, up in zip(self.down, self.up):
            nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
            nn.init.zeros_(up.weight)

    def forward(
        self,
        x: torch.Tensor,
        mix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.base(x)
        if mix is None:
            return out

        delta = torch.zeros_like(out)
        shape = [mix.shape[0]] + [1] * (out.ndim - 1)
        for index, (down, up) in enumerate(zip(self.down, self.up)):
            weight = mix[:, index].view(shape)
            delta = delta + up(down(x)) * weight
        return out + delta * self.scale

    def adapter_parameters(self) -> list[nn.Parameter]:
        return list(self.down.parameters()) + list(self.up.parameters())
