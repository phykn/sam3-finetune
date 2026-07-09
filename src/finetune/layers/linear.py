import math

import torch
from torch import nn


class LoraLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        num_experts: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.num_experts = num_experts
        self.scale = alpha / rank
        self.down = nn.ModuleList(
            nn.Linear(base.in_features, rank, bias=False) for _ in range(num_experts)
        )
        self.up = nn.ModuleList(
            nn.Linear(rank, base.out_features, bias=False) for _ in range(num_experts)
        )
        self._freeze_base()
        self._init_adapter()

    def forward(
        self,
        x: torch.Tensor,
        mix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.base(x)
        if mix is None:
            return out
        delta = 0.0
        for index, (down, up) in enumerate(zip(self.down, self.up)):
            weight = self._mix_weight(mix[:, index], out)
            delta = delta + up(down(x)) * weight
        return out + delta * self.scale

    def adapter_parameters(self) -> list[nn.Parameter]:
        return list(self.down.parameters()) + list(self.up.parameters())

    def _freeze_base(self) -> None:
        for param in self.base.parameters():
            param.requires_grad = False

    def _init_adapter(self) -> None:
        for down, up in zip(self.down, self.up):
            nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
            nn.init.zeros_(up.weight)

    def _mix_weight(self, mix: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        shape = [mix.shape[0]] + [1] * (out.ndim - 1)
        return mix.view(shape)
