import torch
from torch import nn


class SamImage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_s0 = nn.Conv2d(256, 32, 1)
        self.proj_s1 = nn.Conv2d(256, 64, 1)
        self.no_mem = nn.Parameter(torch.empty(1, 1, 256))

    def from_ckpt(self, ckpt, strict=False):
        self.load_state_dict(ckpt.block_state("image.sam_image"), strict=strict)
        return self

    def forward(self, features):
        fpn = features["backbone_fpn"]
        high_res = (
            self.proj_s0(self.tensor(fpn[0])),
            self.proj_s1(self.tensor(fpn[1])),
        )
        image_embed = self.tensor(fpn[-1])
        image_embed = image_embed + self.no_mem.view(1, -1, 1, 1).to(image_embed)

        return image_embed, high_res

    @staticmethod
    def tensor(x) -> torch.Tensor:
        return getattr(x, "tensors", x)
