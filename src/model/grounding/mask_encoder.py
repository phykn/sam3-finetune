import torch
import torch.nn as nn


class MaskEncoder(nn.Module):
    def __init__(
        self,
        mask_downsampler: nn.Module,
        position_encoding: nn.Module,
    ):
        super().__init__()
        self.mask_downsampler = mask_downsampler
        self.position_encoding = position_encoding

    def forward(self, masks, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        masks = self.mask_downsampler(masks)
        masks_pos = self.position_encoding(masks).to(masks.dtype)

        return masks, masks_pos
