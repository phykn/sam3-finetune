from torch import nn

from ..components.video.create import create_maskmem_backbone


class VideoMem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = create_maskmem_backbone()

    def from_ckpt(self, ckpt, strict=False):
        self.encoder.load_state_dict(
            ckpt.block_state("video.maskmem_backbone"),
            strict=strict,
        )
        return self

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
