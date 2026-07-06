import torch
from timm.layers import trunc_normal_
from torch import nn

from ..components.video.create import create_maskmem_backbone


def init_memory_encoder(
    self,
    maskmem_backbone,
    num_maskmem,
    sincos_tpos_enc,
    use_maskmem_tpos_v2,
    directly_add_no_mem_embed,
):
    self.maskmem_backbone = maskmem_backbone
    self.mem_dim = self.hidden_dim
    if hasattr(self.maskmem_backbone, "out_proj") and hasattr(
        self.maskmem_backbone.out_proj, "weight"
    ):
        mem_dim = self.maskmem_backbone.out_proj.weight.shape[0]
        assert (
            mem_dim == self.hidden_dim
        ), "there should be no compression of memory embeddings"

    self.num_maskmem = num_maskmem
    self.sincos_tpos_enc = sincos_tpos_enc
    self.use_maskmem_tpos_v2 = use_maskmem_tpos_v2
    self.maskmem_tpos_enc = nn.Parameter(torch.zeros(num_maskmem, 1, 1, self.mem_dim))
    trunc_normal_(self.maskmem_tpos_enc, std=0.02)

    self.interactivity_no_mem_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
    trunc_normal_(self.interactivity_no_mem_embed, std=0.02)
    self.directly_add_no_mem_embed = directly_add_no_mem_embed


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
        out = self.encoder(self.pix(frame), reference_mask, skip_mask_sigmoid=True)
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
