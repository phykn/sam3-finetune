import torch
from torch import nn

from ...components.sam.prompt_encoder import PositionEmbeddingRandom
from ...components.sam.transformer import TwoWayTransformer
from ...components.transformer.model import Transformer
from ...components.transformer.video import (
    RotaryAttention,
    VideoDecoderLayer,
    VideoTransformerEncoder,
)
from ...components.video.multiplex import MultiplexMaskDecoder

NUM_MULTIMASK_OUTPUTS = 3


def _make_transformer(use_rope_real: bool = False):
    self_attn = RotaryAttention(
        d_model=256,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_rope_real=use_rope_real,
    )
    cross_attn = RotaryAttention(
        d_model=256,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_rope_real=use_rope_real,
    )
    layer = VideoDecoderLayer(
        activation="gelu",
        d_model=256,
        num_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pos_enc_at_attn=False,
        pre_norm=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention_rope=self_attn,
        cross_attention_rope=cross_attn,
    )
    encoder = VideoTransformerEncoder(
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        use_image_in_output=False,
        layer=layer,
        num_layers=4,
        use_act_checkpoint=False,
        batch_first=True,
    )
    return Transformer(encoder=encoder, decoder=None, d_model=256)


def make_two_way_transformer(embed_dim):
    return TwoWayTransformer(
        depth=2,
        embedding_dim=embed_dim,
        mlp_dim=2048,
        num_heads=8,
    )


class VideoTracking(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _make_transformer()
        self.image_pe = PositionEmbeddingRandom(128)
        self.output_valid_embed = nn.Parameter(torch.zeros(16, 256))
        self.output_invalid_embed = nn.Parameter(torch.zeros(16, 256))
        self.mask_decoder = MultiplexMaskDecoder(
            multiplex_count=16,
            num_multimask_outputs=NUM_MULTIMASK_OUTPUTS,
            transformer=make_two_way_transformer(256),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=False,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
            multimask_outputs_only=True,
        )

    def forward(self, frame, memory, multimask=True) -> dict[str, object]:
        encoded = self.encode(frame, memory)
        masks = self.mask_decoder(
            image_embeddings=encoded,
            image_pe=self.image_pe(encoded.shape[-2:]).unsqueeze(0).to(encoded),
            high_res_features=self.high_res(frame),
            multimask_output=multimask,
            extra_per_object_embeddings=self.output_embed(encoded),
        )
        return {
            "propagated_mask_logits": masks["masks"],
            "obj_scores": masks["object_score_logits"],
            "raw": masks,
        }

    def encode(self, frame, memory):
        current = self.seq(frame["vision_features"])
        current_pos = self.seq(frame["vision_pos_enc"][-1])
        mem = self.seq(memory["video_memory"])
        mem_pos = self.seq(memory["memory_pos"][-1])
        out = self.transformer.encoder(
            image=current,
            src=current,
            memory_image=mem,
            memory=mem,
            image_pos=current_pos,
            src_pos=current_pos,
            memory_image_pos=mem_pos,
            memory_pos=mem_pos,
            num_obj_ptr_tokens=0,
        )
        height, width = frame["feat_sizes"][-1]
        return out["memory"].permute(1, 2, 0).view(current.shape[1], 256, height, width)

    def output_embed(self, encoded):
        valid = torch.zeros(
            encoded.shape[0],
            16,
            1,
            device=encoded.device,
            dtype=encoded.dtype,
        )
        valid[:, 0] = 1
        good = self.output_valid_embed.to(device=encoded.device, dtype=encoded.dtype)
        bad = self.output_invalid_embed.to(device=encoded.device, dtype=encoded.dtype)
        return valid * good.unsqueeze(0) + (1 - valid) * bad.unsqueeze(0)

    def high_res(self, frame):
        fpn = frame["backbone_fpn"]
        return [
            self.mask_decoder.conv_s0(self.tensor(fpn[0])),
            self.mask_decoder.conv_s1(self.tensor(fpn[1])),
        ]

    @staticmethod
    def seq(value):
        tensor = VideoTracking.tensor(value)
        if tensor.dim() == 4:
            return tensor.flatten(2).permute(2, 0, 1)
        return tensor

    @staticmethod
    def tensor(value):
        return getattr(value, "tensors", value)
