import torch
from torch import nn

from ..components.sam.mask_decoder import MaskDecoder
from ..components.sam.prompt_encoder import PositionEmbeddingRandom, PromptEncoder
from ..components.video.create import create_transformer, make_two_way_transformer
from ..components.video.mlp import MLP
from ..components.video.multiplex import MultiplexMaskDecoder


def init_backbone_and_transformer(
    self,
    backbone,
    transformer,
    use_high_res_features_in_sam,
    use_obj_ptrs_in_encoder,
    max_obj_ptrs_in_encoder,
    multiplex_controller,
    save_image_features,
):
    self.backbone = backbone
    self.use_high_res_features_in_sam = use_high_res_features_in_sam
    self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
    self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
    self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
    if use_obj_ptrs_in_encoder:
        self.interactive_mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)

    self.multiplex_controller = multiplex_controller
    self.save_image_features = save_image_features
    self.multiplex_count = self.multiplex_controller.multiplex_count

    assert transformer.decoder is None, "transformer should be encoder-only"
    self.transformer = transformer
    self.hidden_dim = transformer.d_model


def build_sam_heads(self):
    self.sam_prompt_embed_dim = self.hidden_dim
    self.sam_image_embedding_size = self.image_size // self.backbone_stride
    build_prompt_encoder(self)
    build_mask_decoders(self)
    build_object_pointer_projection(self)


def build_prompt_encoder(self):
    self.image_pe_layer = PositionEmbeddingRandom(self.hidden_dim // 2)
    self.interactive_sam_prompt_encoder = PromptEncoder(
        embed_dim=self.sam_prompt_embed_dim,
        image_embedding_size=(
            self.sam_image_embedding_size,
            self.sam_image_embedding_size,
        ),
        input_image_size=(self.image_size, self.image_size),
        mask_in_chans=16,
    )


def build_mask_decoders(self):
    self.interactive_sam_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=make_two_way_transformer(self.sam_prompt_embed_dim),
        transformer_dim=self.sam_prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=self.use_high_res_features_in_sam,
        iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
        pred_obj_scores=self.pred_obj_scores,
        pred_obj_scores_mlp=self.pred_obj_scores_mlp,
        use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
        **(self.interactive_sam_mask_decoder_extra_args or {}),
    )
    if self.share_necks:
        del self.interactive_sam_mask_decoder.conv_s0
        del self.interactive_sam_mask_decoder.conv_s1

    self.sam_mask_decoder = MultiplexMaskDecoder(
        multiplex_count=self.multiplex_count,
        num_multimask_outputs=self.num_multimask_outputs,
        transformer=make_two_way_transformer(self.hidden_dim),
        transformer_dim=self.hidden_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=self.use_high_res_features_in_sam,
        iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
        pred_obj_scores=self.pred_obj_scores,
        pred_obj_scores_mlp=self.pred_obj_scores_mlp,
        use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
        decode_mask_with_shared_tokens=self.decode_mask_with_shared_tokens,
        decode_mask_attribute_with_shared_tokens=(
            self.decode_mask_attribute_with_shared_tokens
        ),
        multimask_outputs_only=(
            self.num_multimask_outputs > 0 and self.multimask_output_in_sam
        ),
        **(self.sam_mask_decoder_extra_args or {}),
    )


def build_object_pointer_projection(self):
    if self.use_obj_ptrs_in_encoder:
        self.obj_ptr_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.interactive_obj_ptr_proj = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )
        if self.use_mlp_for_obj_ptr_proj:
            self.obj_ptr_proj = MLP(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
            )
            self.interactive_obj_ptr_proj = MLP(
                self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
                3,
            )
    else:
        self.obj_ptr_proj = nn.Identity()
        self.interactive_obj_ptr_proj = nn.Identity()

    if self.proj_tpos_enc_in_obj_ptrs:
        self.obj_ptr_tpos_proj = nn.Linear(self.hidden_dim, self.mem_dim)
    else:
        self.obj_ptr_tpos_proj = nn.Identity()


def get_propagation_dense_pe(self) -> torch.Tensor:
    return self.image_pe_layer(
        (self.sam_image_embedding_size, self.sam_image_embedding_size)
    ).unsqueeze(0)


class VideoTrack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = create_transformer()
        self.image_pe = PositionEmbeddingRandom(128)
        self.mask_decoder = MultiplexMaskDecoder(
            multiplex_count=16,
            num_multimask_outputs=3,
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
        )

    def from_ckpt(self, ckpt, strict=False):
        self.transformer.load_state_dict(
            ckpt.block_state("video.transformer"),
            strict=strict,
        )
        self.image_pe.load_state_dict(
            ckpt.block_state("video.image_pe_layer"),
            strict=strict,
        )
        self.mask_decoder.load_state_dict(
            ckpt.block_state("video.sam_mask_decoder"),
            strict=strict,
        )
        return self

    def forward(self, frame, memory, multimask=False) -> dict[str, object]:
        encoded = self.encode(frame, memory)
        masks = self.mask_decoder(
            image_embeddings=encoded,
            image_pe=self.image_pe(encoded.shape[-2:]).unsqueeze(0).to(encoded),
            high_res_features=self.high_res(frame),
            multimask_output=multimask,
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

    def high_res(self, frame):
        fpn = frame["backbone_fpn"]
        return [
            self.mask_decoder.conv_s0(self.tensor(fpn[0])),
            self.mask_decoder.conv_s1(self.tensor(fpn[1])),
        ]

    @staticmethod
    def seq(value):
        tensor = VideoTrack.tensor(value)
        if tensor.dim() == 4:
            return tensor.flatten(2).permute(2, 0, 1)
        return tensor

    @staticmethod
    def tensor(value):
        return getattr(value, "tensors", value)
