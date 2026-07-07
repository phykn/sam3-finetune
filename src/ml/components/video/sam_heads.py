import torch
from torch import nn

from ..sam.mask_decoder import MaskDecoder
from ..sam.prompt_encoder import PositionEmbeddingRandom, PromptEncoder
from .create import make_two_way_transformer
from .mlp import MLP
from .multiplex import MultiplexMaskDecoder


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
