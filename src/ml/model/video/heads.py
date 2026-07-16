from torch import nn

from ...blocks.video.tracking import make_two_way_transformer
from ...components.nn.layers import MLP
from ...components.sam.mask_decoder import MaskDecoder
from ...components.sam.prompt_encoder import PromptEncoder


def build_sam_heads(self):
    self.sam_prompt_embed_dim = self.hidden_dim
    self.sam_image_embedding_size = self.image_size // self.backbone_stride
    build_prompt_encoder(self)
    build_interactive_decoder(self)
    build_object_pointer_projection(self)


def build_prompt_encoder(self):
    self.interactive_sam_prompt_encoder = PromptEncoder(
        embed_dim=self.sam_prompt_embed_dim,
        image_embedding_size=(
            self.sam_image_embedding_size,
            self.sam_image_embedding_size,
        ),
        input_image_size=(self.image_size, self.image_size),
        mask_in_chans=16,
    )


def build_interactive_decoder(self):
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
