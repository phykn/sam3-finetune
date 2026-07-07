import torch
from timm.layers import trunc_normal_
from torch import nn


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
        assert mem_dim == self.hidden_dim, (
            "there should be no compression of memory embeddings"
        )

    self.num_maskmem = num_maskmem
    self.sincos_tpos_enc = sincos_tpos_enc
    self.use_maskmem_tpos_v2 = use_maskmem_tpos_v2
    self.maskmem_tpos_enc = nn.Parameter(torch.zeros(num_maskmem, 1, 1, self.mem_dim))
    trunc_normal_(self.maskmem_tpos_enc, std=0.02)

    self.interactivity_no_mem_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
    trunc_normal_(self.interactivity_no_mem_embed, std=0.02)
    self.directly_add_no_mem_embed = directly_add_no_mem_embed


def init_object_pointer_params(
    self,
    pred_obj_scores,
    pred_obj_scores_mlp,
    fixed_no_obj_ptr,
    use_no_obj_ptr,
    use_linear_no_obj_ptr,
    use_mlp_for_obj_ptr_proj,
    no_obj_embed_spatial,
):
    self.pred_obj_scores = pred_obj_scores
    self.pred_obj_scores_mlp = pred_obj_scores_mlp
    self.fixed_no_obj_ptr = fixed_no_obj_ptr
    self.use_no_obj_ptr = use_no_obj_ptr
    self.use_linear_no_obj_ptr = use_linear_no_obj_ptr

    if self.fixed_no_obj_ptr:
        assert self.pred_obj_scores
        assert self.use_obj_ptrs_in_encoder
    if self.pred_obj_scores and self.use_obj_ptrs_in_encoder and self.use_no_obj_ptr:
        if self.use_linear_no_obj_ptr:
            self.no_obj_ptr_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.no_obj_ptr = nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            trunc_normal_(self.no_obj_ptr, std=0.02)

    self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
    self.no_obj_embed_spatial = None
    if no_obj_embed_spatial:
        self.no_obj_embed_spatial = nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        trunc_normal_(self.no_obj_embed_spatial, std=0.02)


def init_condition_embedding_params(
    self,
    add_output_suppression_embeddings,
    add_object_conditional_embeddings,
    add_object_unconditional_embeddings,
    condition_as_mask_input,
    condition_as_mask_input_fg,
    condition_as_mask_input_bg,
):
    self.add_output_suppression_embeddings = add_output_suppression_embeddings
    if self.add_output_suppression_embeddings:
        self.output_valid_embed = nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        self.output_invalid_embed = nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        trunc_normal_(self.output_valid_embed, std=0.02)
        trunc_normal_(self.output_invalid_embed, std=0.02)

    self.add_object_conditional_embeddings = add_object_conditional_embeddings
    if add_object_unconditional_embeddings is None:
        add_object_unconditional_embeddings = add_object_conditional_embeddings
    self.add_object_unconditional_embeddings = add_object_unconditional_embeddings
    if add_object_unconditional_embeddings:
        assert add_object_conditional_embeddings
    if self.add_object_conditional_embeddings:
        self.obj_cond_embed = nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        trunc_normal_(self.obj_cond_embed, std=0.02)
        if self.add_object_unconditional_embeddings:
            self.obj_non_cond_embed = nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            trunc_normal_(self.obj_non_cond_embed, std=0.02)

    self.condition_as_mask_input = condition_as_mask_input
    self.condition_as_mask_input_fg = condition_as_mask_input_fg
    self.condition_as_mask_input_bg = condition_as_mask_input_bg
