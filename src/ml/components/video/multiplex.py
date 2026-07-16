import torch
from torch import nn

from ..nn.layers import LayerNorm2d, MLP
from .mask_selection import score_stability
from .multiplex_ops import (
    check_forward_args,
    check_forward_shapes,
    decode_masks,
    prepare_tokens,
    project_mask_tokens,
    reshape_mask_tokens,
    score_objects,
    select_mask_outputs,
    select_sam_tokens,
    split_tokens,
    upscale,
)


class MultiplexMaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        multiplex_count: int,
        num_multimask_outputs: int = 3,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        decode_mask_with_shared_tokens: bool = False,
        decode_mask_attribute_with_shared_tokens: bool = False,
        multimask_outputs_only: bool = False,
    ) -> None:
        super().__init__()
        self.transformer = transformer

        self._init_mask_layout(
            multiplex_count,
            num_multimask_outputs,
            multimask_outputs_only,
            decode_mask_with_shared_tokens,
            decode_mask_attribute_with_shared_tokens,
        )
        self._init_tokens(
            transformer_dim,
            pred_obj_scores,
            use_multimask_token_for_obj_ptr,
        )
        self._init_upscaling(
            transformer_dim,
            activation,
            use_high_res_features,
        )
        self._init_mask_heads(transformer_dim)
        self._init_score_heads(
            transformer_dim,
            iou_head_hidden_dim,
            iou_head_depth,
            iou_prediction_use_sigmoid,
            pred_obj_scores_mlp,
        )
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def _init_mask_layout(
        self,
        multiplex_count,
        num_multimask_outputs,
        multimask_outputs_only,
        decode_mask_with_shared_tokens,
        decode_mask_attribute_with_shared_tokens,
    ):
        self.multiplex_count = multiplex_count
        self.num_multimask_outputs = num_multimask_outputs
        self.multimask_outputs_only = multimask_outputs_only
        self.decode_mask_with_shared_tokens = decode_mask_with_shared_tokens
        self.decode_mask_attribute_with_shared_tokens = (
            decode_mask_attribute_with_shared_tokens
        )

        if self.decode_mask_with_shared_tokens:
            assert (
                multimask_outputs_only
            ), "multimask_outputs_only must be True if decode_mask_with_shared_tokens"

        self.num_mask_output_per_object = num_multimask_outputs
        if not self.multimask_outputs_only:
            self.num_mask_output_per_object += 1

        self.num_mask_tokens = multiplex_count * self.num_mask_output_per_object
        if self.decode_mask_with_shared_tokens:
            self.num_mask_tokens = multiplex_count

    def _init_tokens(
        self,
        transformer_dim,
        pred_obj_scores,
        use_multimask_token_for_obj_ptr,
    ):
        self.pred_obj_scores = pred_obj_scores
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        if not self.decode_mask_attribute_with_shared_tokens:
            self.iou_token = nn.Embedding(self.multiplex_count, transformer_dim)
            if self.pred_obj_scores:
                self.obj_score_token = nn.Embedding(
                    self.multiplex_count, transformer_dim
                )

        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

    def _init_upscaling(
        self,
        transformer_dim,
        activation,
        use_high_res_features,
    ):
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

    def _init_mask_heads(self, transformer_dim):
        if self.num_multimask_outputs == 0:
            self.output_hypernetworks_mlp = MLP(
                transformer_dim, transformer_dim, transformer_dim // 8, 3
            )
            return

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_output_per_object)
            ]
        )

    def _init_score_heads(
        self,
        transformer_dim,
        iou_head_hidden_dim,
        iou_head_depth,
        iou_prediction_use_sigmoid,
        pred_obj_scores_mlp,
    ):
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self._iou_output_dim(),
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )

        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

    def _iou_output_dim(self):
        if (
            self.decode_mask_attribute_with_shared_tokens
            and not self.decode_mask_with_shared_tokens
        ):
            return 1
        return self.num_mask_output_per_object

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        multimask_output: bool,
        high_res_features: list[torch.Tensor] | None = None,
        extra_per_object_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        check_forward_args(self, multimask_output)

        out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            high_res_features=high_res_features,
            extra_per_object_embeddings=extra_per_object_embeddings,
        )

        masks, iou_pred = select_mask_outputs(
            self,
            masks=out["masks"],
            iou_pred=out["iou_pred"],
            multimask_output=multimask_output,
        )
        sam_tokens_out = select_sam_tokens(
            self,
            mask_tokens_out=out["mask_tokens_out"],
            multimask_output=multimask_output,
        )

        del out["mask_tokens_out"]
        out["masks"] = masks
        out["iou_pred"] = iou_pred
        out["sam_tokens_out"] = sam_tokens_out
        check_forward_shapes(
            self,
            masks=masks,
            iou_pred=iou_pred,
            sam_tokens_out=sam_tokens_out,
            multimask_output=multimask_output,
        )

        return out

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        high_res_features: list[torch.Tensor] | None = None,
        extra_per_object_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = image_embeddings.shape[0]
        tokens = prepare_tokens(self, batch_size, extra_per_object_embeddings)
        src = image_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        batch, channels, height, width = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        obj_score_token_out, iou_token_out, mask_tokens_out = split_tokens(self, hs)

        src = src.transpose(1, 2).view(batch, channels, height, width)
        upscaled_embedding = upscale(self, src, high_res_features)
        mask_tokens_out = reshape_mask_tokens(self, mask_tokens_out, batch_size)
        hyper_in = project_mask_tokens(self, mask_tokens_out)
        masks = decode_masks(self, hyper_in, upscaled_embedding)

        iou_pred = self.iou_prediction_head(iou_token_out).view(
            batch, self.multiplex_count, self.num_mask_output_per_object
        )
        obj_scores = score_objects(self, obj_score_token_out, iou_pred)
        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "mask_tokens_out": mask_tokens_out,
            "object_score_logits": obj_scores,
        }

    def get_stability_scores(self, mask_logits):
        return score_stability(mask_logits, self.dynamic_multimask_stability_delta)
