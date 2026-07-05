import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from ..components.nn.layers import MLP
from .pixel import PixelDecoder


class LinearPresenceHead(nn.Sequential):
    def __init__(self, d_model):
        # Old checkpoints expect a three-module Sequential here.
        super().__init__(nn.Identity(), nn.Identity(), nn.Linear(d_model, 1))

    def forward(self, hs, prompt, prompt_mask):
        return super().forward(hs)


class MaskHead(nn.Module):
    def __init__(self, hidden_dim, mask_dim):
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, obj_queries, pixel_embed):
        mask_embed = self.mask_embed(obj_queries)

        if obj_queries.ndim == 3:
            if pixel_embed.ndim == 3:
                equation = "bqc,chw->bqhw"
            else:
                equation = "bqc,bchw->bqhw"
        elif pixel_embed.ndim == 3:
            equation = "lbqc,chw->lbqhw"
        else:
            equation = "lbqc,bchw->lbqhw"

        return torch.einsum(equation, mask_embed, pixel_embed)


def predict_masks(mask_predictor, obj_queries, mask_features, no_dec, aux_masks):
    if no_dec:
        return mask_predictor(mask_features)
    if aux_masks:
        return mask_predictor(obj_queries, mask_features)
    return mask_predictor(obj_queries[-1], mask_features)


class SegmentationHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        use_encoder_inputs=False,
        aux_masks=False,
        no_dec=False,
        pixel_decoder=None,
        act_ckpt=False,
        shared_conv=False,
        compile_mode_pixel_decoder=None,
    ):
        super().__init__()
        self.use_encoder_inputs = use_encoder_inputs
        self.aux_masks = aux_masks
        if pixel_decoder is not None:
            self.pixel_decoder = pixel_decoder
        else:
            self.pixel_decoder = PixelDecoder(
                hidden_dim,
                upsampling_stages,
                shared_conv=shared_conv,
                compile_mode=compile_mode_pixel_decoder,
            )
        self.no_dec = no_dec
        if no_dec:
            self.mask_predictor = nn.Conv2d(
                hidden_dim, 1, kernel_size=3, stride=1, padding=1
            )
        else:
            self.mask_predictor = MaskHead(hidden_dim, mask_dim=hidden_dim)

        self.act_ckpt = act_ckpt

        self.instance_keys = ["pred_masks"]

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        self._device = None
        return super().to(*args, **kwargs)

    def _unwrap_feats(self, backbone_feats: list[torch.Tensor]) -> list[torch.Tensor]:
        from ..types import NestedTensor

        return [
            feat.tensors if isinstance(feat, NestedTensor) else feat
            for feat in backbone_feats
        ]

    def _decode_encoder_pixels(
        self,
        backbone_feats: list[torch.Tensor],
        image_ids,
        encoder_hidden_states,
    ) -> torch.Tensor:
        image_ids = image_ids.to(backbone_feats[0].device)
        model_device = self.device

        if backbone_feats[0].shape[0] > 1:
            pixel_feats = [
                feat[image_ids, ...].to(model_device) for feat in backbone_feats
            ]
        else:
            pixel_feats = [feat.clone() for feat in backbone_feats]

        encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0)
        spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
        encoder_visual_embed = encoder_hidden_states[..., :spatial_dim].reshape(
            -1,
            *backbone_feats[-1].shape[1:],
        )
        pixel_feats[-1] = encoder_visual_embed

        if self.act_ckpt:
            return checkpoint.checkpoint(
                self.pixel_decoder,
                pixel_feats,
                use_reentrant=False,
            )
        return self.pixel_decoder(pixel_feats)

    def _decode_backbone_pixels(
        self,
        backbone_feats: list[torch.Tensor],
        image_ids,
    ) -> torch.Tensor:
        backbone_feats = [feat.to(self.device) for feat in backbone_feats]
        pixel_embed = self.pixel_decoder(backbone_feats)
        if pixel_embed.shape[0] == 1:
            # For batch_size=1 training, we can avoid the indexing to save memory
            return pixel_embed.squeeze(0)
        return pixel_embed[image_ids, ...]

    def _embed_pixels(
        self,
        backbone_feats: list[torch.Tensor],
        image_ids,
        encoder_hidden_states,
    ) -> torch.Tensor:
        backbone_feats = self._unwrap_feats(backbone_feats)

        if self.use_encoder_inputs:
            return self._decode_encoder_pixels(
                backbone_feats=backbone_feats,
                image_ids=image_ids,
                encoder_hidden_states=encoder_hidden_states,
            )
        return self._decode_backbone_pixels(backbone_feats, image_ids)

    def forward(
        self,
        backbone_feats: list[torch.Tensor],
        obj_queries: torch.Tensor,
        image_ids,
        encoder_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        return {
            "pred_masks": predict_masks(
                self.mask_predictor,
                obj_queries,
                pixel_embed,
                self.no_dec,
                self.aux_masks,
            )
        }


class UniversalSegmentationHead(SegmentationHead):
    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        pixel_decoder,
        aux_masks=False,
        no_dec=False,
        act_ckpt=False,
        presence_head: bool = False,
        dot_product_scorer=None,
        cross_attend_prompt=None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            upsampling_stages=upsampling_stages,
            use_encoder_inputs=True,
            aux_masks=aux_masks,
            no_dec=no_dec,
            pixel_decoder=pixel_decoder,
            act_ckpt=act_ckpt,
        )
        self.d_model = hidden_dim

        if dot_product_scorer is not None:
            assert (
                presence_head
            ), "Specifying a dot product scorer without a presence head is likely a mistake"

        self.presence_head = None
        if presence_head:
            self.presence_head = (
                dot_product_scorer
                if dot_product_scorer is not None
                else LinearPresenceHead(self.d_model)
            )

        self.cross_attend_prompt = cross_attend_prompt
        if self.cross_attend_prompt is not None:
            self.cross_attn_norm = nn.LayerNorm(self.d_model)

        self.semantic_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, 1, kernel_size=1)
        self.instance_seg_head = nn.Conv2d(
            self.pixel_decoder.out_dim, self.d_model, kernel_size=1
        )

    def _attend_prompt(self, encoder_hidden_states, prompt, prompt_mask):
        if self.cross_attend_prompt is None:
            return encoder_hidden_states

        attended = self.cross_attn_norm(encoder_hidden_states)
        attended = self.cross_attend_prompt(
            query=attended,
            key=prompt,
            value=prompt,
            key_padding_mask=prompt_mask,
        )[0]
        return attended + encoder_hidden_states

    def _predict_presence(self, encoder_hidden_states, prompt, prompt_mask):
        if self.presence_head is None:
            return None

        batch_size = encoder_hidden_states.shape[1]
        pooled_enc = encoder_hidden_states.mean(0)
        presence = self.presence_head(
            pooled_enc.view(1, batch_size, 1, self.d_model),
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        return presence.squeeze(0).squeeze(1)

    def forward(
        self,
        backbone_feats: list[torch.Tensor],
        obj_queries: torch.Tensor,
        image_ids,
        encoder_hidden_states: torch.Tensor | None = None,
        prompt: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        assert encoder_hidden_states is not None

        encoder_hidden_states = self._attend_prompt(
            encoder_hidden_states,
            prompt,
            prompt_mask,
        )
        presence_logit = self._predict_presence(
            encoder_hidden_states,
            prompt,
            prompt_mask,
        )

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        instance_embeds = self.instance_seg_head(pixel_embed)

        return {
            "pred_masks": predict_masks(
                self.mask_predictor,
                obj_queries,
                instance_embeds,
                self.no_dec,
                self.aux_masks,
            ),
            "semantic_seg": self.semantic_seg_head(pixel_embed),
            "presence_logit": presence_logit,
        }
