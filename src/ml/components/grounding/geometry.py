import torch
import torch.nn as nn
import torchvision

from ....ops.box import convert_to_xyxy
from ...runtime.checkpointing import activation_ckpt_wrapper
from ..nn.layers import clone_modules
from .prompt import Prompt
from .sequence import concat_padded_sequences


def add_embed(current, update):
    return update if current is None else current + update


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


class SequenceGeometryEncoder(nn.Module):
    def __init__(
        self,
        encode_boxes_as_points: bool,
        points_direct_project: bool,
        points_pool: bool,
        points_pos_enc: bool,
        boxes_direct_project: bool,
        boxes_pool: bool,
        boxes_pos_enc: bool,
        d_model: int,
        pos_enc,
        num_layers: int,
        layer: nn.Module,
        roi_size: int = 7,
        add_cls: bool = True,
        add_post_encode_proj: bool = True,
        mask_encoder: MaskEncoder = None,
        add_mask_label: bool = False,
        use_act_ckpt: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.pos_enc = pos_enc
        self.encode_boxes_as_points = encode_boxes_as_points
        self.roi_size = roi_size
        # Box-as-point mode has regular, top-left, and bottom-right point labels.
        num_labels = 6 if self.encode_boxes_as_points else 2
        self.label_embed = torch.nn.Embedding(num_labels, self.d_model)

        self.cls_embed = None
        if add_cls:
            self.cls_embed = torch.nn.Embedding(1, self.d_model)

        assert (
            points_direct_project or points_pos_enc or points_pool
        ), "Error: need at least one way to encode points"
        assert (
            encode_boxes_as_points
            or boxes_direct_project
            or boxes_pos_enc
            or boxes_pool
        ), "Error: need at least one way to encode boxes"

        self.points_direct_project = None
        if points_direct_project:
            self.points_direct_project = nn.Linear(2, self.d_model)
        self.points_pool_project = None
        if points_pool:
            self.points_pool_project = nn.Linear(self.d_model, self.d_model)
        self.points_pos_enc_project = None
        if points_pos_enc:
            self.points_pos_enc_project = nn.Linear(self.d_model, self.d_model)

        self.boxes_direct_project = None
        self.boxes_pool_project = None
        self.boxes_pos_enc_project = None
        if not encode_boxes_as_points:
            if boxes_direct_project:
                self.boxes_direct_project = nn.Linear(4, self.d_model)
            if boxes_pool:
                self.boxes_pool_project = nn.Conv2d(
                    self.d_model, self.d_model, self.roi_size
                )
            if boxes_pos_enc:
                self.boxes_pos_enc_project = nn.Linear(self.d_model + 2, self.d_model)

        self.final_proj = None
        if add_post_encode_proj:
            self.final_proj = nn.Linear(self.d_model, self.d_model)
            self.norm = nn.LayerNorm(self.d_model)

        self.img_pre_norm = nn.Identity()
        if self.points_pool_project is not None or self.boxes_pool_project is not None:
            self.img_pre_norm = nn.LayerNorm(self.d_model)

        self.encode = None
        if num_layers > 0:
            assert (
                add_cls
            ), "It's currently highly recommended to add a CLS when using a transformer"
            self.encode = clone_modules(layer, num_layers)
            self.encode_norm = nn.LayerNorm(self.d_model)

        if mask_encoder is not None:
            assert isinstance(
                mask_encoder, MaskEncoder
            ), f"Expected mask_encoder of type MaskEncoder. Got {type(mask_encoder)}."
            if add_mask_label:
                self.mask_label_embed = torch.nn.Embedding(2, self.d_model)
        self.add_mask_label = add_mask_label
        self.mask_encoder = mask_encoder
        self.use_act_ckpt = use_act_ckpt

    def _encode_points(self, points, points_mask, points_labels, img_feats):
        points_embed = None
        num_points, batch_size = points.shape[:2]

        if self.points_direct_project is not None:
            points_embed = add_embed(
                points_embed,
                self.points_direct_project(points),
            )

        if self.points_pool_project is not None:
            # points are [Num_points, bs, 2], normalized in [0, 1]
            # the grid needs to be [Bs, H_out, W_out, 2] normalized in [-1,1]
            grid = points.transpose(0, 1).unsqueeze(2)
            grid = (grid * 2) - 1
            sampled = torch.nn.functional.grid_sample(
                img_feats, grid, align_corners=False
            )
            assert list(sampled.shape) == [batch_size, self.d_model, num_points, 1]
            sampled = sampled.squeeze(-1).permute(2, 0, 1)
            proj = self.points_pool_project(sampled)
            points_embed = add_embed(points_embed, proj)

        if self.points_pos_enc_project is not None:
            x, y = points.unbind(-1)
            enc_x, enc_y = self.pos_enc._encode_xy(x.flatten(), y.flatten())
            enc_x = enc_x.view(num_points, batch_size, enc_x.shape[-1])
            enc_y = enc_y.view(num_points, batch_size, enc_y.shape[-1])
            enc = torch.cat([enc_x, enc_y], -1)

            proj = self.points_pos_enc_project(enc)
            points_embed = add_embed(points_embed, proj)

        type_embed = self.label_embed(points_labels.long())
        return type_embed + points_embed, points_mask

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats):
        boxes_embed = None
        num_boxes, batch_size = boxes.shape[:2]

        if self.boxes_direct_project is not None:
            boxes_embed = add_embed(
                boxes_embed,
                self.boxes_direct_project(boxes),
            )

        if self.boxes_pool_project is not None:
            height, width = img_feats.shape[-2:]

            # boxes are [Num_boxes, bs, 4], normalized in [0, 1]
            boxes_xyxy = convert_to_xyxy(boxes)
            scale = torch.tensor([width, height, width, height], dtype=boxes_xyxy.dtype)
            scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)
            scale = scale.view(1, 1, 4)
            boxes_xyxy = boxes_xyxy * scale
            sampled = torchvision.ops.roi_align(
                img_feats, boxes_xyxy.float().transpose(0, 1).unbind(0), self.roi_size
            )
            assert list(sampled.shape) == [
                batch_size * num_boxes,
                self.d_model,
                self.roi_size,
                self.roi_size,
            ]
            proj = self.boxes_pool_project(sampled)
            proj = proj.view(batch_size, num_boxes, self.d_model).transpose(0, 1)
            boxes_embed = add_embed(boxes_embed, proj)

        if self.boxes_pos_enc_project is not None:
            cx, cy, w, h = boxes.unbind(-1)
            enc = self.pos_enc.encode_boxes(
                cx.flatten(), cy.flatten(), w.flatten(), h.flatten()
            )
            enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])

            proj = self.boxes_pos_enc_project(enc)
            boxes_embed = add_embed(boxes_embed, proj)

        type_embed = self.label_embed(boxes_labels.long())
        return type_embed + boxes_embed, boxes_mask

    def _encode_masks(
        self,
        masks: torch.Tensor,
        attn_mask: torch.Tensor,
        mask_labels: torch.Tensor,
        img_feats: torch.Tensor = None,
    ):
        num_masks, batch_size = masks.shape[:2]
        assert (
            num_masks == 1
        ), "We assume one mask per prompt for now. Code should still be functional if this assertion is removed."
        assert list(attn_mask.shape) == [
            batch_size,
            num_masks,
        ], f"Expected attn_mask to be of shape {batch_size}x{num_masks}. Got {list(attn_mask.shape)}."
        masks, pos = self.mask_encoder(
            masks=masks.flatten(0, 1).float(),
            pix_feat=img_feats,
        )
        height, width = masks.shape[-2:]
        tokens_per_mask = height * width
        masks = masks + pos
        masks = masks.view(num_masks, batch_size, *masks.shape[1:])
        masks = masks.flatten(-2)  # [N_masks, B, C, H*W]
        masks = masks.permute(0, 3, 1, 2)
        masks = masks.flatten(0, 1)  # [N_masks*H*W, B, C]
        attn_mask = attn_mask.repeat_interleave(tokens_per_mask, dim=1)
        if self.add_mask_label:
            masks = masks + self.mask_label_embed(mask_labels.long())
        return masks, attn_mask

    def _pool_image_features(self, img_feats, img_sizes):
        if self.points_pool_project is None and self.boxes_pool_project is None:
            return img_feats

        assert len(img_feats) == len(img_sizes)
        img_feat = self.img_pre_norm(img_feats[-1])
        height, width = img_sizes[-1]
        assert img_feat.shape[0] == height * width

        batch_size, channels = img_feat.shape[-2:]
        img_feat = img_feat.permute(1, 2, 0)
        return img_feat.view(batch_size, channels, height, width)

    def _append_box_points(
        self,
        points,
        points_mask,
        points_labels,
        boxes,
        boxes_mask,
        boxes_labels,
    ):
        assert boxes is not None
        assert boxes_mask is not None
        assert boxes_labels is not None
        assert boxes.shape[-1] == 4

        box_corners = convert_to_xyxy(boxes).split(split_size=2, dim=-1)
        for corner, label_offset in zip(box_corners, (2, 4)):
            corner_labels = boxes_labels + label_offset
            points, _ = concat_padded_sequences(
                points,
                points_mask,
                corner,
                boxes_mask,
            )
            points_labels, points_mask = concat_padded_sequences(
                points_labels.unsqueeze(-1),
                points_mask,
                corner_labels.unsqueeze(-1),
                boxes_mask,
            )
            points_labels = points_labels.squeeze(-1)

        return points, points_mask, points_labels

    def _append_cls(self, embeds, mask):
        if self.cls_embed is None:
            return embeds, mask

        bs = embeds.shape[1]
        cls = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1)
        cls_mask = torch.zeros(bs, 1, dtype=mask.dtype, device=mask.device)
        return concat_padded_sequences(embeds, mask, cls, cls_mask)

    def _run_transformer(self, embeds, mask, img_feats, img_pos_embeds):
        if self.final_proj is not None:
            embeds = self.norm(self.final_proj(embeds))

        if self.encode is None:
            return embeds

        for layer in self.encode:
            embeds = activation_ckpt_wrapper(layer)(
                tgt=embeds,
                memory=img_feats,
                tgt_key_padding_mask=mask,
                pos=img_pos_embeds,
                act_ckpt_enable=self.training and self.use_act_ckpt,
            )
        return self.encode_norm(embeds)

    def forward(self, geo_prompt: Prompt, img_feats, img_sizes, img_pos_embeds=None):
        points = geo_prompt.point_embeddings
        points_mask = geo_prompt.point_mask
        points_labels = geo_prompt.point_labels
        boxes = geo_prompt.box_embeddings
        boxes_mask = geo_prompt.box_mask
        boxes_labels = geo_prompt.box_labels
        masks = geo_prompt.mask_embeddings
        masks_mask = geo_prompt.mask_mask
        masks_labels = geo_prompt.mask_labels
        image_tokens = img_feats[-1]  # [H*W, B, C]
        image_pos_embeds = (
            img_pos_embeds[-1]
            if img_pos_embeds is not None
            else torch.zeros_like(image_tokens)
        )
        pooled_img_feats = self._pool_image_features(img_feats, img_sizes)

        if self.encode_boxes_as_points:
            points, points_mask, points_labels = self._append_box_points(
                points=points,
                points_mask=points_mask,
                points_labels=points_labels,
                boxes=boxes,
                boxes_mask=boxes_mask,
                boxes_labels=boxes_labels,
            )

        final_embeds, final_mask = self._encode_points(
            points=points,
            points_mask=points_mask,
            points_labels=points_labels,
            img_feats=pooled_img_feats,
        )

        if not self.encode_boxes_as_points:
            boxes_embeds, boxes_mask = self._encode_boxes(
                boxes=boxes,
                boxes_mask=boxes_mask,
                boxes_labels=boxes_labels,
                img_feats=pooled_img_feats,
            )

            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, boxes_embeds, boxes_mask
            )

        if masks is not None and self.mask_encoder is not None:
            masks_embed, masks_mask = self._encode_masks(
                masks=masks,
                attn_mask=masks_mask,
                mask_labels=masks_labels,
                img_feats=pooled_img_feats,
            )
            if points.size(0) == boxes.size(0) == 0:
                return masks_embed, masks_mask

        bs = final_embeds.shape[1]
        assert final_mask.shape[0] == bs
        final_embeds, final_mask = self._append_cls(final_embeds, final_mask)
        final_embeds = self._run_transformer(
            embeds=final_embeds,
            mask=final_mask,
            img_feats=image_tokens,
            img_pos_embeds=image_pos_embeds,
        )

        if masks is not None and self.mask_encoder is not None:
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, masks_embed, masks_mask
            )
        return final_embeds, final_mask
