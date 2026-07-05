from ....types import NestedTensor

NECK_OUTPUTS = ("interactive", "sam2_backbone_out")


def forward_image(
    self,
    img_batch,
    *,
    need_sam3_out: bool = False,
    need_interactive_out: bool = False,
    need_propagation_out: bool = False,
):
    if self.share_necks:
        need_propagation_out = need_interactive_out or need_propagation_out
        need_interactive_out = False
        backbone_out = self.backbone.forward_image(
            img_batch,
            need_sam3_out=need_sam3_out,
            need_sam2_out=need_propagation_out,
        )
        backbone_out["interactive"] = backbone_out["sam2_backbone_out"]
    else:
        backbone_out = self.backbone.forward_image(
            img_batch,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )

    project_high_res_features(
        self,
        backbone_out,
        need_interactive_out=need_interactive_out,
        need_propagation_out=need_propagation_out,
    )
    clone_backbone_tensors(self, backbone_out)
    return backbone_out


def project_high_res_features(
    self,
    backbone_out,
    *,
    need_interactive_out,
    need_propagation_out,
):
    if not self.use_high_res_features_in_sam:
        return

    if need_interactive_out:
        project_neck_features(
            backbone_out["interactive"], self.interactive_sam_mask_decoder
        )
    if need_propagation_out:
        project_neck_features(backbone_out["sam2_backbone_out"], self.sam_mask_decoder)


def project_neck_features(neck_out, decoder):
    neck_out["backbone_fpn"][0].tensors = decoder.conv_s0(
        neck_out["backbone_fpn"][0].tensors
    )
    neck_out["backbone_fpn"][1].tensors = decoder.conv_s1(
        neck_out["backbone_fpn"][1].tensors
    )


def clone_backbone_tensors(self, backbone_out):
    for neck_out in backbone_out.values():
        if not isinstance(neck_out, dict) or "backbone_fpn" not in neck_out:
            continue

        for idx in range(len(neck_out["backbone_fpn"])):
            neck_out["backbone_fpn"][idx].tensors = self._maybe_clone(
                neck_out["backbone_fpn"][idx].tensors
            )
            neck_out["vision_pos_enc"][idx] = self._maybe_clone(
                neck_out["vision_pos_enc"][idx]
            )


def prepare_backbone_features(self, backbone_out):
    backbone_features = {}

    for neck_key in NECK_OUTPUTS:
        if neck_key not in backbone_out:
            continue

        neck_out = backbone_out[neck_key]
        assert len(neck_out["backbone_fpn"]) == len(neck_out["vision_pos_enc"])
        assert len(neck_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = neck_out["backbone_fpn"][-self.num_feature_levels :]
        pos_embeds = neck_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in pos_embeds]
        vision_feats = [x.tensors.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in pos_embeds]
        vision_masks = [x.mask for x in feature_maps]

        for idx, mask in enumerate(vision_masks):
            if mask is not None:
                vision_masks[idx] = mask.flatten(1)

        backbone_features[neck_key] = {
            "vision_feats": vision_feats,
            "vision_pos_embeds": vision_pos_embeds,
            "vision_masks": vision_masks,
            "feat_sizes": feat_sizes,
        }

    return backbone_features


def prepare_backbone_features_per_frame(
    self,
    img_batch,
    img_ids,
    *,
    need_interactive_out: bool = False,
    need_propagation_out: bool = False,
):
    assert img_ids.numel() == 1
    image = img_batch.tensors[img_ids]
    image_mask = img_batch.mask[img_ids] if img_batch.mask is not None else None

    backbone_out = self.forward_image(
        NestedTensor(tensors=image, mask=image_mask),
        need_interactive_out=need_interactive_out,
        need_propagation_out=need_propagation_out,
    )

    features = self._prepare_backbone_features(backbone_out)
    return image, features


def get_interactive_pix_mem(self, features, feat_sizes):
    assert self.directly_add_no_mem_embed
    pix_feat = features[-1] + self.interactivity_no_mem_embed

    batch_size = features[-1].size(1)
    channels = self.hidden_dim
    height, width = feat_sizes[-1]
    return pix_feat.permute(1, 2, 0).view(batch_size, channels, height, width)


def get_image_feature(self, inference_state, frame_idx, batch_size):
    image, backbone_out = inference_state["cached_features"].get(
        frame_idx, (None, None)
    )
    if backbone_out is None:
        image = inference_state["images"][frame_idx].cuda().float().unsqueeze(0)
        backbone_out = self.forward_image(
            NestedTensor(tensors=image, mask=None),
            need_sam3_out=True,
            need_interactive_out=True,
            need_propagation_out=True,
        )
        inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

    features = self._prepare_backbone_features(backbone_out)
    return image, features


def get_maskmem_pos_enc(self, inference_state, current_out):
    model_constants = inference_state["constants"]
    out_maskmem_pos_enc = current_out.get("maskmem_pos_enc")
    if out_maskmem_pos_enc is None:
        return None

    if "maskmem_pos_enc" not in model_constants:
        assert isinstance(out_maskmem_pos_enc, list)
        maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
        model_constants["maskmem_pos_enc"] = maskmem_pos_enc
    else:
        maskmem_pos_enc = model_constants["maskmem_pos_enc"]

    batch_size = out_maskmem_pos_enc[0].size(0)
    return [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]
