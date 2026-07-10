import torch

from ..outputs import OUTPUT_KEYS


def run_memory_encoder(
    self,
    inference_state,
    frame_idx,
    batch_size,
    high_res_masks,
    object_score_logits,
    is_mask_from_pts,
    conditioning_objects=None,
):
    image, backbone_features = self._get_image_feature(
        inference_state, frame_idx, batch_size
    )
    propagation = backbone_features["sam2_backbone_out"]
    propagation_vision_feats = propagation["vision_feats"]
    propagation_vision_pos_embeds = propagation["vision_pos_embeds"]
    propagation_feat_sizes = propagation["feat_sizes"]

    if conditioning_objects is None:
        conditioning_objects = _find_conditioning_objects(inference_state, frame_idx)

    maskmem_features, maskmem_pos_enc = self._encode_new_memory(
        image=image,
        current_vision_feats=propagation_vision_feats,
        feat_sizes=propagation_feat_sizes,
        pred_masks_high_res=high_res_masks,
        object_score_logits=object_score_logits,
        is_mask_from_pts=is_mask_from_pts,
        conditioning_objects=conditioning_objects,
        multiplex_state=inference_state["multiplex_state"],
    )

    storage_device = inference_state["storage_device"]
    maskmem_features = maskmem_features.to(torch.bfloat16)
    maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
    maskmem_pos_enc = self._get_maskmem_pos_enc(
        inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
    )

    image_features = propagation_vision_feats[-1]
    image_features = image_features.to(storage_device, non_blocking=True)
    image_pos_enc = propagation_vision_pos_embeds[-1]
    image_pos_enc = image_pos_enc.to(storage_device, non_blocking=True)
    return maskmem_features, maskmem_pos_enc, image_features, image_pos_enc


def _find_conditioning_objects(inference_state, frame_idx):
    output_dict = inference_state["output_dict"]
    for storage_key in OUTPUT_KEYS:
        storage = output_dict[storage_key]
        if frame_idx in storage:
            return storage[frame_idx]["conditioning_objects"]
    raise ValueError(f"conditioning objects not found at {frame_idx=}")
