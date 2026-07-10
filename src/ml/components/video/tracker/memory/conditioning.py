import torch

from ..multiplex.state import MultiplexState
from .context import collect_memory_context


def reshape_current_feature(vision_feat, batch_size, channels, height, width):
    return vision_feat.permute(1, 2, 0).view(batch_size, channels, height, width)


def run_memory_encoder(
    model,
    *,
    current_vision_feats,
    current_vision_pos_embeds,
    vision_feat,
    vision_mask,
    vision_pos_embed,
    feat_sizes,
    context,
):
    prompt = torch.cat(context["prompts"], dim=0)
    prompt_pos_embed = torch.cat(context["prompt_pos"], dim=0)
    prompt_mask = None

    if model.save_image_features:
        assert prompt_mask is None
        assert vision_mask is None
        image_feat = torch.cat(context["image_feats"], dim=0)
        image_pos_embed = torch.cat(context["image_pos"], dim=0)

        return model.transformer.encoder(
            image=current_vision_feats[-1],
            src=vision_feat,
            memory_image=image_feat,
            memory=prompt,
            image_pos=current_vision_pos_embeds[-1],
            src_pos=vision_pos_embed,
            memory_image_pos=image_pos_embed,
            memory_pos=prompt_pos_embed,
            num_obj_ptr_tokens=context["num_obj_ptr_tokens"],
        )

    return model.transformer.encoder(
        src=vision_feat,
        src_key_padding_mask=vision_mask,
        src_pos=vision_pos_embed,
        prompt=prompt,
        prompt_pos=prompt_pos_embed,
        prompt_key_padding_mask=prompt_mask,
        feat_sizes=feat_sizes,
        num_obj_ptr_tokens=context["num_obj_ptr_tokens"],
    )


def prepare_memory_conditioned_features(
    self,
    *,
    frame_idx,
    is_init_cond_frame,
    current_vision_feats,
    current_vision_masks,
    current_vision_pos_embeds,
    feat_sizes,
    output_dict,
    num_frames,
    use_prev_mem_frame=True,
    multiplex_state: MultiplexState,
):
    current = prepare_current_memory_inputs(
        self,
        current_vision_feats,
        current_vision_masks,
        current_vision_pos_embeds,
        feat_sizes,
        multiplex_state,
    )

    if self.num_maskmem == 0:
        return current["reshaped"]

    if is_init_cond_frame or not use_prev_mem_frame:
        raise RuntimeError(
            "Any init cond frame should have gone to _use_mask_as_output instead"
        )

    context = get_memory_context(
        self,
        frame_idx=frame_idx,
        output_dict=output_dict,
        num_frames=num_frames,
        current=current,
        multiplex_state=multiplex_state,
    )
    if should_skip_memory_encoder(self, context):
        return current["reshaped"]

    encoder_out = run_memory_encoder(
        self,
        current_vision_feats=current_vision_feats,
        current_vision_pos_embeds=current_vision_pos_embeds,
        vision_feat=current["vision_feat"],
        vision_mask=current["vision_mask"],
        vision_pos_embed=current["vision_pos_embed"],
        feat_sizes=feat_sizes,
        context=context,
    )
    return reshape_current_feature(
        encoder_out["memory"],
        current["batch_size"],
        current["channels"],
        current["height"],
        current["width"],
    )


def prepare_current_memory_inputs(
    self,
    current_vision_feats,
    current_vision_masks,
    current_vision_pos_embeds,
    feat_sizes,
    multiplex_state,
):
    batch_size = multiplex_state.num_buckets
    channels = self.hidden_dim
    height, width = feat_sizes[-1]
    vision_feat = current_vision_feats[-1].expand(-1, batch_size, -1)
    vision_mask = (
        current_vision_masks[-1].expand(-1, batch_size, -1)
        if current_vision_masks[-1] is not None
        else None
    )
    vision_pos_embed = current_vision_pos_embeds[-1].expand(-1, batch_size, -1)

    return {
        "batch_size": batch_size,
        "channels": channels,
        "height": height,
        "width": width,
        "device": current_vision_feats[-1].device,
        "vision_feat": vision_feat,
        "vision_mask": vision_mask,
        "vision_pos_embed": vision_pos_embed,
        "reshaped": reshape_current_feature(
            vision_feat,
            batch_size,
            channels,
            height,
            width,
        ),
    }


def get_memory_context(
    self,
    *,
    frame_idx,
    output_dict,
    num_frames,
    current,
    multiplex_state,
):
    return collect_memory_context(
        self,
        frame_idx=frame_idx,
        output_dict=output_dict,
        num_frames=num_frames,
        device=current["device"],
        batch_size=current["batch_size"],
        channels=current["channels"],
        multiplex_state=multiplex_state,
    )


def should_skip_memory_encoder(self, context):
    if len(context["prompts"]) == 0:
        return True
    return self.save_image_features and (
        len(context["image_feats"]) == 0 or len(context["image_pos"]) == 0
    )
