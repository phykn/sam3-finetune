import torch

from .mask_selection import select_dynamic_multimask


def check_forward_args(decoder, multimask_output):
    if decoder.num_multimask_outputs <= 0:
        assert not multimask_output, (
            f"multimask_output must be False with {decoder.num_multimask_outputs=}"
        )

    if decoder.multimask_outputs_only:
        assert multimask_output, (
            f"multimask_output must be True with {decoder.multimask_outputs_only=}"
        )


def select_mask_outputs(decoder, masks, iou_pred, multimask_output):
    if multimask_output:
        if not decoder.multimask_outputs_only:
            masks = masks[:, :, 1:, :, :]
            iou_pred = iou_pred[:, :, 1:]
        return masks, iou_pred

    if decoder.dynamic_multimask_via_stability and not decoder.training:
        return select_dynamic_multimask(
            masks,
            iou_pred,
            delta=decoder.dynamic_multimask_stability_delta,
            threshold=decoder.dynamic_multimask_stability_thresh,
        )

    return masks[:, :, 0:1, :, :], iou_pred[:, :, 0:1]


def select_sam_tokens(decoder, mask_tokens_out, multimask_output):
    if multimask_output and decoder.use_multimask_token_for_obj_ptr:
        if decoder.multimask_outputs_only:
            return mask_tokens_out
        return mask_tokens_out[:, :, 1:]

    # Memory tokens must match multi-click training outputs.
    return mask_tokens_out[:, :, 0:1]


def check_forward_shapes(
    decoder,
    masks,
    iou_pred,
    sam_tokens_out,
    multimask_output,
):
    if multimask_output:
        expected_count = (
            decoder.num_mask_output_per_object
            if decoder.multimask_outputs_only
            else decoder.num_multimask_outputs
        )
        assert masks.shape[2] == expected_count, f"{masks.shape=}, {expected_count=}"
        assert iou_pred.shape[2] == expected_count, (
            f"{iou_pred.shape=}, {expected_count=}"
        )
        if decoder.use_multimask_token_for_obj_ptr:
            if decoder.decode_mask_with_shared_tokens:
                assert sam_tokens_out.shape[2] == 1, f"{sam_tokens_out.shape=}"
            else:
                assert sam_tokens_out.shape[2] == expected_count, (
                    f"{sam_tokens_out.shape=}, {expected_count=}"
                )
        return

    assert masks.shape[2] == 1, f"{masks.shape=}"
    assert iou_pred.shape[2] == 1, f"{iou_pred.shape=}"
    assert sam_tokens_out.shape[2] == 1, f"{sam_tokens_out.shape=}"


def prepare_tokens(decoder, batch_size, extra_per_object_embeddings):
    prefix_tokens = []
    if decoder.pred_obj_scores and not decoder.decode_mask_attribute_with_shared_tokens:
        prefix_tokens.append(decoder.obj_score_token.weight)
    if not decoder.decode_mask_attribute_with_shared_tokens:
        prefix_tokens.append(decoder.iou_token.weight)

    if extra_per_object_embeddings is None:
        mask_tokens = decoder.mask_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        mask_tokens = prepare_object_mask_tokens(
            decoder,
            batch_size,
            extra_per_object_embeddings,
        )

    if not prefix_tokens:
        return mask_tokens

    prefix = torch.cat(prefix_tokens, dim=0)
    prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat([prefix, mask_tokens], dim=1)


def prepare_object_mask_tokens(decoder, batch_size, extra_per_object_embeddings):
    count = decoder.num_mask_output_per_object
    if decoder.decode_mask_with_shared_tokens:
        count = 1

    mask_tokens = decoder.mask_tokens.weight.view(
        1,
        decoder.multiplex_count,
        count,
        -1,
    )
    mask_tokens = mask_tokens.expand(batch_size, -1, -1, -1)
    mask_tokens = mask_tokens + extra_per_object_embeddings.unsqueeze(2)
    return mask_tokens.flatten(1, 2)


def split_tokens(decoder, hs):
    if decoder.decode_mask_attribute_with_shared_tokens:
        assert hs.shape[1] == decoder.num_mask_tokens, (
            f"{hs.shape=}, {decoder.num_mask_tokens=}"
        )
        mask_tokens_out = hs[:, : decoder.num_mask_tokens]
        obj_score_token_out = mask_tokens_out if decoder.pred_obj_scores else None
        return obj_score_token_out, mask_tokens_out, mask_tokens_out

    start = 0
    obj_score_token_out = None
    if decoder.pred_obj_scores:
        obj_score_token_out = hs[:, start : start + decoder.multiplex_count, :]
        start += decoder.multiplex_count

    iou_token_out = hs[:, start : start + decoder.multiplex_count, :]
    start += decoder.multiplex_count
    mask_tokens_out = hs[:, start : start + decoder.num_mask_tokens, :]
    assert hs.shape[1] == start + decoder.num_mask_tokens, (
        f"{hs.shape=}, {start=}, {decoder.num_mask_tokens=}"
    )
    return obj_score_token_out, iou_token_out, mask_tokens_out


def upscale(decoder, src, high_res_features):
    if not decoder.use_high_res_features:
        return decoder.output_upscaling(src)

    dc1, ln1, act1, dc2, act2 = decoder.output_upscaling
    feat_s0, feat_s1 = high_res_features
    x = act1(ln1(dc1(src) + feat_s1))
    return act2(dc2(x) + feat_s0)


def reshape_mask_tokens(decoder, mask_tokens_out, batch_size):
    if decoder.decode_mask_with_shared_tokens:
        return mask_tokens_out.view(batch_size, decoder.multiplex_count, 1, -1)

    return mask_tokens_out.view(
        batch_size,
        decoder.multiplex_count,
        decoder.num_mask_output_per_object,
        -1,
    )


def project_mask_tokens(decoder, mask_tokens_out):
    if decoder.num_multimask_outputs == 0:
        return decoder.output_hypernetworks_mlp(mask_tokens_out[:, :, 0, :]).unsqueeze(
            2
        )

    out = []
    for index in range(decoder.num_mask_output_per_object):
        token = 0 if decoder.decode_mask_with_shared_tokens else index
        out.append(
            decoder.output_hypernetworks_mlps[index](mask_tokens_out[:, :, token, :])
        )
    return torch.stack(out, dim=2)


def decode_masks(decoder, hyper_in, upscaled_embedding):
    batch_size, channels, height, width = upscaled_embedding.shape
    return torch.bmm(
        hyper_in.flatten(1, 2),
        upscaled_embedding.view(batch_size, channels, height * width),
    ).view(
        batch_size,
        decoder.multiplex_count,
        decoder.num_mask_output_per_object,
        height,
        width,
    )


def score_objects(decoder, obj_score_token_out, iou_pred):
    if not decoder.pred_obj_scores:
        return 10.0 * iou_pred.new_ones(iou_pred.shape[0], iou_pred.shape[1])

    scores = decoder.pred_obj_score_head(obj_score_token_out)
    if (
        decoder.decode_mask_attribute_with_shared_tokens
        and not decoder.decode_mask_with_shared_tokens
    ):
        return scores.view(
            iou_pred.shape[0],
            decoder.multiplex_count,
            decoder.num_mask_output_per_object,
        ).sum(-1, keepdim=True)
    return scores
