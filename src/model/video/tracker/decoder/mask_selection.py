import torch


def score_stability(mask_logits, delta):
    mask_logits = mask_logits.flatten(-2)
    area_i = torch.sum(mask_logits > delta, dim=-1).float()
    area_u = torch.sum(mask_logits > -delta, dim=-1).float()
    return torch.where(area_u > 0, area_i / area_u, 1.0)


def select_dynamic_multimask(all_mask_logits, all_iou_scores, delta, threshold):
    batch_size, multiplex_count = all_mask_logits.shape[:2]
    all_mask_logits = all_mask_logits.flatten(0, 1)
    all_iou_scores = all_iou_scores.flatten(0, 1)

    multimask_logits = all_mask_logits[:, 1:, :, :]
    multimask_iou_scores = all_iou_scores[:, 1:]
    best_score_indices = torch.argmax(multimask_iou_scores, dim=-1)
    batch_indices = torch.arange(
        multimask_iou_scores.size(0), device=all_iou_scores.device
    )
    best_multimask_logits = multimask_logits[batch_indices, best_score_indices]
    best_multimask_logits = best_multimask_logits.unsqueeze(1)
    best_multimask_iou_scores = multimask_iou_scores[
        batch_indices,
        best_score_indices,
    ]
    best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

    singlemask_logits = all_mask_logits[:, 0:1, :, :]
    singlemask_iou_scores = all_iou_scores[:, 0:1]
    is_stable = score_stability(singlemask_logits, delta) >= threshold

    mask_logits = torch.where(
        is_stable[..., None, None].expand_as(singlemask_logits),
        singlemask_logits,
        best_multimask_logits,
    )
    iou_scores = torch.where(
        is_stable.expand_as(singlemask_iou_scores),
        singlemask_iou_scores,
        best_multimask_iou_scores,
    )

    return (
        mask_logits.unflatten(0, (batch_size, multiplex_count)),
        iou_scores.unflatten(0, (batch_size, multiplex_count)),
    )
