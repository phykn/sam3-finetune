import torch


def score_stability(mask_logits, delta):
    mask_logits = mask_logits.flatten(-2)
    area_i = torch.sum(mask_logits > delta, dim=-1).float()
    area_u = torch.sum(mask_logits > -delta, dim=-1).float()
    return torch.where(area_u > 0, area_i / area_u, 1.0)


def select_dynamic_multimask(logits, iou, delta, threshold):
    batch_size, multiplex_count = logits.shape[:2]
    logits = logits.flatten(0, 1)
    iou = iou.flatten(0, 1)

    multimask_logits = logits[:, 1:, :, :]
    multimask_iou = iou[:, 1:]
    best = torch.argmax(multimask_iou, dim=-1)
    batch = torch.arange(multimask_iou.size(0), device=iou.device)
    best_logits = multimask_logits[batch, best].unsqueeze(1)
    best_iou = multimask_iou[batch, best].unsqueeze(1)

    single_logits = logits[:, 0:1, :, :]
    single_iou = iou[:, 0:1]
    is_stable = score_stability(single_logits, delta) >= threshold

    mask_logits = torch.where(
        is_stable[..., None, None].expand_as(single_logits),
        single_logits,
        best_logits,
    )
    iou_scores = torch.where(
        is_stable.expand_as(single_iou),
        single_iou,
        best_iou,
    )

    return (
        mask_logits.unflatten(0, (batch_size, multiplex_count)),
        iou_scores.unflatten(0, (batch_size, multiplex_count)),
    )
