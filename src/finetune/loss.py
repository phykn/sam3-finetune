import torch
import torch.nn.functional as F


def _num_boxes(value: int | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device).clamp_min(1)
    return torch.tensor(float(max(value, 1)), device=device)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: int | torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduce: bool = True,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pt = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - pt) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if not reduce:
        return loss
    return loss.mean(1).sum() / _num_boxes(num_boxes, inputs.device)


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: int | torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if not reduce:
        return loss
    return loss.sum() / _num_boxes(num_boxes, inputs.device)


def iou_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pred_ious: torch.Tensor,
    num_boxes: int | torch.Tensor,
    use_l1: bool = False,
) -> torch.Tensor:
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)
    if use_l1:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    return loss.sum() / _num_boxes(num_boxes, inputs.device)


def auto_bg_label_weight(
    weights: torch.Tensor,
    object_logits: torch.Tensor,
    is_auto_bg: torch.Tensor,
) -> torch.Tensor:
    out = weights.clone()
    if out.shape[-1] == 0:
        return out

    is_auto_bg = is_auto_bg.to(device=out.device, dtype=torch.bool).flatten()
    object_prob = object_logits.detach().sigmoid().flatten().to(device=out.device)
    if object_prob.numel() == 1 and out.shape[0] > 1:
        object_prob = object_prob.repeat(out.shape[0])
    out[is_auto_bg, 0] = out[is_auto_bg, 0] * (1.0 - object_prob[is_auto_bg])
    return out


def noisy_mask_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    brier = (prob - targets).flatten(1).pow(2).mean(1)
    dice = dice_loss(inputs, targets, num_boxes=1, reduce=False)
    loss = brier + dice
    if weight is None:
        return loss.mean()

    weight = weight.to(device=loss.device, dtype=loss.dtype).flatten()
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def label_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)
