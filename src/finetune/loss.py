import torch
import torch.nn.functional as F

from .ddp import sum_value, world_size


def mask_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return loss.flatten(1).mean(1)


def mask_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid().flatten(1)
    target = target.flatten(1)
    overlap = 2 * (probability * target).sum(1)
    total = probability.sum(1) + target.sum(1)
    return 1 - (overlap + 1) / (total + 1)


@torch.no_grad()
def target_iou(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted = logits.flatten(2) > 0
    expected = target.flatten(2) > 0.5
    intersection = (predicted & expected).sum(-1).float()
    union = (predicted | expected).sum(-1).float()
    return intersection / union.clamp_min(1)


def class_weights(
    weights: torch.Tensor,
    logits: torch.Tensor,
    is_auto_bg: torch.Tensor,
) -> torch.Tensor:
    out = weights.detach().clone()
    auto = is_auto_bg.to(device=out.device, dtype=torch.bool).flatten()
    particle = logits[:, 0].detach().sigmoid()
    out[auto, 0] *= 1 - particle[auto]
    return out


def mean_loss(
    local_sum: torch.Tensor,
    local_weight: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    weight = torch.as_tensor(
        local_weight,
        device=local_sum.device,
        dtype=local_sum.dtype,
    )
    global_weight = sum_value(weight)
    if global_weight.item() <= 0:
        return local_sum * 0, 0.0
    backward = local_sum * (world_size() / global_weight)
    logged = sum_value(local_sum) / global_weight
    return backward, float(logged.cpu())


@torch.no_grad()
def class_stats(
    loss: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    is_auto_bg: torch.Tensor,
) -> dict[str, float]:
    known = weights.detach().clone()
    auto = is_auto_bg.to(device=known.device, dtype=torch.bool).flatten()
    known[auto] = 0
    correct = ((logits >= 0) == (target >= 0.5)).to(weights)
    stats = {}
    for index in range(logits.shape[1]):
        _, loss_value = mean_loss(loss[:, index].sum(), weights[:, index].sum())
        _, acc_value = mean_loss(
            (correct[:, index] * known[:, index]).sum(),
            known[:, index].sum(),
        )
        stats[f"class_loss_{index}"] = loss_value
        stats[f"class_acc_{index}"] = acc_value
    return stats


def finetune_loss(
    batch: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    mask_logits = out["mask_logits"].float()
    iou_scores = out["iou_scores"].float()
    class_logits = out["class_logits"].float()
    target = batch["target"].float()
    mask_valid = batch["mask_valid"].to(mask_logits).flatten()

    if mask_logits.shape != target.shape:
        raise ValueError("mask target shape must match mask logits")
    if iou_scores.shape != mask_logits.shape[:2]:
        raise ValueError("IoU scores must align with masks")
    if class_logits.shape[:2] != mask_logits.shape[:2]:
        raise ValueError("class logits must align with masks")

    bce_sum = (mask_bce(mask_logits, target) * mask_valid).sum()
    dice_sum = (mask_dice(mask_logits, target) * mask_valid).sum()
    actual_iou = target_iou(mask_logits, target)
    iou_per_sample = F.mse_loss(
        iou_scores,
        actual_iou,
        reduction="none",
    ).mean(1)
    iou_sum = (iou_per_sample * mask_valid).sum()

    class_logits = class_logits[:, 0]
    class_target = batch["label_target"].float()
    if class_logits.shape != class_target.shape:
        raise ValueError("class target shape must match class logits")
    base_weights = batch["label_weight"].float()
    weights = class_weights(
        base_weights,
        class_logits,
        batch["is_auto_bg"],
    )
    class_loss = (
        F.binary_cross_entropy_with_logits(
            class_logits,
            class_target,
            reduction="none",
        )
        * weights
    )
    class_sum = class_loss.sum()
    stats = class_stats(
        class_loss,
        class_logits,
        class_target,
        base_weights,
        batch["is_auto_bg"],
    )

    bce, bce_value = mean_loss(bce_sum, mask_valid.sum())
    dice, dice_value = mean_loss(dice_sum, mask_valid.sum())
    iou, iou_value = mean_loss(iou_sum, mask_valid.sum())
    classes, class_value = mean_loss(class_sum, base_weights.sum())
    total = bce + dice + iou + classes
    stats.update(
        {
            "loss": bce_value + dice_value + iou_value + class_value,
            "mask_bce": bce_value,
            "mask_dice": dice_value,
            "iou_loss": iou_value,
            "class_loss": class_value,
        }
    )
    return total, stats
