import torch
import torch.nn.functional as F


def resize_masks(
    masks: torch.Tensor,
    orig_hw: tuple[int, int],
    threshold: float,
) -> torch.Tensor:
    return (
        F.interpolate(
            masks.float(),
            orig_hw,
            mode="bilinear",
            align_corners=False,
        )
        > threshold
    )


def make_full(
    masks: torch.Tensor,
    scores: torch.Tensor,
    orig_hw: tuple[int, int],
    threshold: float,
):
    logits = masks.clamp(-32.0, 32.0).float()
    return {
        "masks": resize_masks(masks, orig_hw, threshold)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy(),
        "scores": scores.squeeze(0).float().detach().cpu().numpy(),
        "logits": logits.squeeze(0).detach().cpu().numpy(),
    }


def make_low(
    masks: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
):
    logits = masks.clamp(-32.0, 32.0).float()
    return {
        "masks": (logits > threshold).squeeze(0).detach().cpu().numpy(),
        "scores": scores.squeeze(0).float().detach().cpu().numpy(),
        "logits": logits.squeeze(0).detach().cpu().numpy(),
    }
