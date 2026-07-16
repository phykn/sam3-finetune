import torch
import torch.nn.functional as F

from ...data import pack


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


def make_low(
    masks: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    classes: torch.Tensor | None = None,
) -> dict[str, object]:
    logits = masks.clamp(-32.0, 32.0).float()
    out = {
        "masks": (logits > threshold).detach().cpu().numpy(),
        "scores": scores.float().detach().cpu().numpy(),
        "logits": logits.detach().cpu().numpy(),
    }
    if classes is not None:
        classes = classes.float().detach().cpu()
        out["class_logits"] = classes.numpy()
        out["class_scores"] = classes.sigmoid().numpy()
    return out


def make_objects(
    masks: torch.Tensor,
    scores: torch.Tensor,
    orig_hw: tuple[int, int],
    classes: torch.Tensor | None = None,
) -> list[dict[str, object]]:
    logits = masks.clamp(-32.0, 32.0).float().detach().cpu().numpy()
    masks = resize_masks(masks, orig_hw, 0.0).detach().cpu().numpy()
    scores = scores.float().detach().cpu().numpy()
    class_logits = None
    class_scores = None
    if classes is not None:
        classes = classes.float().detach().cpu()
        class_logits = classes.numpy()
        class_scores = classes.sigmoid().numpy()

    out = []
    for prompt_index in range(masks.shape[0]):
        for candidate_index in range(masks.shape[1]):
            box, roi = pack.box_roi(masks[prompt_index, candidate_index])
            if roi.size == 0:
                continue
            metrics = {"score": float(scores[prompt_index, candidate_index])}
            if class_logits is not None:
                metrics["class_logits"] = (
                    class_logits[prompt_index, candidate_index].astype(float).tolist()
                )
                metrics["class_scores"] = (
                    class_scores[prompt_index, candidate_index].astype(float).tolist()
                )
            out.append(
                {
                    "object_id": len(out) + 1,
                    "class_id": None,
                    "box": box,
                    "roi": roi.astype(bool),
                    "logit": logits[prompt_index, candidate_index].copy(),
                    "prompt_index": prompt_index,
                    "candidate_index": candidate_index,
                    "metrics": metrics,
                }
            )
    return out
