import numpy as np
import torch
import torch.nn.functional as F


def vectors(image, masks):
    feat = tensor(image["backbone_fpn"][-1]).float()
    masks = torch.as_tensor(np.asarray(masks), dtype=torch.float32, device=feat.device)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim == 3:
        masks = masks[:, None]

    masks = F.interpolate(
        masks,
        feat.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0, 1)
    if feat.shape[0] == 1 and masks.shape[0] != 1:
        feat = feat.expand(masks.shape[0], -1, -1, -1)

    denom = masks.sum(dim=(-2, -1)).clamp_min(1e-6)
    out = (feat * masks).sum(dim=(-2, -1)) / denom
    return F.normalize(out, dim=-1)


def scores(ref, target):
    if ref.ndim == 1:
        return (target * ref[None]).sum(dim=-1)
    return (target[:, None] * ref[None]).sum(dim=-1).max(dim=1).values


def select(similarities, scores, threshold, top_k):
    similarities = np.asarray(similarities)
    scores = np.asarray(scores)
    keep = np.flatnonzero(similarities >= threshold)
    keep = keep[np.argsort(-scores[keep])]
    if top_k is not None:
        keep = keep[:top_k]
    return keep


def tensor(value):
    return getattr(value, "tensors", value)
