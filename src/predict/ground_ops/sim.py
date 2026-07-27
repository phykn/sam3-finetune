import numpy as np
import torch
import torch.nn.functional as F


def box_vectors(
    image: dict[str, object],
    boxes: object,
    orig_hw: tuple[int, int],
) -> torch.Tensor:
    out = []
    for value in image["backbone_fpn"]:
        feat = tensor(value).float()
        box = torch.as_tensor(boxes, dtype=torch.float32, device=feat.device)
        feat_h, feat_w = feat.shape[-2:]
        orig_h, orig_w = orig_hw
        scale = box.new_tensor([feat_w / orig_w, feat_h / orig_h] * 2)
        box = box * scale
        start = box[:, :2].floor()
        end = box[:, 2:].ceil()
        x = torch.arange(feat_w, device=feat.device)[None, None]
        y = torch.arange(feat_h, device=feat.device)[None, :, None]
        masks = (
            (x >= start[:, 0, None, None])
            & (x < end[:, 0, None, None])
            & (y >= start[:, 1, None, None])
            & (y < end[:, 1, None, None])
        ).float()
        out.append(_vectors(feat, masks))
    return F.normalize(torch.cat(out, dim=-1), dim=-1)


def mask_vectors(image: dict[str, object], masks: object) -> torch.Tensor:
    feat = tensor(image["backbone_fpn"][-1]).float()
    if isinstance(masks, torch.Tensor):
        masks = masks.to(device=feat.device, dtype=torch.float32)
    else:
        masks = torch.as_tensor(
            np.asarray(masks),
            dtype=torch.float32,
            device=feat.device,
        )
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim == 3:
        masks = masks[:, None]

    out = []
    for value in image["backbone_fpn"]:
        feat = tensor(value).float()
        scaled = F.interpolate(
            masks,
            feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        out.append(_vectors(feat, scaled[:, 0]))
    return F.normalize(torch.cat(out, dim=-1), dim=-1)


def _vectors(feat: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 3:
        masks = masks[:, None]
    if feat.shape[0] == 1 and masks.shape[0] != 1:
        feat = feat.expand(masks.shape[0], -1, -1, -1)

    denom = masks.sum(dim=(-2, -1)).clamp_min(1e-6)
    out = (feat * masks).sum(dim=(-2, -1)) / denom
    return F.normalize(out, dim=-1)


def max_scores(ref: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if ref.ndim == 1:
        return (target * ref[None]).sum(dim=-1)
    return (target[:, None] * ref[None]).sum(dim=-1).max(dim=1).values


def tensor(value: object) -> torch.Tensor:
    return getattr(value, "tensors", value)
