import torch
import torch.nn.functional as F

from ...data import pack
from ...ops.box import cxcywh_to_xyxy, nms_indices
from . import sim


def candidates(
    out,
    image,
    class_ids,
    bank,
    orig_hw,
    score_thr,
    sim_thr,
    negative_bank=None,
    negative_margin=0.0,
):
    items = []
    for batch, class_id in enumerate(class_ids):
        scores = out["pred_logits"][batch, :, 0].float().sigmoid()
        keep = torch.nonzero(scores >= score_thr).flatten()
        if keep.numel() == 0:
            continue

        logits = out["pred_masks"][batch, keep].float()
        masks = logits > 0
        valid = masks.flatten(1).any(1)
        if not valid.any():
            continue
        keep = keep[valid]
        logits = logits[valid]
        masks = masks[valid]

        boxes = cxcywh_to_xyxy(out["pred_boxes"][batch, keep].float())
        height, width = orig_hw
        boxes = boxes * boxes.new_tensor([width, height, width, height])
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)

        vectors = sim.mask_vectors(image, masks)
        similarities = sim.max_scores(bank[int(class_id)], vectors)
        valid = similarities >= sim_thr
        negative = None if negative_bank is None else negative_bank.get(int(class_id))
        negative_similarities = None
        if negative is not None:
            negative_similarities = sim.max_scores(negative, vectors)
            valid &= similarities > negative_similarities + negative_margin
        keep = keep[valid]
        logits = logits[valid]
        boxes = boxes[valid]
        similarities = similarities[valid]
        if negative_similarities is not None:
            negative_similarities = negative_similarities[valid]
        scores = scores[keep].detach().cpu()
        boxes = boxes.detach().cpu()
        logits = logits.detach().cpu()
        similarities = similarities.detach().cpu()
        if negative_similarities is not None:
            negative_similarities = negative_similarities.detach().cpu()
        for index in range(len(keep)):
            metrics = {
                "score": scores[index],
                "similarity": similarities[index],
            }
            if negative_similarities is not None:
                metrics["negative_similarity"] = negative_similarities[index]
            items.append(
                {
                    "class_id": int(class_id),
                    "nms_box": boxes[index],
                    "logit": logits[index],
                    "metrics": metrics,
                }
            )
    return items


def finish(items, nms_thr, top_k, orig_hw, mask_batch_size, device):
    selected = []
    for class_id in sorted({item["class_id"] for item in items}):
        group = [item for item in items if item["class_id"] == class_id]
        keep = nms_indices(
            torch.stack([item["nms_box"] for item in group]).cpu(),
            torch.stack([item["metrics"]["score"] for item in group]).cpu(),
            nms_thr,
        )
        if top_k is not None:
            keep = keep[:top_k]
        selected.extend(group[index] for index in keep)

    if not selected:
        return []

    out = []
    for start in range(0, len(selected), mask_batch_size):
        chunk = selected[start : start + mask_batch_size]
        logits = torch.stack([item["logit"] for item in chunk]).float().cpu()
        masks = (
            F.interpolate(
                logits.to(device)[:, None],
                orig_hw,
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            > 0
        ).cpu()
        valid = torch.nonzero(masks.flatten(1).any(1)).flatten().tolist()
        chunk = [chunk[index] for index in valid]
        if not chunk:
            continue

        logits = logits[valid].numpy()
        masks = masks[valid].numpy()
        for index, (item, mask, logit) in enumerate(
            zip(chunk, masks, logits, strict=True)
        ):
            box, roi = pack.box_roi(mask)
            out.append(
                {
                    "class_id": item["class_id"],
                    "box": box,
                    "roi": roi.astype(bool),
                    "logit": logit,
                    "metrics": {
                        key: float(value) for key, value in item["metrics"].items()
                    },
                    "object_id": len(out) + 1,
                }
            )
    return out
