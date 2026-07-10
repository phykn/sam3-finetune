import numpy as np
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
):
    items = []
    for batch, class_id in enumerate(class_ids):
        scores = out["pred_logits"][batch, :, 0].float().sigmoid()
        keep = torch.nonzero(scores >= score_thr).flatten()
        if keep.numel() == 0:
            continue

        logits = out["pred_masks"][batch, keep].float()
        masks = (
            F.interpolate(
                logits[:, None],
                orig_hw,
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            > 0
        )
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
        for index in torch.nonzero(valid).flatten().tolist():
            items.append(
                {
                    "class_id": int(class_id),
                    "nms_box": tuple(float(value) for value in boxes[index]),
                    "mask": masks[index].detach().cpu().numpy(),
                    "logit": logits[index].detach().cpu().numpy(),
                    "metrics": {
                        "score": float(scores[keep[index]].item()),
                        "similarity": float(similarities[index].item()),
                    },
                }
            )
    return items


def finish(items, nms_thr, top_k):
    selected = []
    for class_id in sorted({item["class_id"] for item in items}):
        group = [item for item in items if item["class_id"] == class_id]
        keep = nms_indices(
            np.asarray([item["nms_box"] for item in group]),
            np.asarray([item["metrics"]["score"] for item in group]),
            nms_thr,
        )
        if top_k is not None:
            keep = keep[:top_k]
        selected.extend(group[index] for index in keep)

    out = []
    for object_id, item in enumerate(selected, start=1):
        item = dict(item)
        item.pop("nms_box")
        item["box"], roi = pack.box_roi(item.pop("mask"))
        item["roi"] = roi.astype(bool)
        item["object_id"] = object_id
        out.append(item)
    return out
