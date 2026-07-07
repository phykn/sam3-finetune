import numpy as np
from PIL import Image

from .boxes import find_box, scale_area, scale_box


def make_candidate(mask, logit, score, point, crop, tile, crop_index, full_size):
    local_box = find_box(mask)
    if local_box is None:
        return None
    x0, y0, x1, y1 = local_box
    crop_x, crop_y, _crop_x1, _crop_y1 = crop
    bbox = scale_box(local_box, mask.shape, crop)
    return {
        "segmentation": mask[y0:y1, x0:x1].copy(),
        "logit": logit[y0:y1, x0:x1].copy(),
        "bbox": bbox,
        "low_box": local_box,
        "low_shape": mask.shape,
        "area": scale_area(mask.sum(), mask.shape, crop),
        "score": float(score),
        "stability_score": score_stability(logit),
        "point": (float(point[0] + crop_x), float(point[1] + crop_y)),
        "crop": crop,
        "tile": int(tile),
        "crop_index": int(crop_index),
        "image_size": full_size,
    }


def expand_mask(item, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = item["bbox"]
    seg = item["segmentation"]
    target = (y1 - y0, x1 - x0)
    if item.get("logit") is not None:
        seg = resize_logit(item["logit"], (target[1], target[0])) > 0
    elif seg.shape != target:
        seg = resize_mask(seg, (target[1], target[0]))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = seg
    return mask


def resize_mask(mask, size):
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 127


def resize_logit(logit, size):
    image = Image.fromarray(logit.astype(np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR))


def score_stability(logit):
    high = logit > 1.0
    low = logit > -1.0
    union = int(low.sum())
    if union == 0:
        return 0.0
    return float(high.sum() / union)


def format_masks(masks):
    return format_logits(masks).astype(bool)


def format_logits(logits):
    logits = np.asarray(logits)
    if logits.ndim == 4 and logits.shape[1] == 1:
        return logits[:, 0]
    return logits
