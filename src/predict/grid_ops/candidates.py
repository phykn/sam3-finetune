import numpy as np
from PIL import Image

from .boxes import find_box, scale_area, scale_box


def make_candidate(
    mask: np.ndarray,
    logit: np.ndarray,
    score: float,
    point: np.ndarray,
    crop: tuple[int, int, int, int],
    tile: int,
    crop_index: int,
    full_size: tuple[int, int],
) -> dict[str, object] | None:
    local_box = find_box(mask)
    if local_box is None:
        return None
    crop_x, crop_y, _crop_x1, _crop_y1 = crop
    bbox = scale_box(local_box, mask.shape, crop)
    return {
        "logit": logit.astype(np.float16, copy=True),
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


def make_objects(items: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for item in items:
        crop_x0, crop_y0, crop_x1, crop_y1 = item["crop"]
        mask = (
            resize_logit(
                item["logit"],
                (crop_x1 - crop_x0, crop_y1 - crop_y0),
            )
            > 0
        )
        local_box = find_box(mask)
        if local_box is None:
            continue
        x0, y0, x1, y1 = local_box
        box = (crop_x0 + x0, crop_y0 + y0, crop_x0 + x1, crop_y0 + y1)
        roi = mask[y0:y1, x0:x1]
        metrics = {
            "score": float(item["score"]),
            "stability": float(item["stability_score"]),
        }
        for key in ("class_logits", "class_scores"):
            if key in item:
                metrics[key] = np.asarray(item[key], dtype=float).tolist()
        out.append(
            {
                "object_id": len(out) + 1,
                "class_id": None,
                "box": box,
                "roi": roi,
                "points": [[float(item["point"][0]), float(item["point"][1]), 1]],
                "metrics": metrics,
            }
        )
    return out


def resize_logit(logit: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(logit.astype(np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR))


def score_stability(logit: np.ndarray) -> float:
    high = logit > 1.0
    low = logit > -1.0
    union = int(low.sum())
    if union == 0:
        return 0.0
    return float(high.sum() / union)


def format_masks(masks: object) -> np.ndarray:
    return format_logits(masks).astype(bool)


def format_logits(logits: object) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 4 and logits.shape[1] == 1:
        return logits[:, 0]
    return logits
