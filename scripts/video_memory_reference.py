
import gc
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ReferenceMaskResult:
    mask: np.ndarray
    source: str
    score: float | None
    selected_index: int | None
    refined_score: float | None = None


def make_box_mask(image: Image.Image, box: Sequence[int] | None) -> np.ndarray:
    x0, y0, x1, y1 = resolve_box(image, box)
    mask = np.zeros((image.height, image.width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def resolve_box(image: Image.Image, box: Sequence[int] | None) -> list[int]:
    width, height = image.size
    if box is None:
        x0 = int(width * 0.25)
        y0 = int(height * 0.20)
        x1 = int(width * 0.75)
        y1 = int(height * 0.85)
    else:
        x0, y0, x1, y1 = [int(value) for value in box]
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    return [x0, y0, x1, y1]


def select_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    mask_arr = np.asarray(masks)
    score_arr = np.asarray(scores, dtype=np.float32)
    if mask_arr.ndim < 3:
        raise ValueError("masks must have candidate and spatial dimensions")

    candidate_count = int(np.prod(mask_arr.shape[:-2]))
    flat_scores = score_arr.reshape(-1)
    if flat_scores.size != candidate_count:
        raise ValueError(
            f"score count {flat_scores.size} does not match mask count {candidate_count}"
        )

    flat_masks = mask_arr.reshape(candidate_count, *mask_arr.shape[-2:])
    selected_index = int(np.argmax(flat_scores))
    return (
        flat_masks[selected_index].astype(bool),
        float(flat_scores[selected_index]),
        selected_index,
    )


def build_reference_mask(
    image: Image.Image,
    box: Sequence[int] | None,
    source: str,
    checkpoint: str | Path | None,
    device: str,
) -> ReferenceMaskResult:
    if source == "box":
        return ReferenceMaskResult(
            mask=make_box_mask(image, box),
            source="box",
            score=None,
            selected_index=None,
        )
    if source != "sam":
        raise ValueError("source must be 'box' or 'sam'")
    if checkpoint is None:
        raise ValueError("checkpoint is required when source='sam'")
    return predict_sam_mask_from_box(
        image=image,
        box=resolve_box(image, box),
        checkpoint=checkpoint,
        device=device,
    )


def make_point_prompt_arrays(
    positive_points: Sequence[Sequence[float]] | None,
    negative_points: Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    points: list[list[float]] = []
    labels: list[int] = []
    for point in positive_points or ():
        if len(point) != 2:
            raise ValueError("points must contain x and y")
        points.append([float(point[0]), float(point[1])])
        labels.append(1)
    for point in negative_points or ():
        if len(point) != 2:
            raise ValueError("points must contain x and y")
        points.append([float(point[0]), float(point[1])])
        labels.append(0)
    if not points:
        return None, None
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def predict_sam_mask_from_prompts(
    image: Image.Image,
    *,
    checkpoint: str | Path,
    device: str,
    positive_points: Sequence[Sequence[float]] | None = None,
    negative_points: Sequence[Sequence[float]] | None = None,
    box: Sequence[int] | None = None,
) -> ReferenceMaskResult:
    import torch

    from src.predictor import Sam3Predictor

    point_coords, point_labels = make_point_prompt_arrays(
        positive_points=positive_points,
        negative_points=negative_points,
    )
    box_array = None if box is None else np.asarray(resolve_box(image, box), dtype=np.float32)
    if point_coords is None and box_array is None:
        raise ValueError("at least one point or box prompt is required")

    predictor = None
    try:
        predictor = Sam3Predictor.from_checkpoint(checkpoint, device=device)
        device_type = torch.device(device).type
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device_type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            predictor.set_image(image)
            masks, scores, low_res_masks = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_array,
                multimask_output=True,
            )
            mask, score, selected_index = select_best_mask(masks, scores)
            low_res = np.asarray(low_res_masks).reshape(
                -1, *np.asarray(low_res_masks).shape[-2:]
            )[selected_index]
            refined_masks, refined_scores, _refined_low_res = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_array,
                mask_input=low_res,
                multimask_output=False,
            )
            refined_mask, refined_score, _refined_index = select_best_mask(
                refined_masks,
                refined_scores,
            )
        return ReferenceMaskResult(
            mask=refined_mask,
            source="sam_prompt",
            score=score,
            selected_index=selected_index,
            refined_score=refined_score,
        )
    finally:
        del predictor
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def predict_sam_mask_from_box(
    image: Image.Image,
    box: Sequence[int],
    checkpoint: str | Path,
    device: str,
) -> ReferenceMaskResult:
    import torch

    from src.predictor import Sam3Predictor

    predictor = None
    try:
        predictor = Sam3Predictor.from_checkpoint(checkpoint, device=device)
        device_type = torch.device(device).type
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device_type == "cuda"
            else nullcontext()
        )
        box_array = np.asarray(box, dtype=np.float32)
        with autocast_context:
            predictor.set_image(image)
            masks, scores, low_res_masks = predictor.predict(
                box=box_array,
                multimask_output=True,
            )
            mask, score, selected_index = select_best_mask(masks, scores)
            low_res = np.asarray(low_res_masks).reshape(
                -1, *np.asarray(low_res_masks).shape[-2:]
            )[selected_index]
            refined_masks, refined_scores, _refined_low_res = predictor.predict(
                box=box_array,
                mask_input=low_res,
                multimask_output=False,
            )
            refined_mask, refined_score, _refined_index = select_best_mask(
                refined_masks,
                refined_scores,
            )
        return ReferenceMaskResult(
            mask=refined_mask,
            source="sam",
            score=score,
            selected_index=selected_index,
            refined_score=refined_score,
        )
    finally:
        del predictor
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
