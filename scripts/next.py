import argparse
import gc
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict.refine import select_best_mask
from src.types import MemoryPrediction, MemoryReference

DEFAULT_FRAMES = (
    "asset/heli_3.jpg",
    "asset/heli_2.jpg",
    "asset/heli_1.jpg",
)
OBJ_ID = 1
REFERENCE_BOX = (135.0, 180.0, 545.0, 425.0)
REFERENCE_POINT = (285.0, 350.0)
MASK_COLORS = (
    (255, 0, 110),
    (67, 97, 238),
    (255, 183, 3),
)
PROMPT_COLOR = (0, 180, 216)


@dataclass
class NextPaths:
    frames: tuple[Path, ...]
    checkpoint: Path
    output_dir: Path

    @property
    def output(self) -> Path:
        return self.output_dir / "next.png"


@dataclass
class ReferenceMaskResult:
    mask: np.ndarray
    score: float
    selected_index: int
    refined_score: float


@dataclass
class FrameResult:
    sequence_index: int
    image: Image.Image
    prediction: MemoryPrediction
    mask: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a helicopter mask from one frame to the next.",
        allow_abbrev=False,
    )
    parser.add_argument("--frames", nargs="+", default=list(DEFAULT_FRAMES))
    parser.add_argument("--checkpoint", default="weight/sam3.1_multiplex.pt")
    parser.add_argument("--output-dir", default="outputs/next")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace, *, root: Path = ROOT) -> NextPaths:
    return NextPaths(
        frames=tuple(root / frame for frame in args.frames),
        checkpoint=root / args.checkpoint,
        output_dir=root / args.output_dir,
    )


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when --device cuda is selected.")
    return device


def reference_box_array() -> np.ndarray:
    return np.asarray(REFERENCE_BOX, dtype=np.float32)


def reference_point_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([REFERENCE_POINT], dtype=np.float32),
        np.asarray([1], dtype=np.int64),
    )


def describe_reference_prompt() -> dict[str, object]:
    box = reference_box_array()
    point_coords, point_labels = reference_point_arrays()
    x0, y0, x1, y1 = box.tolist()
    return {
        "type": "point_box",
        "box": {
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
        },
        "points": [
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "label": int(label),
            }
            for point, label in zip(point_coords.tolist(), point_labels.tolist())
        ],
    }


def predict_reference_mask(predictor, image: Image.Image) -> ReferenceMaskResult:
    box = reference_box_array()
    point_coords, point_labels = reference_point_arrays()
    embedding = predictor.encode_image(image)
    masks, scores, low_res_masks = predictor.predict_from_embedding(
        embedding,
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )
    _mask, score, selected_index = select_best_mask(masks, scores)
    flat_low_res = np.asarray(low_res_masks).reshape(
        -1,
        *np.asarray(low_res_masks).shape[-2:],
    )
    refined_masks, refined_scores, _refined_low_res = predictor.predict_from_embedding(
        embedding=embedding,
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        mask_input=flat_low_res[selected_index],
        multimask_output=False,
    )
    refined_mask, refined_score, _refined_index = select_best_mask(
        refined_masks,
        refined_scores,
    )
    return ReferenceMaskResult(
        mask=refined_mask.astype(bool),
        score=float(score),
        selected_index=int(selected_index),
        refined_score=float(refined_score),
    )


def predict_sequence(
    predictor,
    images: list[Image.Image],
    reference_mask: np.ndarray,
) -> list[FrameResult]:
    references = [
        MemoryReference(
            image=images[0],
            mask=reference_mask,
            obj_id=OBJ_ID,
        )
    ]
    results = []
    for sequence_index, image in enumerate(images[1:], start=2):
        prediction = predictor.predict(
            target_image=image,
            references=references,
        )
        mask = prediction_mask(prediction, obj_id=OBJ_ID)
        results.append(
            FrameResult(
                sequence_index=sequence_index,
                image=image,
                prediction=prediction,
                mask=mask,
            )
        )
        references.append(
            MemoryReference(
                image=image,
                mask=mask,
                obj_id=OBJ_ID,
            )
        )
    return results


def prediction_mask(prediction: MemoryPrediction, obj_id: int = OBJ_ID) -> np.ndarray:
    if obj_id not in prediction.obj_ids:
        raise ValueError(f"obj_id {obj_id} is not present in prediction")
    object_index = prediction.obj_ids.index(obj_id)
    mask = np.asarray(prediction.masks).astype(bool)[object_index]
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3 and mask.shape[0] == 1:
        return mask[0]
    return mask.reshape(-1, *mask.shape[-2:])[0]


def prediction_score(prediction: MemoryPrediction, obj_id: int = OBJ_ID) -> float:
    if obj_id not in prediction.obj_ids:
        raise ValueError(f"obj_id {obj_id} is not present in prediction")
    object_index = prediction.obj_ids.index(obj_id)
    scores = np.asarray(prediction.scores, dtype=np.float32).reshape(
        len(prediction.obj_ids),
        -1,
    )
    return float(scores[object_index, 0])


def summarize_reference(result: ReferenceMaskResult) -> dict[str, object]:
    return {
        "area": int(result.mask.sum()),
        "bbox": mask_to_box(result.mask),
        "score": float(result.score),
        "selected_index": int(result.selected_index),
        "refined_score": float(result.refined_score),
    }


def summarize_prediction(
    prediction: MemoryPrediction,
    obj_id: int = OBJ_ID,
) -> dict[str, object]:
    mask = prediction_mask(prediction, obj_id=obj_id)
    return {
        "frame_index": int(prediction.frame_index),
        "obj_id": int(obj_id),
        "bbox": mask_to_box(mask),
        "area": int(mask.sum()),
        "score": prediction_score(prediction, obj_id=obj_id),
    }


def save_next_visualization(
    images: list[Image.Image],
    reference_result: ReferenceMaskResult,
    frame_results: list[FrameResult],
    path: Path,
) -> None:
    panel_width = 400
    gap = 14
    padding = 16
    panels = [
        _make_reference_panel(images[0], reference_result.mask, panel_width),
        *[
            _make_prediction_panel(result, panel_width, index + 1)
            for index, result in enumerate(frame_results)
        ],
    ]
    canvas_width = (
        padding * 2 + sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    )
    canvas_height = padding * 2 + max(panel.height for panel in panels)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (246, 246, 242))
    x = padding
    for panel in panels:
        canvas.paste(panel, (x, padding))
        x += panel.width + gap
    canvas.save(path)


def _make_reference_panel(
    image: Image.Image,
    mask: np.ndarray,
    width: int,
) -> Image.Image:
    body = _resize_image(image, width)
    overlay = _overlay_mask(body, _resize_mask(mask, body.size), MASK_COLORS[0], 95)
    draw = ImageDraw.Draw(overlay)
    scale_x = body.width / image.width
    scale_y = body.height / image.height
    draw.rectangle(
        _scale_bbox(REFERENCE_BOX, scale_x, scale_y),
        outline=(*PROMPT_COLOR, 255),
        width=2,
    )
    point_x = REFERENCE_POINT[0] * scale_x
    point_y = REFERENCE_POINT[1] * scale_y
    draw.ellipse(
        (point_x - 5, point_y - 5, point_x + 5, point_y + 5),
        outline=(*PROMPT_COLOR, 255),
        width=2,
    )
    draw.line((point_x - 8, point_y, point_x + 8, point_y), fill=(*PROMPT_COLOR, 255))
    draw.line((point_x, point_y - 8, point_x, point_y + 8), fill=(*PROMPT_COLOR, 255))
    return _with_header(overlay.convert("RGB"), "frame 1 reference")


def _make_prediction_panel(
    result: FrameResult,
    width: int,
    color_index: int,
) -> Image.Image:
    body = _resize_image(result.image, width)
    color = MASK_COLORS[color_index % len(MASK_COLORS)]
    overlay = _overlay_mask(body, _resize_mask(result.mask, body.size), color, 95)
    scale_x = body.width / result.image.width
    scale_y = body.height / result.image.height
    bbox = mask_to_box(result.mask)
    if bbox is not None:
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(
            _scale_bbox(bbox, scale_x, scale_y), outline=(*color, 255), width=2
        )
        x0, y0, _x1, _y1 = _scale_bbox(bbox, scale_x, scale_y)
        score = prediction_score(result.prediction, obj_id=OBJ_ID)
        draw.text(
            (x0 + 3, max(0, y0 - 13)),
            f"obj={OBJ_ID} score={score:.2f}",
            fill=(*color, 255),
        )
    return _with_header(
        overlay.convert("RGB"),
        f"frame {result.sequence_index} predicted",
    )


def _resize_image(image: Image.Image, width: int) -> Image.Image:
    height = max(1, int(round(image.height * width / image.width)))
    return image.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                size,
                Image.Resampling.NEAREST,
            )
        )
        > 127
    )


def _overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: int,
) -> Image.Image:
    overlay = image.convert("RGBA")
    layer = Image.new("RGBA", overlay.size, (*color, 255))
    layer.putalpha(Image.fromarray(mask.astype(np.uint8) * alpha, mode="L"))
    return Image.alpha_composite(overlay, layer)


def _scale_bbox(
    bbox: tuple[float, float, float, float] | tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = np.asarray(bbox, dtype=np.float32).tolist()
    return (
        int(round(x0 * scale_x)),
        int(round(y0 * scale_y)),
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
    )


def mask_to_box(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _with_header(body: Image.Image, title: str) -> Image.Image:
    header_height = 32
    panel = Image.new("RGB", (body.width, body.height + header_height), (30, 30, 30))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, header_height), fill=(246, 246, 242))
    draw.text((10, 9), title, fill=(30, 30, 30))
    panel.paste(body.convert("RGB"), (0, header_height))
    return panel


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    device = resolve_device(args.device)
    if len(paths.frames) < 2:
        raise ValueError("at least two frames are required")
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    from src.predict.next_frame import NextFramePredictor
    from src.predict.prompted import Sam3Predictor

    images = [Image.open(path).convert("RGB") for path in paths.frames]
    started_at = time.perf_counter()
    image_predictor = None
    try:
        image_predictor = Sam3Predictor.from_checkpoint(paths.checkpoint, device=device)
        device_type = torch.device(device).type
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device_type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            reference_result = predict_reference_mask(image_predictor, images[0])
    finally:
        del image_predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    memory_predictor = NextFramePredictor.from_checkpoint(
        paths.checkpoint, device=device
    )
    with torch.inference_mode():
        frame_results = predict_sequence(
            memory_predictor, images, reference_result.mask
        )
    elapsed = time.perf_counter() - started_at
    save_next_visualization(images, reference_result, frame_results, paths.output)

    print(
        json.dumps(
            {
                "checkpoint": str(paths.checkpoint),
                "frames": [str(path) for path in paths.frames],
                "frame_order": "left_to_right",
                "device": device,
                "reference_prompt": describe_reference_prompt(),
                "reference_mask": summarize_reference(reference_result),
                "elapsed_sec": round(elapsed, 3),
                "predictions": [
                    {
                        "sequence_index": result.sequence_index,
                        **summarize_prediction(result.prediction, obj_id=OBJ_ID),
                    }
                    for result in frame_results
                ],
                "output": str(paths.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
