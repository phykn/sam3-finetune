import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict.context.postprocess import (
    context_prediction_to_full_mask,
    nms_context_predictions,
)
from src.predict.refine import MaskRefiner, select_best_mask
from src.types import ContextPrediction, ContextReference

REFERENCE_BOX = (270.0, 450.0, 610.0, 900.0)
VISUAL_NMS_IOU_THRESHOLD = 0.35


@dataclass
class ContextPaths:
    reference_image: Path
    target_image: Path
    checkpoint: Path
    output_dir: Path

    @property
    def output(self) -> Path:
        return self.output_dir / "context.png"


@dataclass
class ReferenceMaskResult:
    mask: np.ndarray
    score: float
    selected_index: int
    refined_score: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find a frog in a target image from a reference frog mask.",
        allow_abbrev=False,
    )
    parser.add_argument("--reference-image", default="asset/frog_reference.jpg")
    parser.add_argument("--target-image", default="asset/frog_target.jpg")
    parser.add_argument("--checkpoint", default="weight/sam3.1_multiplex.pt")
    parser.add_argument("--output-dir", default="outputs/context")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace, *, root: Path = ROOT) -> ContextPaths:
    return ContextPaths(
        reference_image=root / args.reference_image,
        target_image=root / args.target_image,
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


def describe_reference_prompt() -> dict[str, object]:
    box = reference_box_array()
    x0, y0, x1, y1 = box.tolist()
    return {
        "type": "box",
        "box": {
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
        },
    }


def describe_reference_context() -> dict[str, object]:
    return {
        "type": "sam_mask",
        "mask_source": "sam_box_prompt",
        "source_prompt": describe_reference_prompt(),
    }


def predict_reference_mask(
    predictor,
    reference_image: Image.Image,
) -> ReferenceMaskResult:
    box = reference_box_array()
    embedding = predictor.encode_image(reference_image)
    masks, scores, low_res_masks = predictor.predict_from_embedding(
        embedding,
        box=box,
        multimask_output=True,
    )
    _mask, score, selected_index = select_best_mask(masks, scores)
    flat_low_res = np.asarray(low_res_masks).reshape(
        -1,
        *np.asarray(low_res_masks).shape[-2:],
    )
    refined = MaskRefiner(predictor).refine(
        embedding=embedding,
        box=box,
        mask_input=flat_low_res[selected_index],
    )
    return ReferenceMaskResult(
        mask=refined.mask.astype(bool),
        score=float(score),
        selected_index=int(selected_index),
        refined_score=float(refined.score),
    )


def summarize_reference_mask(result: ReferenceMaskResult) -> dict[str, object]:
    return {
        "area": int(result.mask.astype(bool).sum()),
        "score": float(result.score),
        "selected_index": int(result.selected_index),
        "refined_score": float(result.refined_score),
    }


def summarize_predictions(
    predictions: list[ContextPrediction],
) -> list[dict[str, object]]:
    summaries = []
    for prediction in predictions:
        summaries.append(
            {
                "bbox": list(prediction.bbox),
                "area": int(prediction.area),
                "score": float(prediction.score),
                "context_score": float(prediction.context_score),
                "predicted_iou": float(prediction.predicted_iou),
                "stability_score": float(prediction.stability_score),
                "area_score": float(prediction.area_score),
                "point_coords": [float(value) for value in prediction.point_coords],
            }
        )
    return summaries


def apply_visual_nms(
    predictions: list[ContextPrediction],
) -> list[ContextPrediction]:
    return nms_context_predictions(
        predictions,
        iou_threshold=VISUAL_NMS_IOU_THRESHOLD,
        max_masks=None,
    )


def save_context_visualization(
    reference_image: Image.Image,
    reference_mask: np.ndarray,
    target_image: Image.Image,
    predictions: list[ContextPrediction],
    nms_predictions: list[ContextPrediction],
    path: Path,
) -> None:
    panel_width = 440
    gap = 14
    padding = 16
    reference_panel = _make_reference_panel(
        reference_image,
        reference_mask,
        panel_width,
    )
    target_panel = _make_prediction_panel(
        target_image,
        predictions,
        panel_width,
        "target all predictions",
    )
    nms_panel = _make_prediction_panel(
        target_image,
        nms_predictions,
        panel_width,
        f"target nms {VISUAL_NMS_IOU_THRESHOLD:.2f}",
    )
    panels = [reference_panel, target_panel, nms_panel]
    canvas_width = padding * 2 + sum(panel.width for panel in panels) + gap * 2
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
    overlay = _overlay_full_mask(body, _resize_mask(mask, body.size), (0, 210, 255), 90)
    return _with_header(overlay, "reference mask")


def _make_prediction_panel(
    image: Image.Image,
    predictions: list[ContextPrediction],
    width: int,
    title: str,
) -> Image.Image:
    body = _resize_image(image, width)
    overlay = body.convert("RGBA")
    scale_x = body.width / image.width
    scale_y = body.height / image.height
    for index, prediction in enumerate(predictions):
        color = _vis_color(index)
        mask = _resize_mask(context_prediction_to_full_mask(prediction), body.size)
        layer = Image.new("RGBA", body.size, (*color, 255))
        layer.putalpha(Image.fromarray(mask.astype(np.uint8) * 85, mode="L"))
        overlay = Image.alpha_composite(overlay, layer)
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(
            _scale_bbox(prediction.bbox, scale_x, scale_y),
            outline=(*color, 255),
            width=2,
        )
        x0, y0, _x1, _y1 = _scale_bbox(prediction.bbox, scale_x, scale_y)
        draw.text(
            (x0 + 3, max(0, y0 - 13)),
            f"{index + 1} c={prediction.context_score:.2f}",
            fill=(*color, 255),
        )
    return _with_header(overlay.convert("RGB"), title)


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


def _scale_bbox(
    bbox: tuple[int, int, int, int] | np.ndarray,
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


def _overlay_full_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: int,
) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (*color, 255))
    layer.putalpha(Image.fromarray(mask.astype(np.uint8) * alpha, mode="L"))
    return Image.alpha_composite(base, layer).convert("RGB")


def _vis_color(index: int) -> tuple[int, int, int]:
    palette = [
        (230, 57, 70),
        (42, 157, 143),
        (69, 123, 157),
        (244, 162, 97),
        (131, 56, 236),
        (255, 190, 11),
        (6, 214, 160),
        (239, 71, 111),
    ]
    return palette[index % len(palette)]


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
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    from src.predict.context import ContextMatcher
    from src.predict.prompted import Sam3Predictor

    reference_image = Image.open(paths.reference_image).convert("RGB")
    target_image = Image.open(paths.target_image).convert("RGB")
    predictor = Sam3Predictor.from_checkpoint(paths.checkpoint, device=device)
    matcher = ContextMatcher(
        predictor,
        max_masks=None,
        use_reference_mask_prior=True,
    )
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else torch.no_grad()
    )
    started_at = time.perf_counter()
    with torch.inference_mode(), autocast_context:
        reference_mask = predict_reference_mask(predictor, reference_image)
        predictions = matcher.predict(
            target_image=target_image,
            references=[
                ContextReference(
                    image=reference_image,
                    mask=reference_mask.mask,
                )
            ],
        )
    elapsed = time.perf_counter() - started_at
    nms_predictions = apply_visual_nms(predictions)

    save_context_visualization(
        reference_image,
        reference_mask.mask,
        target_image,
        predictions,
        nms_predictions,
        paths.output,
    )

    print(
        json.dumps(
            {
                "checkpoint": str(paths.checkpoint),
                "reference_image": str(paths.reference_image),
                "target_image": str(paths.target_image),
                "device": device,
                "context_settings": {
                    "max_masks": None,
                    "use_reference_mask_prior": True,
                },
                "context_input": describe_reference_context(),
                "reference_mask": summarize_reference_mask(reference_mask),
                "elapsed_sec": round(elapsed, 3),
                "prediction_count": len(predictions),
                "predictions": summarize_predictions(predictions),
                "visual_nms": {
                    "iou_threshold": VISUAL_NMS_IOU_THRESHOLD,
                    "prediction_count": len(nms_predictions),
                    "predictions": summarize_predictions(nms_predictions),
                },
                "output": str(paths.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
