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

from src.types import ContextPrediction, MaskInstance


@dataclass
class TileConfig:
    tile: int
    points_per_side: int


@dataclass
class RefinePaths:
    image: Path
    checkpoint: Path
    output_dir: Path

    @property
    def output(self) -> Path:
        return self.output_dir / "refine.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine every valid grid mask on the same image."
    )
    parser.add_argument("--image", default="asset/frog_target.jpg")
    parser.add_argument("--checkpoint", default="weight/sam3.1_multiplex.pt")
    parser.add_argument("--output-dir", default="outputs/refine")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--tiles", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--points-per-side", nargs="+", type=int, default=[32, 16])
    parser.add_argument("--overlap", type=float, default=0.25)
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace, *, root: Path = ROOT) -> RefinePaths:
    return RefinePaths(
        image=root / args.image,
        checkpoint=root / args.checkpoint,
        output_dir=root / args.output_dir,
    )


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when --device cuda is selected.")
    return device


def resolve_tile_configs(args: argparse.Namespace) -> list[TileConfig]:
    if any(tile <= 0 for tile in args.tiles):
        raise ValueError("--tiles entries must be positive integers")
    if any(points <= 0 for points in args.points_per_side):
        raise ValueError("--points-per-side entries must be positive integers")
    if len(args.points_per_side) == 1:
        points_values = args.points_per_side * len(args.tiles)
    elif len(args.points_per_side) == len(args.tiles):
        points_values = args.points_per_side
    else:
        raise ValueError(
            "--points-per-side must have length 1 or match the number of --tiles"
        )
    return [
        TileConfig(tile=tile, points_per_side=points)
        for tile, points in zip(args.tiles, points_values)
    ]


def build_grid_kwargs(
    args: argparse.Namespace,
    configs: list[TileConfig],
) -> dict[str, object]:
    if not configs:
        raise ValueError("configs must not be empty")
    return {
        "points_per_side": configs[0].points_per_side,
        "crop_grids": [config.tile for config in configs],
        "crop_points_per_side": [config.points_per_side for config in configs],
        "crop_overlap_ratio": args.overlap,
    }


def summarize_instances(
    instances: list[MaskInstance],
) -> list[dict[str, object]]:
    summaries = []
    for instance in instances:
        summaries.append(
            {
                "bbox": list(instance.bbox),
                "area": int(instance.area),
                "score": float(instance.score),
                "context_score": _optional_float(instance.context_score),
                "base_score": _optional_float(instance.base_score),
                "predicted_iou": _optional_float(instance.predicted_iou),
                "stability_score": _optional_float(instance.stability_score),
                "point_coords": (
                    None
                    if instance.point_coords is None
                    else [float(value) for value in instance.point_coords]
                ),
                "crop_grid": instance.crop_grid,
                "crop_index": instance.crop_index,
            }
        )
    return summaries


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


def save_refine_visualization(
    image: Image.Image,
    base_instances: list[MaskInstance],
    refined_predictions: list[ContextPrediction],
    path: Path,
) -> None:
    panel_width = 420
    gap = 14
    padding = 16
    base_panel = _make_instance_panel(
        image,
        base_instances,
        "grid context",
        panel_width,
    )
    refined_panel = _make_prediction_panel(
        image,
        refined_predictions,
        "context prediction",
        panel_width,
    )
    panels = [base_panel, refined_panel]
    canvas_width = padding * 2 + sum(panel.width for panel in panels) + gap
    canvas_height = padding * 2 + max(panel.height for panel in panels)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (246, 246, 242))
    x = padding
    for panel in panels:
        canvas.paste(panel, (x, padding))
        x += panel.width + gap
    canvas.save(path)


def _make_instance_panel(
    image: Image.Image,
    instances: list[MaskInstance],
    title: str,
    width: int,
) -> Image.Image:
    body = _resize_image(image, width)
    draw_image = body.convert("RGBA")
    scale_x = body.width / image.width
    scale_y = body.height / image.height
    for index, instance in enumerate(instances):
        color = _vis_color(index)
        mask = _resize_mask(instance.to_full_mask(), body.size)
        draw_image = _overlay_full_mask(draw_image.convert("RGB"), mask, color, 85)
        draw = ImageDraw.Draw(draw_image)
        draw.rectangle(
            _scale_bbox(instance.bbox, scale_x, scale_y),
            outline=color,
            width=2,
        )
        label = str(index + 1)
        if instance.base_score is not None:
            label = f"{label} b={instance.base_score:.2f}"
        x0, y0, _x1, _y1 = _scale_bbox(instance.bbox, scale_x, scale_y)
        draw.text((x0 + 4, max(0, y0 - 14)), label, fill=color)
    return _with_header(draw_image.convert("RGB"), title)


def _make_prediction_panel(
    image: Image.Image,
    predictions: list[ContextPrediction],
    title: str,
    width: int,
) -> Image.Image:
    body = _resize_image(image, width)
    draw_image = body.convert("RGBA")
    scale_x = body.width / image.width
    scale_y = body.height / image.height
    for index, prediction in enumerate(predictions):
        color = _vis_color(index)
        mask = _resize_mask(_prediction_to_full_mask(prediction), body.size)
        draw_image = _overlay_full_mask(draw_image.convert("RGB"), mask, color, 95)
        draw = ImageDraw.Draw(draw_image)
        draw.rectangle(
            _scale_bbox(prediction.bbox, scale_x, scale_y),
            outline=color,
            width=3,
        )
        x0, y0, _x1, _y1 = _scale_bbox(prediction.bbox, scale_x, scale_y)
        draw.text(
            (x0 + 4, max(0, y0 - 14)),
            f"{index + 1} c={prediction.context_score:.2f}",
            fill=color,
        )
    return _with_header(draw_image.convert("RGB"), title)


def _prediction_to_full_mask(prediction: ContextPrediction) -> np.ndarray:
    width, height = prediction.image_size
    x0, y0, x1, y1 = prediction.bbox
    full_mask = np.zeros((height, width), dtype=bool)
    full_mask[y0:y1, x0:x1] = prediction.segmentation
    return full_mask


def _with_header(body: Image.Image, title: str) -> Image.Image:
    header_height = 32
    panel = Image.new("RGB", (body.width, body.height + header_height), (30, 30, 30))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, header_height), fill=(246, 246, 242))
    draw.text((10, 9), title, fill=(30, 30, 30))
    panel.paste(body.convert("RGB"), (0, header_height))
    return panel


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
    bbox: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        int(round(x0 * scale_x)),
        int(round(y0 * scale_y)),
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
    )


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


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    device = resolve_device(args.device)
    tile_configs = resolve_tile_configs(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    from src.predict.prompted import Sam3Predictor
    from src.predict.refine import ContextGridRefiner

    image = Image.open(paths.image).convert("RGB")
    predictor = Sam3Predictor.from_checkpoint(paths.checkpoint, device=device)
    refiner = ContextGridRefiner.from_predictor(
        predictor,
        grid_kwargs=build_grid_kwargs(args, tile_configs),
        matcher_kwargs={"max_masks": None},
    )
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else torch.no_grad()
    )
    started_at = time.perf_counter()
    with torch.inference_mode(), autocast_context:
        result = refiner.refine(image)
    elapsed = time.perf_counter() - started_at

    save_refine_visualization(
        image,
        result.base_instances,
        result.refined_predictions,
        paths.output,
    )

    print(
        json.dumps(
            {
                "checkpoint": str(paths.checkpoint),
                "image": str(paths.image),
                "device": device,
                "tiles": [config.tile for config in tile_configs],
                "points_per_side": [config.points_per_side for config in tile_configs],
                "overlap": args.overlap,
                "grid_candidate_count": len(result.base_instances),
                "context_reference_count": len(result.context_references),
                "refined_count": len(result.refined_predictions),
                "elapsed_sec": round(elapsed, 3),
                "grid_results": summarize_instances(result.base_instances),
                "refined_results": summarize_predictions(result.refined_predictions),
                "output": str(paths.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
