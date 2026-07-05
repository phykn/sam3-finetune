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

from src.types import MaskInstance


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
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--grid-max-masks", type=int, default=100)
    parser.add_argument("--max-masks-per-crop", type=int, default=None)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.0)
    parser.add_argument("--stability-score-thresh", type=float, default=0.75)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-nms-thresh", type=float, default=None)
    parser.add_argument("--keep-crop-edge-masks", action="store_true")
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--allow-cross-crop-prompt-decode", action="store_true")
    parser.add_argument("--refine-batch-size", type=int, default=8)
    parser.add_argument("--refine-multimask", action="store_true")
    parser.add_argument("--mask-foreground", type=float, default=4.0)
    parser.add_argument("--mask-background", type=float, default=-4.0)
    parser.add_argument("--max-masks", type=int, default=8)
    parser.add_argument("--show-masks", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
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
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "box_nms_thresh": args.box_nms_thresh,
        "max_masks": args.grid_max_masks,
        "crop_grids": [config.tile for config in configs],
        "crop_points_per_side": [config.points_per_side for config in configs],
        "crop_overlap_ratio": args.overlap,
        "crop_nms_thresh": args.crop_nms_thresh,
        "max_masks_per_crop": args.max_masks_per_crop,
        "filter_crop_edge_masks": not args.keep_crop_edge_masks,
        "image_batch_size": args.image_batch_size,
        "prompt_batch_size": args.prompt_batch_size,
        "allow_cross_crop_prompt_decode": args.allow_cross_crop_prompt_decode,
    }


def build_refiner_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "batch_size": args.refine_batch_size,
        "multimask_output": args.refine_multimask,
        "mask_foreground": args.mask_foreground,
        "mask_background": args.mask_background,
    }


def summarize_instances(
    instances: list[MaskInstance],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    summaries = []
    for instance in instances[:top_k]:
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


def save_refine_visualization(
    image: Image.Image,
    base_instances: list[MaskInstance],
    refined_instances: list[MaskInstance],
    path: Path,
    *,
    max_masks: int,
) -> None:
    panel_width = 420
    gap = 14
    padding = 16
    base_all_panel = _make_instance_panel(
        image,
        base_instances,
        "grid masks",
        panel_width,
        max_masks=max(1, min(len(base_instances), 24)),
    )
    base_panel = _make_instance_panel(
        image,
        base_instances,
        "grid before",
        panel_width,
        max_masks=max_masks,
    )
    refined_panel = _make_instance_panel(
        image,
        refined_instances,
        "refined after",
        panel_width,
        max_masks=max_masks,
    )
    panels = [base_all_panel, base_panel, refined_panel]
    canvas_width = padding * 2 + sum(panel.width for panel in panels) + gap * 2
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
    *,
    max_masks: int,
) -> Image.Image:
    body = _resize_image(image, width)
    draw_image = body.convert("RGBA")
    scale_x = body.width / image.width
    scale_y = body.height / image.height
    for index, instance in enumerate(instances[:max_masks]):
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
    from src.predict.refine import GridMaskRefiner

    image = Image.open(paths.image).convert("RGB")
    predictor = Sam3Predictor.from_checkpoint(paths.checkpoint, device=device)
    refiner = GridMaskRefiner.from_predictor(
        predictor,
        grid_kwargs=build_grid_kwargs(args, tile_configs),
        **build_refiner_kwargs(args),
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
        result.refined_instances,
        paths.output,
        max_masks=args.show_masks,
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
                "refined_count": len(result.refined_instances),
                "elapsed_sec": round(elapsed, 3),
                "grid_preview": summarize_instances(
                    result.base_instances,
                    top_k=args.top_k,
                ),
                "refined_preview": summarize_instances(
                    result.refined_instances,
                    top_k=args.top_k,
                ),
                "output": str(paths.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
