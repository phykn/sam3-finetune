import argparse
import json
import sys
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class TileConfig:
    tile: int
    points_per_side: int


@dataclass
class GridPaths:
    image: Path
    checkpoint: Path
    output_dir: Path

    @property
    def points(self) -> Path:
        return self.output_dir / "grid_points.png"

    @property
    def extra_overlay(self) -> Path:
        return self.output_dir / "grid_overlay.png"

    @property
    def extra_grid(self) -> Path:
        return self.output_dir / "grid_masks.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a combined grid/automatic mask smoke case."
    )
    parser.add_argument("--image", default="asset/frog_target.jpg")
    parser.add_argument("--checkpoint", default="weight/sam3.1_multiplex.pt")
    parser.add_argument("--output-dir", default="outputs/grid")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--tiles", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--points-per-side", nargs="+", type=int, default=[32, 16])
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--max-masks", type=int, default=100)
    parser.add_argument("--max-masks-per-crop", type=int, default=None)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.0)
    parser.add_argument("--stability-score-thresh", type=float, default=0.75)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-nms-thresh", type=float, default=None)
    parser.add_argument("--keep-crop-edge-masks", action="store_true")
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--allow-cross-crop-prompt-decode", action="store_true")
    parser.add_argument("--show-masks", type=int, default=8)
    parser.add_argument("--save-extra", action="store_true")
    parser.add_argument("--overlay-max-masks", type=int, default=50)
    parser.add_argument("--grid-max-masks", type=int, default=24)
    parser.add_argument("--grid-columns", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args(argv)


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


def resolve_paths(args: argparse.Namespace, *, root: Path = ROOT) -> GridPaths:
    return GridPaths(
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


def build_generator_kwargs(
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
        "max_masks": args.max_masks,
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


def summarize_proposals(proposals, *, top_k: int) -> list[dict[str, object]]:
    summaries = []
    for proposal in proposals[:top_k]:
        summaries.append(
            {
                "bbox": list(proposal.bbox),
                "area": int(proposal.area),
                "predicted_iou": float(proposal.predicted_iou),
                "stability_score": float(proposal.stability_score),
                "point_coords": list(proposal.point_coords),
                "crop_grid": int(proposal.crop_grid),
                "crop_index": int(proposal.crop_index),
                "crop_box": list(proposal.crop_box),
            }
        )
    return summaries


def build_grid_points(
    image_size: tuple[int, int],
    configs: list[TileConfig],
    overlap: float,
) -> list[tuple[float, float, int]]:
    from src.predict.grid.geometry import build_point_grid, generate_crop_boxes

    width, height = image_size
    grid_points = []
    for config in configs:
        normalized_points = build_point_grid(config.points_per_side)
        crop_boxes = generate_crop_boxes(width, height, config.tile, overlap)
        for x0, y0, x1, y1 in crop_boxes:
            crop_width = x1 - x0
            crop_height = y1 - y0
            for normalized_x, normalized_y in normalized_points:
                grid_points.append(
                    (
                        float(x0 + normalized_x * crop_width),
                        float(y0 + normalized_y * crop_height),
                        config.tile,
                    )
                )
    return grid_points


def save_grid_point_visualization(
    image: Image.Image,
    proposals,
    grid_points: list[tuple[float, float, int]],
    path: Path,
    *,
    show_masks: int,
) -> None:
    selected = list(proposals[: max(show_masks, 0)])
    left_width = 640
    scale = left_width / image.width
    left_height = max(1, int(round(image.height * scale)))
    thumb_width = 300
    thumb_height = max(1, int(round(image.height * thumb_width / image.width)))
    preview_columns = 2 if len(selected) > 1 else 1
    header_height = 34
    gap = 12
    padding = 16
    preview_rows = max(1, ceil(max(len(selected), 1) / preview_columns))
    preview_width = preview_columns * thumb_width + (preview_columns - 1) * gap
    preview_height = (
        preview_rows * (header_height + thumb_height) + (preview_rows - 1) * gap
    )
    canvas_width = padding * 3 + left_width + preview_width
    canvas_height = padding * 2 + max(left_height, preview_height)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (246, 246, 242))

    left_panel = _make_grid_point_panel(
        image,
        grid_points,
        selected,
        (left_width, left_height),
    )
    canvas.paste(left_panel, (padding, padding))

    preview_x0 = padding * 2 + left_width
    if selected:
        for index, proposal in enumerate(selected):
            col = index % preview_columns
            row = index // preview_columns
            x = preview_x0 + col * (thumb_width + gap)
            y = padding + row * (header_height + thumb_height + gap)
            preview = _make_mask_preview(
                image,
                proposal,
                index,
                (thumb_width, thumb_height),
                header_height,
            )
            canvas.paste(preview, (x, y))
    else:
        draw = ImageDraw.Draw(canvas)
        draw.text((preview_x0, padding), "no proposals", fill=(40, 40, 40))

    canvas.save(path)


def _make_grid_point_panel(
    image: Image.Image,
    grid_points: list[tuple[float, float, int]],
    selected,
    size: tuple[int, int],
) -> Image.Image:
    panel = image.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale_x = size[0] / image.width
    scale_y = size[1] / image.height
    for x, y, tile in grid_points:
        point_x = x * scale_x
        point_y = y * scale_y
        color = (255, 255, 255, 72) if tile == 1 else (42, 157, 143, 82)
        draw.ellipse((point_x - 1, point_y - 1, point_x + 1, point_y + 1), fill=color)

    for index, proposal in enumerate(selected):
        color = _vis_color(index)
        point_x = proposal.point_coords[0] * scale_x
        point_y = proposal.point_coords[1] * scale_y
        bbox = _scale_bbox(proposal.bbox, scale_x, scale_y)
        draw.rectangle(bbox, outline=color, width=3)
        draw.ellipse(
            (point_x - 6, point_y - 6, point_x + 6, point_y + 6),
            fill=(*color, 210),
            outline=(15, 15, 15, 255),
            width=2,
        )
        draw.text((point_x + 8, point_y - 8), str(index + 1), fill=(*color, 255))
    return Image.alpha_composite(panel, overlay).convert("RGB")


def _make_mask_preview(
    image: Image.Image,
    proposal,
    index: int,
    size: tuple[int, int],
    header_height: int,
) -> Image.Image:
    from src.predict.grid.proposals import proposal_to_full_mask

    color = _vis_color(index)
    cell = Image.new("RGB", (size[0], size[1] + header_height), (30, 30, 30))
    draw = ImageDraw.Draw(cell)
    label = (
        f"#{index + 1} tile={proposal.crop_grid} "
        f"iou={proposal.predicted_iou:.2f} stab={proposal.stability_score:.2f}"
    )
    draw.rectangle((0, 0, size[0], header_height), fill=(246, 246, 242))
    draw.rectangle((8, 10, 20, 22), fill=color)
    draw.text((26, 9), label, fill=(30, 30, 30))

    preview = image.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    full_mask = proposal_to_full_mask(proposal)
    mask = Image.fromarray(full_mask.astype(np.uint8) * 130, mode="L").resize(
        size,
        Image.Resampling.NEAREST,
    )
    layer = Image.new("RGBA", size, (*color, 255))
    layer.putalpha(mask)
    preview = Image.alpha_composite(preview, layer)
    preview_draw = ImageDraw.Draw(preview)
    scale_x = size[0] / image.width
    scale_y = size[1] / image.height
    preview_draw.rectangle(
        _scale_bbox(proposal.bbox, scale_x, scale_y), outline=color, width=3
    )
    point_x = proposal.point_coords[0] * scale_x
    point_y = proposal.point_coords[1] * scale_y
    preview_draw.ellipse(
        (point_x - 5, point_y - 5, point_x + 5, point_y + 5),
        fill=(*color, 230),
        outline=(20, 20, 20, 255),
        width=2,
    )
    cell.paste(preview.convert("RGB"), (0, header_height))
    return cell


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    device = resolve_device(args.device)
    tile_configs = resolve_tile_configs(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    from src.predict.grid.generator import AutomaticMaskGenerator
    from src.predict.grid.proposals import (
        count_proposals_by_crop_grid,
        save_proposal_grid,
        save_proposal_overlay,
    )
    from src.predict.prompted import Sam3Predictor

    image = Image.open(paths.image).convert("RGB")
    predictor = Sam3Predictor.from_checkpoint(paths.checkpoint, device=device)
    generator = AutomaticMaskGenerator(
        predictor, **build_generator_kwargs(args, tile_configs)
    )
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else torch.no_grad()
    )
    started_at = time.perf_counter()
    with autocast_context:
        proposals = generator.generate(image)
    elapsed = time.perf_counter() - started_at

    grid_points = build_grid_points(image.size, tile_configs, args.overlap)
    save_grid_point_visualization(
        image,
        proposals,
        grid_points,
        paths.points,
        show_masks=args.show_masks,
    )

    extra_paths = {}
    if args.save_extra:
        save_proposal_overlay(
            image,
            proposals,
            paths.extra_overlay,
            max_masks=args.overlay_max_masks,
        )
        save_proposal_grid(
            image,
            proposals,
            paths.extra_grid,
            max_masks=args.grid_max_masks,
            columns=args.grid_columns,
        )
        extra_paths = {
            "overlay_path": str(paths.extra_overlay),
            "grid_path": str(paths.extra_grid),
        }

    print(
        json.dumps(
            {
                "checkpoint": str(paths.checkpoint),
                "image": str(paths.image),
                "device": device,
                "tiles": [config.tile for config in tile_configs],
                "points_per_side": [config.points_per_side for config in tile_configs],
                "overlap": args.overlap,
                "filter_crop_edge_masks": not args.keep_crop_edge_masks,
                "elapsed_sec": round(elapsed, 3),
                "proposal_count": len(proposals),
                "proposal_count_by_crop_grid": count_proposals_by_crop_grid(proposals),
                "top_proposals": summarize_proposals(
                    proposals,
                    top_k=args.top_k,
                ),
                "points_path": str(paths.points),
                **extra_paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
