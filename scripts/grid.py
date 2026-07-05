import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.devices import resolve_device
from common.paths import ensure_workspace_on_path, resolve_workspace_path, ROOT

ensure_workspace_on_path()


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
    def output(self) -> Path:
        return self.output_dir / "grid.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed grid prediction.")
    parser.add_argument("--image", default="asset/frog_target.jpg")
    parser.add_argument("--checkpoint", default="weight/sam3.1_multiplex.pt")
    parser.add_argument("--output-dir", default="outputs/grid")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--tiles", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--points-per-side", nargs="+", type=int, default=[32, 16])
    parser.add_argument("--overlap", type=float, default=0.25)
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
        image=resolve_workspace_path(args.image, root=root),
        checkpoint=resolve_workspace_path(args.checkpoint, root=root),
        output_dir=resolve_workspace_path(args.output_dir, root=root),
    )


def build_generator_kwargs(
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


def summarize_proposals(proposals) -> list[dict[str, object]]:
    summaries = []
    for proposal in proposals:
        summaries.append(
            {
                "bbox": list(proposal.bbox),
                "area": int(proposal.area),
                "predicted_iou": float(proposal.predicted_iou),
                "stability_score": float(proposal.stability_score),
                "point_coords": [float(value) for value in proposal.point_coords],
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


def save_grid_visualization(
    image: Image.Image,
    proposals,
    grid_points: list[tuple[float, float, int]],
    path: Path,
) -> None:
    panel_width = 520
    gap = 14
    padding = 16
    points_panel = _make_grid_points_panel(
        image,
        proposals,
        grid_points,
        panel_width,
    )
    masks_panel = _make_mask_overlay_panel(
        image,
        proposals,
        panel_width,
    )
    panels = [points_panel, masks_panel]
    canvas_width = padding * 2 + sum(panel.width for panel in panels) + gap
    canvas_height = padding * 2 + max(panel.height for panel in panels)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (246, 246, 242))
    x = padding
    for panel in panels:
        canvas.paste(panel, (x, padding))
        x += panel.width + gap
    canvas.save(path)


def _make_grid_points_panel(
    image: Image.Image,
    proposals,
    grid_points: list[tuple[float, float, int]],
    width: int,
) -> Image.Image:
    body = _resize_image(image, width)
    overlay = body.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    scale_x = body.width / image.width
    scale_y = body.height / image.height
    for x, y, tile in grid_points:
        color = (255, 255, 255, 72) if tile == 1 else (42, 157, 143, 92)
        px = x * scale_x
        py = y * scale_y
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)

    for index, proposal in enumerate(proposals):
        color = _vis_color(index)
        point_x = proposal.point_coords[0] * scale_x
        point_y = proposal.point_coords[1] * scale_y
        draw.rectangle(
            _scale_bbox(proposal.bbox, scale_x, scale_y),
            outline=(*color, 255),
            width=2,
        )
        draw.ellipse(
            (point_x - 4, point_y - 4, point_x + 4, point_y + 4),
            fill=(*color, 210),
            outline=(15, 15, 15, 255),
            width=1,
        )
        draw.text((point_x + 5, point_y - 8), str(index + 1), fill=(*color, 255))
    return _with_header(overlay.convert("RGB"), "grid points")


def _make_mask_overlay_panel(
    image: Image.Image,
    proposals,
    width: int,
) -> Image.Image:
    from src.predict.grid.proposals import proposal_to_full_mask

    body = _resize_image(image, width)
    overlay = body.convert("RGBA")
    scale_x = body.width / image.width
    scale_y = body.height / image.height
    for index, proposal in enumerate(proposals):
        color = _vis_color(index)
        mask = proposal_to_full_mask(proposal)
        resized_mask = _resize_mask(mask, body.size)
        layer = Image.new("RGBA", body.size, (*color, 255))
        layer.putalpha(Image.fromarray(resized_mask.astype(np.uint8) * 70, mode="L"))
        overlay = Image.alpha_composite(overlay, layer)
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(
            _scale_bbox(proposal.bbox, scale_x, scale_y),
            outline=(*color, 255),
            width=2,
        )
        x0, y0, _x1, _y1 = _scale_bbox(proposal.bbox, scale_x, scale_y)
        draw.text((x0 + 3, max(0, y0 - 12)), str(index + 1), fill=(*color, 255))
    return _with_header(overlay.convert("RGB"), "grid masks")


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


def _with_header(body: Image.Image, title: str) -> Image.Image:
    header_height = 32
    panel = Image.new("RGB", (body.width, body.height + header_height), (30, 30, 30))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, header_height), fill=(246, 246, 242))
    draw.text((10, 9), title, fill=(30, 30, 30))
    panel.paste(body, (0, header_height))
    return panel


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    device = resolve_device(args.device)
    tile_configs = resolve_tile_configs(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    from src.predict.grid.generator import AutomaticMaskGenerator
    from src.predict.grid.proposals import count_proposals_by_crop_grid
    from src.predict.prompted import Sam3Predictor

    image = Image.open(paths.image).convert("RGB")
    predictor = Sam3Predictor.from_checkpoint(paths.checkpoint, device=device)
    generator = AutomaticMaskGenerator(
        predictor,
        **build_generator_kwargs(args, tile_configs),
    )
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else torch.no_grad()
    )
    started_at = time.perf_counter()
    with torch.inference_mode(), autocast_context:
        proposals = generator.generate(image)
    elapsed = time.perf_counter() - started_at

    grid_points = build_grid_points(image.size, tile_configs, args.overlap)
    save_grid_visualization(image, proposals, grid_points, paths.output)

    print(
        json.dumps(
            {
                "checkpoint": str(paths.checkpoint),
                "image": str(paths.image),
                "device": device,
                "tiles": [config.tile for config in tile_configs],
                "points_per_side": [config.points_per_side for config in tile_configs],
                "overlap": args.overlap,
                "elapsed_sec": round(elapsed, 3),
                "proposal_count": len(proposals),
                "proposal_count_by_crop_grid": count_proposals_by_crop_grid(proposals),
                "proposals": summarize_proposals(proposals),
                "output": str(paths.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
