import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

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

    def overlay_for(self, tile: int) -> Path:
        return self.output_dir / f"tile{tile}_overlay.png"

    def grid_for(self, tile: int) -> Path:
        return self.output_dir / f"tile{tile}_grid.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grid/automatic mask smoke cases by tile size."
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
    config: TileConfig,
) -> dict[str, object]:
    return {
        "points_per_side": config.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "box_nms_thresh": args.box_nms_thresh,
        "max_masks": args.max_masks,
        "crop_grids": [config.tile],
        "crop_points_per_side": [config.points_per_side],
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
    results = []

    for config in tile_configs:
        generator = AutomaticMaskGenerator(
            predictor, **build_generator_kwargs(args, config)
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

        overlay_path = paths.overlay_for(config.tile)
        grid_path = paths.grid_for(config.tile)
        save_proposal_overlay(
            image,
            proposals,
            overlay_path,
            max_masks=args.overlay_max_masks,
        )
        save_proposal_grid(
            image,
            proposals,
            grid_path,
            max_masks=args.grid_max_masks,
            columns=args.grid_columns,
        )

        results.append(
            {
                "tile": config.tile,
                "points_per_side": config.points_per_side,
                "overlap": args.overlap,
                "filter_crop_edge_masks": not args.keep_crop_edge_masks,
                "elapsed_sec": round(elapsed, 3),
                "proposal_count": len(proposals),
                "proposal_count_by_crop_grid": count_proposals_by_crop_grid(proposals),
                "top_proposals": summarize_proposals(
                    proposals,
                    top_k=args.top_k,
                ),
                "overlay_path": str(overlay_path),
                "grid_path": str(grid_path),
            }
        )

    print(
        json.dumps(
            {
                "checkpoint": str(paths.checkpoint),
                "image": str(paths.image),
                "device": device,
                "tiles": [config.tile for config in tile_configs],
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
