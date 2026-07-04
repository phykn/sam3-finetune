
import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.masks.generator import AutomaticMaskGenerator
from src.masks.proposals import (
    count_proposals_by_crop_grid,
    save_proposal_grid,
    save_proposal_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-grids", nargs="*", type=int, default=None)
    parser.add_argument("--crop-points-per-side", nargs="*", type=int, default=None)
    parser.add_argument("--crop-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--max-masks", type=int, default=100)
    parser.add_argument("--max-masks-per-crop", type=int, default=None)
    parser.add_argument("--keep-crop-edge-masks", action="store_true")
    parser.add_argument("--crop-encode-batch-size", type=int, default=1)
    parser.add_argument("--prompt-decode-batch-size", type=int, default=1)
    parser.add_argument("--image-batch-size", type=int, default=None)
    parser.add_argument("--prompt-batch-size", type=int, default=None)
    parser.add_argument("--allow-cross-crop-prompt-decode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    image_path = ROOT / "asset" / "sample.jpg"
    checkpoint_path = ROOT / "weight" / "sam3.1_multiplex.pt"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    crop_grids = args.crop_grids if args.crop_grids else None
    crop_points_per_side = (
        args.crop_points_per_side if args.crop_points_per_side else None
    )

    image = Image.open(image_path).convert("RGB")
    image_batch_size = (
        args.crop_encode_batch_size
        if args.image_batch_size is None
        else args.image_batch_size
    )
    prompt_batch_size = (
        args.prompt_decode_batch_size
        if args.prompt_batch_size is None
        else args.prompt_batch_size
    )
    generator = AutomaticMaskGenerator.from_checkpoint(
        checkpoint_path,
        device="cuda",
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.75,
        box_nms_thresh=0.7,
        max_masks=args.max_masks,
        crop_grids=crop_grids,
        crop_points_per_side=crop_points_per_side,
        crop_overlap_ratio=args.crop_overlap_ratio,
        max_masks_per_crop=args.max_masks_per_crop,
        filter_crop_edge_masks=not args.keep_crop_edge_masks,
        image_batch_size=image_batch_size,
        prompt_batch_size=prompt_batch_size,
        allow_cross_crop_prompt_decode=args.allow_cross_crop_prompt_decode,
    )
    started_at = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        proposals = generator.generate(image)
    elapsed = time.perf_counter() - started_at

    overlay_path = output_dir / "auto_masks_overlay.png"
    grid_path = output_dir / "auto_masks_grid.png"
    save_proposal_overlay(image, proposals, overlay_path, max_masks=50)
    save_proposal_grid(image, proposals, grid_path, max_masks=24, columns=6)

    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"crop_grids: {crop_grids}")
    print(f"crop_points_per_side: {crop_points_per_side}")
    print(f"image_batch_size: {image_batch_size}")
    print(f"prompt_batch_size: {prompt_batch_size}")
    print(f"allow_cross_crop_prompt_decode: {args.allow_cross_crop_prompt_decode}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"proposal_count: {len(proposals)}")
    print(f"proposal_count_by_crop_grid: {count_proposals_by_crop_grid(proposals)}")
    for idx, proposal in enumerate(proposals[:10]):
        print(
            f"proposal[{idx}]: bbox={proposal.bbox} area={proposal.area} "
            f"iou={proposal.predicted_iou:.4f} "
            f"stability={proposal.stability_score:.4f} "
            f"point={proposal.point_coords} crop_grid={proposal.crop_grid} "
            f"crop_index={proposal.crop_index} crop_box={proposal.crop_box}"
        )
    print(f"overlay_path: {overlay_path}")
    print(f"grid_path: {grid_path}")


if __name__ == "__main__":
    main()
