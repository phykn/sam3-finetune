from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auto_mask_generator import (
    Sam3AutomaticMaskGenerator,
    save_proposal_grid,
    save_proposal_overlay,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    image_path = ROOT / "asset" / "sample.jpg"
    checkpoint_path = ROOT / "weight" / "sam3.1_multiplex.pt"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    generator = Sam3AutomaticMaskGenerator.from_checkpoint(
        checkpoint_path,
        device="cuda",
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.75,
        box_nms_thresh=0.7,
        max_masks=100,
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        proposals = generator.generate(image)

    overlay_path = output_dir / "auto_masks_overlay.png"
    grid_path = output_dir / "auto_masks_grid.png"
    save_proposal_overlay(image, proposals, overlay_path, max_masks=50)
    save_proposal_grid(image, proposals, grid_path, max_masks=24, columns=6)

    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"proposal_count: {len(proposals)}")
    for idx, proposal in enumerate(proposals[:10]):
        print(
            f"proposal[{idx}]: bbox={proposal.bbox} area={proposal.area} "
            f"iou={proposal.predicted_iou:.4f} "
            f"stability={proposal.stability_score:.4f} "
            f"point={proposal.point_coords}"
        )
    print(f"overlay_path: {overlay_path}")
    print(f"grid_path: {grid_path}")


if __name__ == "__main__":
    main()
