from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory_predictor import Sam3MemoryPredictor, Sam3MemoryReference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test the src SAM3.1 video memory predictor."
    )
    parser.add_argument(
        "--checkpoint",
        default="weight/sam3.1_multiplex.pt",
        help="Local SAM3.1 multiplex checkpoint path.",
    )
    parser.add_argument(
        "--image",
        default="asset/sample.jpg",
        help="Image used as reference and target pseudo-video frames.",
    )
    parser.add_argument(
        "--reference-count",
        type=int,
        default=2,
        help="Number of reference frames to add before the target frame.",
    )
    parser.add_argument(
        "--obj-id",
        type=int,
        default=1,
        help="Object id shared by all reference masks.",
    )
    parser.add_argument(
        "--mask-box",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="Reference mask box in original image pixels. Defaults to center box.",
    )
    parser.add_argument(
        "--output",
        default="outputs/video_memory_smoke.png",
        help="Overlay output path.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. The tracker path expects CUDA for real inference.",
    )
    return parser.parse_args()


def make_box_mask(image: Image.Image, box: list[int] | None) -> np.ndarray:
    width, height = image.size
    if box is None:
        x0 = int(width * 0.25)
        y0 = int(height * 0.20)
        x1 = int(width * 0.75)
        y1 = int(height * 0.85)
    else:
        x0, y0, x1, y1 = box
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.array([0, 210, 120], dtype=np.float32)
    alpha = 0.45
    rgb[mask] = rgb[mask] * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def main() -> None:
    args = parse_args()
    if args.reference_count <= 0:
        raise ValueError("--reference-count must be positive")

    image = Image.open(args.image).convert("RGB")
    mask = make_box_mask(image, args.mask_box)
    references = [
        Sam3MemoryReference(image=image, mask=mask, obj_id=args.obj_id)
        for _ in range(args.reference_count)
    ]

    predictor = Sam3MemoryPredictor.from_checkpoint(
        args.checkpoint,
        device=args.device,
    )
    prediction = predictor.predict(target_image=image, references=references)

    if prediction.masks.size == 0:
        raise RuntimeError("video memory smoke produced no masks")
    target_mask = prediction.masks[0, 0].astype(bool)
    output = overlay_mask(image, target_mask)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)

    print(
        {
            "frame_index": prediction.frame_index,
            "obj_ids": prediction.obj_ids,
            "mask_shape": list(prediction.masks.shape),
            "scores": prediction.scores.tolist(),
            "output": str(output_path),
            "loaded_keys": predictor.load_report.loaded_keys
            if predictor.load_report
            else None,
            "ignored_keys": predictor.load_report.ignored_keys
            if predictor.load_report
            else None,
        }
    )


if __name__ == "__main__":
    main()
