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
        default=None,
        help="Legacy image used as both reference and target when explicit paths are omitted.",
    )
    parser.add_argument(
        "--reference-image",
        action="append",
        default=None,
        help="Reference image path. Repeat to provide multiple reference frames.",
    )
    parser.add_argument(
        "--target-image",
        default=None,
        help="Target image path. Defaults to asset/frog_target.jpg.",
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


def mask_edges(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & ~interior


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 180),
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    color_arr = np.asarray(color, dtype=np.float32)
    alpha = 0.50
    rgb[mask] = rgb[mask] * (1.0 - alpha) + color_arr * alpha
    rgb[mask_edges(mask)] = np.array([255, 255, 255], dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def main() -> None:
    args = parse_args()
    if args.reference_count <= 0:
        raise ValueError("--reference-count must be positive")

    default_reference = "asset/frog_reference.jpg"
    default_target = "asset/frog_target.jpg"
    if args.reference_image is not None:
        reference_paths = args.reference_image
    else:
        reference_paths = [args.image or default_reference]
    if len(reference_paths) == 1 and args.reference_count > 1:
        reference_paths = reference_paths * args.reference_count
    target_path = args.target_image or args.image or default_target

    reference_images = [Image.open(path).convert("RGB") for path in reference_paths]
    target_image = Image.open(target_path).convert("RGB")
    mask = make_box_mask(reference_images[0], args.mask_box)
    references = [
        Sam3MemoryReference(image=image, mask=mask, obj_id=args.obj_id)
        for image in reference_images
    ]

    predictor = Sam3MemoryPredictor.from_checkpoint(
        args.checkpoint,
        device=args.device,
    )
    prediction = predictor.predict(target_image=target_image, references=references)

    if prediction.masks.size == 0:
        raise RuntimeError("video memory smoke produced no masks")
    target_mask = prediction.masks[0, 0].astype(bool)
    output = overlay_mask(target_image, target_mask)
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
