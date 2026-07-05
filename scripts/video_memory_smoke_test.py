import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.video_memory_reference import build_reference_mask, ReferenceMaskResult
from src.predict.next_frame import MemoryReference, NextFramePredictor


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
        help="Reference prompt box in original image pixels. Defaults to center box.",
    )
    parser.add_argument(
        "--reference-mask-source",
        choices=("sam", "box"),
        default="sam",
        help="Use a predicted SAM mask from --mask-box or the raw box mask as reference.",
    )
    parser.add_argument(
        "--reference-overlay",
        default=None,
        help="Optional path for a reference mask overlay.",
    )
    parser.add_argument(
        "--target-point",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Optional positive point on the target image in original pixels.",
    )
    parser.add_argument(
        "--target-point-mode",
        choices=("interaction", "memory"),
        default="interaction",
        help=(
            "interaction stores the target point as a conditioning prompt; memory "
            "combines reference memory propagation with the target point."
        ),
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


def mask_edges(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
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

    mask_results = []
    mask_cache: dict[tuple[str, tuple[int, int]], ReferenceMaskResult] = {}
    for path, image in zip(reference_paths, reference_images):
        cache_key = (str(Path(path).resolve()), image.size)
        if cache_key not in mask_cache:
            mask_cache[cache_key] = build_reference_mask(
                image=image,
                box=args.mask_box,
                source=args.reference_mask_source,
                checkpoint=args.checkpoint,
                device=args.device,
            )
        mask_results.append(mask_cache[cache_key])

    first_mask = mask_results[0].mask
    if args.reference_overlay is not None:
        reference_overlay_path = Path(args.reference_overlay)
        reference_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_mask(reference_images[0], first_mask).save(reference_overlay_path)
    else:
        reference_overlay_path = None

    references = [
        MemoryReference(image=image, mask=result.mask, obj_id=args.obj_id)
        for image, result in zip(reference_images, mask_results)
    ]

    predictor = NextFramePredictor.from_checkpoint(
        args.checkpoint,
        device=args.device,
    )
    if args.target_point is None:
        target_point_coords = None
        target_point_labels = None
    else:
        target_point_coords = np.asarray([args.target_point], dtype=np.float32)
        target_point_labels = np.asarray([1], dtype=np.int64)

    prediction = predictor.predict(
        target_image=target_image,
        references=references,
        target_point_coords=target_point_coords,
        target_point_labels=target_point_labels,
        target_point_mode=args.target_point_mode,
    )

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
            "reference_mask_source": mask_results[0].source,
            "reference_mask_area": int(first_mask.sum()),
            "reference_score": mask_results[0].score,
            "reference_refined_score": mask_results[0].refined_score,
            "reference_selected_index": mask_results[0].selected_index,
            "reference_overlay": (
                str(reference_overlay_path)
                if reference_overlay_path is not None
                else None
            ),
            "target_point": args.target_point,
            "target_point_mode": args.target_point_mode,
        }
    )


if __name__ == "__main__":
    main()
