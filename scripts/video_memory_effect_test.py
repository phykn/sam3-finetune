import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.memory_inference import VideoMemoryInference, MemoryReference
from scripts.video_memory_reference import build_reference_mask, make_box_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare correct and control reference masks for video memory."
    )
    parser.add_argument("--checkpoint", default="weight/sam3.1_multiplex.pt")
    parser.add_argument("--reference-image", default="asset/frog_reference.jpg")
    parser.add_argument("--target-image", default="asset/frog_target.jpg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument(
        "--frog-box",
        type=int,
        nargs=4,
        default=[245, 420, 650, 890],
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Reference frog mask box in reference-image pixels.",
    )
    parser.add_argument(
        "--reference-mask-source",
        choices=("sam", "box"),
        default="sam",
        help="Use a predicted SAM mask from --frog-box or the raw box mask as the correct reference.",
    )
    parser.add_argument(
        "--control-box",
        type=int,
        nargs=4,
        default=[70, 70, 360, 310],
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Reference background/control mask box in reference-image pixels.",
    )
    parser.add_argument(
        "--target-frog-box",
        type=int,
        nargs=4,
        default=[360, 230, 805, 720],
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Weak target frog box for overlap diagnostics.",
    )
    parser.add_argument(
        "--reference-repeat",
        type=int,
        default=1,
        help="Repeat the same reference frame this many times.",
    )
    parser.add_argument("--output-dir", default="outputs/video_memory_effect")
    return parser.parse_args()


def box_mask(size: tuple[int, int], box: list[int]) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
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


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    color_arr = np.asarray(color, dtype=np.float32)
    alpha = 0.50
    rgb[mask] = rgb[mask] * (1.0 - alpha) + color_arr * alpha
    rgb[mask_edges(mask)] = np.array([255, 255, 255], dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def run_case(
    predictor: VideoMemoryInference,
    name: str,
    reference_image: Image.Image,
    target_image: Image.Image,
    reference_mask: np.ndarray,
    target_box_mask: np.ndarray,
    obj_id: int,
    reference_repeat: int,
    output_dir: Path,
) -> tuple[dict[str, object], np.ndarray]:
    references = [
        MemoryReference(image=reference_image, mask=reference_mask, obj_id=obj_id)
        for _ in range(reference_repeat)
    ]
    prediction = predictor.predict(target_image=target_image, references=references)
    mask = prediction.masks[0, 0].astype(bool)
    overlay_color = (255, 0, 180) if name == "correct_reference" else (0, 180, 255)
    overlay = overlay_mask(target_image, mask, color=overlay_color)
    output_path = output_dir / f"{name}.png"
    overlay.save(output_path)
    return {
        "name": name,
        "frame_index": prediction.frame_index,
        "obj_ids": prediction.obj_ids,
        "score": prediction.scores.tolist(),
        "mask_area": int(mask.sum()),
        "mask_area_ratio": float(mask.mean()),
        "mask_bbox": mask_bbox(mask),
        "weak_target_box_iou": mask_iou(mask, target_box_mask),
        "output": str(output_path),
    }, mask


def main() -> None:
    args = parse_args()
    if args.reference_repeat <= 0:
        raise ValueError("--reference-repeat must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_image = Image.open(args.reference_image).convert("RGB")
    target_image = Image.open(args.target_image).convert("RGB")
    target_box = box_mask(target_image.size, args.target_frog_box)
    target_box_overlay = overlay_mask(target_image, target_box, color=(255, 190, 0))
    target_box_overlay.save(output_dir / "weak_target_box.png")

    correct_mask_result = build_reference_mask(
        image=reference_image,
        box=args.frog_box,
        source=args.reference_mask_source,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    correct_mask = correct_mask_result.mask
    control_mask = make_box_mask(reference_image, args.control_box)
    correct_reference_overlay_path = output_dir / "correct_reference_mask.png"
    control_reference_overlay_path = output_dir / "control_reference_mask.png"
    overlay_mask(reference_image, correct_mask, color=(255, 0, 180)).save(
        correct_reference_overlay_path
    )
    overlay_mask(reference_image, control_mask, color=(0, 180, 255)).save(
        control_reference_overlay_path
    )

    predictor = VideoMemoryInference.from_checkpoint(
        args.checkpoint,
        device=args.device,
    )
    correct_result, correct_pred_mask = run_case(
        predictor,
        "correct_reference",
        reference_image,
        target_image,
        correct_mask,
        target_box,
        args.obj_id,
        args.reference_repeat,
        output_dir,
    )
    control_result, control_pred_mask = run_case(
        predictor,
        "control_reference",
        reference_image,
        target_image,
        control_mask,
        target_box,
        args.obj_id,
        args.reference_repeat,
        output_dir,
    )
    pairwise_iou = mask_iou(correct_pred_mask, control_pred_mask)
    results = [correct_result, control_result]
    report = {
        "plan": [
            "Use Photo 1 as the reference frame and Photo 2 as the target frame.",
            "Run one case with a frog reference mask and one control case with a background reference mask.",
            "Compare target overlays plus weak-box IoU, mask area, bbox, score, and pairwise output IoU.",
        ],
        "reference_image": args.reference_image,
        "target_image": args.target_image,
        "reference_repeat": args.reference_repeat,
        "reference_mask_source": correct_mask_result.source,
        "reference_selected_index": correct_mask_result.selected_index,
        "reference_score": correct_mask_result.score,
        "reference_refined_score": correct_mask_result.refined_score,
        "reference_mask_area": int(correct_mask.sum()),
        "correct_reference_mask_overlay": str(correct_reference_overlay_path),
        "control_reference_mask_overlay": str(control_reference_overlay_path),
        "frog_box": args.frog_box,
        "control_box": args.control_box,
        "target_frog_box": args.target_frog_box,
        "correct_vs_control_mask_iou": pairwise_iou,
        "results": results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
