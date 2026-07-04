import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.video_memory_reference import ReferenceMaskResult, build_reference_mask
from scripts.video_memory_reference import predict_sam_mask_from_prompts
from src.context.matcher import ContextMatcher
from src.context.postprocess import context_prediction_to_full_mask
from src.context.types import ContextReference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a reference image+mask as visual context for target masks."
    )
    parser.add_argument("--checkpoint", default="weight/sam3.1_multiplex.pt")
    parser.add_argument("--reference-image", default="asset/frog_reference.jpg")
    parser.add_argument("--target-image", default="asset/frog_target.jpg")
    parser.add_argument(
        "--reference-box",
        nargs=4,
        type=int,
        default=[275, 420, 600, 670],
        metavar=("X0", "Y0", "X1", "Y1"),
    )
    parser.add_argument(
        "--reference-mask-source",
        choices=("sam", "box"),
        default="sam",
    )
    parser.add_argument(
        "--reference-mask",
        default=None,
        help="Optional binary mask image. When set, skips --reference-mask-source.",
    )
    parser.add_argument(
        "--reference-positive-point",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        default=None,
        help="Positive point prompt used to predict the reference mask.",
    )
    parser.add_argument(
        "--reference-negative-point",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        default=None,
        help="Negative point prompt used to predict the reference mask.",
    )
    parser.add_argument(
        "--reference-prompt-box",
        nargs=4,
        type=int,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Optional box prompt used together with reference point prompts.",
    )
    parser.add_argument("--feature-layer", default="image_embed")
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--decode-batch-size", type=int, default=16)
    parser.add_argument("--max-masks", type=int, default=10)
    parser.add_argument("--min-cell-distance", type=float, default=2.0)
    parser.add_argument("--mask-nms-thresh", type=float, default=0.7)
    parser.add_argument(
        "--candidate-score-mode",
        choices=("point", "shape"),
        default="point",
    )
    parser.add_argument("--context-score-weight", type=float, default=1.0)
    parser.add_argument("--predicted-iou-weight", type=float, default=0.1)
    parser.add_argument("--stability-score-weight", type=float, default=0.05)
    parser.add_argument("--area-score-weight", type=float, default=0.0)
    parser.add_argument(
        "--negative-context-mode",
        choices=("none", "image", "local"),
        default="local",
    )
    parser.add_argument("--negative-context-weight", type=float, default=0.75)
    parser.add_argument("--negative-context-scale", type=float, default=2.0)
    parser.add_argument("--use-reference-mask-prior", action="store_true")
    parser.add_argument("--mask-prior-scale", type=float, default=1.0)
    parser.add_argument("--mask-prior-foreground", type=float, default=4.0)
    parser.add_argument("--mask-prior-background", type=float, default=-4.0)
    parser.add_argument("--min-context-score", type=float, default=None)
    parser.add_argument("--min-mask-area", type=int, default=1)
    parser.add_argument(
        "--target-point",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        default=None,
        help="Optional target positive point. When set, skips automatic candidate search.",
    )
    parser.add_argument(
        "--output",
        default="outputs/context_prompt/context_overlay.png",
    )
    parser.add_argument(
        "--reference-overlay",
        default="outputs/context_prompt/reference_overlay.png",
    )
    return parser.parse_args()


def parse_feature_layer(value: str) -> str | int:
    if value == "image_embed":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("--feature-layer must be 'image_embed' or an integer") from exc


def load_reference_mask_image(
    path: Path, *, expected_size: tuple[int, int]
) -> np.ndarray:
    mask_image = Image.open(path).convert("L")
    if mask_image.size != expected_size:
        raise ValueError(
            f"reference mask size {mask_image.size} does not match image size {expected_size}"
        )
    return np.asarray(mask_image, dtype=np.uint8) > 127


def overlay_predictions(
    image: Image.Image,
    predictions,
    path: Path,
    *,
    max_masks: int,
) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    colors = [
        (255, 0, 180),
        (0, 190, 255),
        (255, 190, 0),
        (80, 255, 120),
        (180, 90, 255),
    ]
    for index, prediction in enumerate(predictions[:max_masks]):
        full_mask = context_prediction_to_full_mask(prediction)
        color = colors[index % len(colors)]
        color_layer = Image.new("RGBA", base.size, (*color, 0))
        alpha = Image.fromarray(full_mask.astype(np.uint8) * 120, mode="L")
        color_layer.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, color_layer)
    Image.alpha_composite(base, overlay).save(path)


def overlay_reference(image: Image.Image, mask: np.ndarray, path: Path) -> None:
    base = image.convert("RGBA")
    color = Image.new("RGBA", base.size, (255, 0, 180, 0))
    color.putalpha(Image.fromarray(mask.astype(np.uint8) * 120, mode="L"))
    Image.alpha_composite(base, color).save(path)


def main() -> None:
    args = parse_args()
    checkpoint = ROOT / args.checkpoint
    reference_image = Image.open(ROOT / args.reference_image).convert("RGB")
    target_image = Image.open(ROOT / args.target_image).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.reference_mask is not None:
        reference_mask_path = ROOT / args.reference_mask
        reference_mask = ReferenceMaskResult(
            mask=load_reference_mask_image(
                reference_mask_path,
                expected_size=reference_image.size,
            ),
            source="mask",
            score=None,
            selected_index=None,
        )
    elif (
        args.reference_positive_point is not None
        or args.reference_negative_point is not None
        or args.reference_prompt_box is not None
    ):
        reference_mask_path = None
        reference_mask = predict_sam_mask_from_prompts(
            image=reference_image,
            checkpoint=checkpoint,
            device=device,
            positive_points=args.reference_positive_point,
            negative_points=args.reference_negative_point,
            box=args.reference_prompt_box,
        )
    else:
        reference_mask_path = None
        reference_mask = build_reference_mask(
            image=reference_image,
            box=args.reference_box,
            source=args.reference_mask_source,
            checkpoint=checkpoint,
            device=device,
        )
    predictor = ContextMatcher.from_checkpoint(
        checkpoint,
        device=device,
        feature_layer=parse_feature_layer(args.feature_layer),
        candidate_count=args.candidate_count,
        decode_batch_size=args.decode_batch_size,
        max_masks=args.max_masks,
        min_cell_distance=args.min_cell_distance,
        mask_nms_thresh=args.mask_nms_thresh,
        candidate_score_mode=args.candidate_score_mode,
        context_score_weight=args.context_score_weight,
        predicted_iou_weight=args.predicted_iou_weight,
        stability_score_weight=args.stability_score_weight,
        area_score_weight=args.area_score_weight,
        negative_context_mode=args.negative_context_mode,
        negative_context_weight=args.negative_context_weight,
        negative_context_scale=args.negative_context_scale,
        use_reference_mask_prior=args.use_reference_mask_prior,
        mask_prior_scale=args.mask_prior_scale,
        mask_prior_foreground=args.mask_prior_foreground,
        mask_prior_background=args.mask_prior_background,
        min_context_score=args.min_context_score,
        min_mask_area=args.min_mask_area,
    )

    output_path = ROOT / args.output
    reference_overlay_path = ROOT / args.reference_overlay
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_overlay_path.parent.mkdir(parents=True, exist_ok=True)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else torch.no_grad()
    )
    started_at = time.perf_counter()
    with autocast_context:
        predictions = predictor.predict(
            target_image=target_image,
            references=[
                ContextReference(
                    image=reference_image,
                    mask=reference_mask.mask,
                )
            ],
            max_masks=args.max_masks,
            target_point_coords=(
                None
                if args.target_point is None
                else np.asarray(args.target_point, dtype=np.float32)
            ),
        )
    elapsed = time.perf_counter() - started_at

    overlay_reference(reference_image, reference_mask.mask, reference_overlay_path)
    overlay_predictions(
        target_image, predictions, output_path, max_masks=args.max_masks
    )

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "device": device,
                "feature_layer": args.feature_layer,
                "candidate_count": args.candidate_count,
                "decode_batch_size": args.decode_batch_size,
                "max_masks": args.max_masks,
                "candidate_score_mode": args.candidate_score_mode,
                "context_score_weight": args.context_score_weight,
                "predicted_iou_weight": args.predicted_iou_weight,
                "stability_score_weight": args.stability_score_weight,
                "area_score_weight": args.area_score_weight,
                "negative_context_mode": args.negative_context_mode,
                "negative_context_weight": args.negative_context_weight,
                "negative_context_scale": args.negative_context_scale,
                "use_reference_mask_prior": args.use_reference_mask_prior,
                "mask_prior_scale": args.mask_prior_scale,
                "elapsed_sec": round(elapsed, 3),
                "reference_mask_source": reference_mask.source,
                "reference_mask_path": (
                    None if reference_mask_path is None else str(reference_mask_path)
                ),
                "reference_mask_area": int(reference_mask.mask.sum()),
                "target_point": args.target_point,
                "prediction_count": len(predictions),
                "predictions": [
                    {
                        "bbox": prediction.bbox,
                        "area": prediction.area,
                        "point_coords": prediction.point_coords,
                        "context_score": prediction.context_score,
                        "predicted_iou": prediction.predicted_iou,
                        "stability_score": prediction.stability_score,
                        "area_score": prediction.area_score,
                        "score": prediction.score,
                    }
                    for prediction in predictions
                ],
                "reference_overlay": str(reference_overlay_path),
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
