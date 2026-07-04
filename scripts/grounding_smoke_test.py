import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict.grounding.inference import GroundingInference
from src.predict.grounding.postprocess import (
    filter_grounding_prediction,
    label_grounding_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM-free cached-visual image grounding."
    )
    parser.add_argument("--checkpoint", default="weight/sam3.1_multiplex.pt")
    parser.add_argument("--visual-cache", default="weight/visual_language_sam31.pt")
    parser.add_argument("--image", default="asset/frog_target.jpg")
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        default=[850.0, 610.0, 1270.0, 800.0],
        metavar=("X0", "Y0", "X1", "Y1"),
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-masks", type=int, default=8)
    parser.add_argument("--concept-id", type=int, default=0)
    parser.add_argument("--object-id-start", type=int, default=0)
    parser.add_argument(
        "--mask-nms-thresh",
        type=float,
        default=None,
        help="Optional model-output mask NMS threshold. No color/position filtering.",
    )
    parser.add_argument(
        "--output",
        default="outputs/grounding/cached_visual_grounding.png",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON metadata output with concept_id and object_id.",
    )
    return parser.parse_args()


def save_overlay(
    image: Image.Image,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    path: Path,
    *,
    max_masks: int,
    concept_ids: np.ndarray | None = None,
    object_ids: np.ndarray | None = None,
) -> None:
    base = image.convert("RGBA")
    colors = [
        (255, 40, 40, 115),
        (40, 120, 255, 115),
        (255, 190, 0, 120),
        (190, 40, 255, 115),
        (0, 210, 190, 120),
        (255, 90, 170, 115),
        (255, 255, 255, 100),
        (30, 30, 30, 100),
    ]
    order = np.argsort(-scores)
    h, w = image.height, image.width
    for rank, index in enumerate(order[:max_masks]):
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[masks[index].astype(bool)] = colors[rank % len(colors)]
        base = Image.alpha_composite(base, Image.fromarray(arr, "RGBA"))
    draw = ImageDraw.Draw(base)
    for rank, index in enumerate(order[:max_masks]):
        x0, y0, x1, y1 = boxes[index].tolist()
        color = colors[rank % len(colors)][:3] + (255,)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        if concept_ids is None or object_ids is None:
            label = f"{rank + 1}:{scores[index]:.2f}"
        else:
            label = (
                f"c{int(concept_ids[index])}/o{int(object_ids[index])}:"
                f"{scores[index]:.2f}"
            )
        draw.text((x0 + 3, y0 + 3), label, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(path)


def save_metadata(
    path: Path,
    *,
    labeled,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instances": [
            {
                "concept_id": int(labeled.concept_ids[index]),
                "object_id": int(labeled.object_ids[index]),
                "score": float(labeled.prediction.scores[index]),
                "box_xyxy": [
                    float(value)
                    for value in labeled.prediction.boxes_xyxy[index].tolist()
                ],
                "mask_area": int(labeled.prediction.masks[index].sum()),
            }
            for index in range(len(labeled.prediction.scores))
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(ROOT / args.image).convert("RGB")
    started_at = time.perf_counter()
    predictor = GroundingInference.from_checkpoint(
        ROOT / args.checkpoint,
        ROOT / args.visual_cache,
        device=device,
        confidence_threshold=args.threshold,
    )
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else torch.no_grad()
    )
    with autocast_context:
        prediction = predictor.predict(
            image,
            boxes_xyxy=np.asarray(args.box, dtype=np.float32),
            confidence_threshold=args.threshold,
        )
    raw_mask_count = len(prediction.scores)
    if args.mask_nms_thresh is not None:
        prediction = filter_grounding_prediction(
            prediction,
            score_threshold=args.threshold,
            mask_nms_thresh=args.mask_nms_thresh,
            max_masks=args.max_masks,
        )
    labeled = label_grounding_prediction(
        prediction,
        concept_id=args.concept_id,
        object_id_start=args.object_id_start,
    )
    elapsed = time.perf_counter() - started_at
    output = ROOT / args.output
    save_overlay(
        image,
        prediction.masks,
        prediction.boxes_xyxy,
        prediction.scores,
        output,
        max_masks=args.max_masks,
        concept_ids=labeled.concept_ids,
        object_ids=labeled.object_ids,
    )
    json_output = None if args.json_output is None else ROOT / args.json_output
    if json_output is not None:
        save_metadata(json_output, labeled=labeled)
    print(f"device: {device}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"raw_mask_count: {raw_mask_count}")
    print(f"mask_count: {len(prediction.scores)}")
    print(f"concept_id: {args.concept_id}")
    print(f"object_ids: {labeled.object_ids.tolist()}")
    print(
        f"scores_top: {prediction.scores[np.argsort(-prediction.scores)[:10]].tolist()}"
    )
    print(f"output: {output}")
    if json_output is not None:
        print(f"json_output: {json_output}")


if __name__ == "__main__":
    main()
