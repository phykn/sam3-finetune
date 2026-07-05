import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single positive point prompt with Sam3Predictor."
    )
    parser.add_argument("--image", default="asset/sample.jpg")
    parser.add_argument("--checkpoint", default="weight/sam3.1_multiplex.pt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--x", type=float, default=195.0)
    parser.add_argument("--y", type=float, default=295.0)
    return parser.parse_args(argv)


def build_point_prompt(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    point_coords = np.array([[x, y]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int64)
    return point_coords, point_labels


def describe_prompt(
    point_coords: np.ndarray,
    point_labels: np.ndarray,
) -> dict[str, object]:
    points = []
    for coord, label in zip(point_coords.tolist(), point_labels.tolist()):
        points.append(
            {
                "x": float(coord[0]),
                "y": float(coord[1]),
                "label": int(label),
            }
        )
    return {
        "type": "single_positive_point",
        "points": points,
    }


def resolve_paths(args: argparse.Namespace, *, root: Path = ROOT) -> SimpleNamespace:
    output_dir = root / args.output_dir
    return SimpleNamespace(
        image=root / args.image,
        checkpoint=root / args.checkpoint,
        mask=output_dir / "prompt_mask.png",
        overlay=output_dir / "prompt_overlay.png",
    )


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when --device cuda is selected.")
    return device


def save_overlay_with_prompt(
    image: Image.Image,
    mask: np.ndarray,
    point_coords: np.ndarray,
    path: Path,
) -> None:
    base = image.convert("RGBA")
    mask_layer = Image.new("RGBA", base.size, (255, 0, 0, 0))
    mask_layer.putalpha(Image.fromarray(mask.astype(np.uint8) * 120, mode="L"))
    overlay = Image.alpha_composite(base, mask_layer)

    draw = ImageDraw.Draw(overlay)
    radius = 8
    for x, y in point_coords.tolist():
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=(0, 255, 255, 255),
            width=3,
        )
        draw.line(
            [x - radius - 4, y, x + radius + 4, y], fill=(0, 255, 255, 255), width=2
        )
        draw.line(
            [x, y - radius - 4, x, y + radius + 4], fill=(0, 255, 255, 255), width=2
        )

    overlay.save(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    device = resolve_device(args.device)
    paths.mask.parent.mkdir(parents=True, exist_ok=True)

    from src.io.save import save_mask
    from src.predict.prompted import Sam3Predictor

    image = Image.open(paths.image).convert("RGB")
    point_coords, point_labels = build_point_prompt(args.x, args.y)
    prompt = describe_prompt(point_coords, point_labels)

    predictor = Sam3Predictor.from_checkpoint(paths.checkpoint, device=device)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else torch.no_grad()
    )
    started_at = time.perf_counter()
    with autocast_context:
        predictor.set_image(image)
        masks, scores, low_res = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
    elapsed = time.perf_counter() - started_at

    mask = masks[0].astype(bool)
    save_mask(mask, paths.mask)
    save_overlay_with_prompt(image, mask, point_coords, paths.overlay)

    print(
        json.dumps(
            {
                "checkpoint": str(paths.checkpoint),
                "image": str(paths.image),
                "device": device,
                "prompt": prompt,
                "elapsed_sec": round(elapsed, 3),
                "masks_shape": list(masks.shape),
                "low_res_shape": list(low_res.shape),
                "scores": scores.tolist(),
                "mask_path": str(paths.mask),
                "overlay_path": str(paths.overlay),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
