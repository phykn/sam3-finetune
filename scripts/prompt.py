import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROMPT_CASES = ("point", "points", "box", "point_box", "mask")


@dataclass
class PromptPaths:
    image: Path
    checkpoint: Path
    output_dir: Path

    def mask_for(self, case_name: str) -> Path:
        return self.output_dir / f"{case_name}_mask.png"

    def overlay_for(self, case_name: str) -> Path:
        return self.output_dir / f"{case_name}_overlay.png"


@dataclass
class PromptCase:
    name: str
    point_coords: np.ndarray | None
    point_labels: np.ndarray | None
    box: np.ndarray | None
    mask_input: np.ndarray | None
    prompt: dict[str, object]
    display_box: np.ndarray | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prompted Sam3Predictor smoke cases on one image."
    )
    parser.add_argument("--image", default="asset/frog_target.jpg")
    parser.add_argument("--checkpoint", default="weight/sam3.1_multiplex.pt")
    parser.add_argument("--output-dir", default="outputs/prompt")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument(
        "--case",
        action="append",
        choices=("all", *PROMPT_CASES),
        default=None,
        help="Prompt case to run. Repeat to run a subset. Defaults to all cases.",
    )
    parser.add_argument("--x", type=float, default=560.0)
    parser.add_argument("--y", type=float, default=500.0)
    parser.add_argument("--neg-x", type=float, default=300.0)
    parser.add_argument("--neg-y", type=float, default=430.0)
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        default=[380.0, 270.0, 790.0, 705.0],
        metavar=("X0", "Y0", "X1", "Y1"),
    )
    return parser.parse_args(argv)


def resolve_cases(case_values: list[str] | None) -> list[str]:
    if not case_values or "all" in case_values:
        return list(PROMPT_CASES)
    cases = []
    for case_value in case_values:
        if case_value not in cases:
            cases.append(case_value)
    return cases


def build_point_prompt(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    point_coords = np.array([[x, y]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int64)
    return point_coords, point_labels


def build_points_prompt(
    x: float,
    y: float,
    neg_x: float,
    neg_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    point_coords = np.array([[x, y], [neg_x, neg_y]], dtype=np.float32)
    point_labels = np.array([1, 0], dtype=np.int64)
    return point_coords, point_labels


def box_to_array(box: list[float]) -> np.ndarray:
    return np.asarray(box, dtype=np.float32)


def box_to_dict(box: np.ndarray) -> dict[str, float]:
    x0, y0, x1, y1 = box.tolist()
    return {
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
    }


def build_filled_box_mask(
    image_size: tuple[int, int],
    box: list[float] | np.ndarray,
) -> np.ndarray:
    width, height = image_size
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float32).tolist()
    left = max(0, min(width, int(round(x0))))
    top = max(0, min(height, int(round(y0))))
    right = max(0, min(width, int(round(x1))))
    bottom = max(0, min(height, int(round(y1))))
    if right <= left or bottom <= top:
        raise ValueError("--box must define a non-empty region")

    mask = np.zeros((height, width), dtype=np.float32)
    mask[top:bottom, left:right] = 1.0
    return mask


def describe_prompt(
    *,
    prompt_type: str = "point",
    point_coords: np.ndarray | None = None,
    point_labels: np.ndarray | None = None,
    box: np.ndarray | None = None,
    mask_input: np.ndarray | None = None,
    mask_source: str | None = None,
) -> dict[str, object]:
    prompt: dict[str, object] = {"type": prompt_type}
    if point_coords is not None and point_labels is not None:
        points = []
        for coord, label in zip(point_coords.tolist(), point_labels.tolist()):
            points.append(
                {
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "label": int(label),
                }
            )
        prompt["points"] = points
    if box is not None:
        prompt["box"] = box_to_dict(box)
    if mask_input is not None:
        prompt["mask_source"] = mask_source
        prompt["mask_input_shape"] = list(mask_input.shape)
        prompt["mask_area"] = int(mask_input.astype(bool).sum())
    return prompt


def build_prompt_case(
    case_name: str,
    args: argparse.Namespace,
    *,
    image_size: tuple[int, int],
) -> PromptCase:
    box = box_to_array(args.box)
    if case_name == "point":
        point_coords, point_labels = build_point_prompt(args.x, args.y)
        return PromptCase(
            name=case_name,
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            mask_input=None,
            prompt=describe_prompt(
                prompt_type="point",
                point_coords=point_coords,
                point_labels=point_labels,
            ),
        )
    if case_name == "points":
        point_coords, point_labels = build_points_prompt(
            args.x,
            args.y,
            args.neg_x,
            args.neg_y,
        )
        return PromptCase(
            name=case_name,
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            mask_input=None,
            prompt=describe_prompt(
                prompt_type="points",
                point_coords=point_coords,
                point_labels=point_labels,
            ),
        )
    if case_name == "box":
        return PromptCase(
            name=case_name,
            point_coords=None,
            point_labels=None,
            box=box,
            mask_input=None,
            prompt=describe_prompt(prompt_type="box", box=box),
            display_box=box,
        )
    if case_name == "point_box":
        point_coords, point_labels = build_point_prompt(args.x, args.y)
        return PromptCase(
            name=case_name,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=None,
            prompt=describe_prompt(
                prompt_type="point_box",
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
            ),
            display_box=box,
        )
    if case_name == "mask":
        mask_input = build_filled_box_mask(image_size, box)
        return PromptCase(
            name=case_name,
            point_coords=None,
            point_labels=None,
            box=None,
            mask_input=mask_input,
            prompt=describe_prompt(
                prompt_type="mask",
                box=box,
                mask_input=mask_input,
                mask_source="filled_box",
            ),
            display_box=box,
        )
    raise ValueError(f"Unknown prompt case: {case_name}")


def resolve_paths(args: argparse.Namespace, *, root: Path = ROOT) -> PromptPaths:
    output_dir = root / args.output_dir
    return PromptPaths(
        image=root / args.image,
        checkpoint=root / args.checkpoint,
        output_dir=output_dir,
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
    prompt_case: PromptCase,
    path: Path,
) -> None:
    base = image.convert("RGBA")
    mask_layer = Image.new("RGBA", base.size, (255, 0, 0, 0))
    mask_layer.putalpha(Image.fromarray(mask.astype(np.uint8) * 120, mode="L"))
    overlay = Image.alpha_composite(base, mask_layer)

    if prompt_case.mask_input is not None:
        prompt_layer = Image.new("RGBA", base.size, (0, 210, 255, 0))
        prompt_alpha = Image.fromarray(
            prompt_case.mask_input.astype(bool).astype(np.uint8) * 70,
            mode="L",
        )
        prompt_layer.putalpha(prompt_alpha)
        overlay = Image.alpha_composite(overlay, prompt_layer)

    draw = ImageDraw.Draw(overlay)
    if prompt_case.display_box is not None:
        x0, y0, x1, y1 = prompt_case.display_box.tolist()
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 255, 255), width=4)

    radius = 8
    if prompt_case.point_coords is not None and prompt_case.point_labels is not None:
        for (x, y), label in zip(
            prompt_case.point_coords.tolist(),
            prompt_case.point_labels.tolist(),
        ):
            color = (0, 255, 255, 255) if label == 1 else (255, 220, 0, 255)
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                outline=color,
                width=3,
            )
            draw.line([x - radius - 4, y, x + radius + 4, y], fill=color, width=2)
            draw.line([x, y - radius - 4, x, y + radius + 4], fill=color, width=2)

    overlay.save(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    device = resolve_device(args.device)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    from src.io.save import save_mask
    from src.predict.prompted import Sam3Predictor

    image = Image.open(paths.image).convert("RGB")
    cases = [
        build_prompt_case(case_name, args, image_size=image.size)
        for case_name in resolve_cases(args.case)
    ]

    predictor = Sam3Predictor.from_checkpoint(paths.checkpoint, device=device)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else torch.no_grad()
    )
    started_at = time.perf_counter()
    results = []
    with autocast_context:
        predictor.set_image(image)
        for prompt_case in cases:
            masks, scores, low_res = predictor.predict(
                point_coords=prompt_case.point_coords,
                point_labels=prompt_case.point_labels,
                box=prompt_case.box,
                mask_input=prompt_case.mask_input,
                multimask_output=False,
            )
            mask = masks[0].astype(bool)
            mask_path = paths.mask_for(prompt_case.name)
            overlay_path = paths.overlay_for(prompt_case.name)
            save_mask(mask, mask_path)
            save_overlay_with_prompt(image, mask, prompt_case, overlay_path)
            results.append(
                {
                    "case": prompt_case.name,
                    "prompt": prompt_case.prompt,
                    "masks_shape": list(masks.shape),
                    "low_res_shape": list(low_res.shape),
                    "scores": scores.tolist(),
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                }
            )
    elapsed = time.perf_counter() - started_at

    print(
        json.dumps(
            {
                "checkpoint": str(paths.checkpoint),
                "image": str(paths.image),
                "device": device,
                "cases": [prompt_case.name for prompt_case in cases],
                "elapsed_sec": round(elapsed, 3),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
