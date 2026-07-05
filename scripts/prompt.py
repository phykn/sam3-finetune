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
POSITIVE_POINT = (560.0, 500.0)
NEGATIVE_POINT = (300.0, 430.0)
DEFAULT_BOX = (380.0, 270.0, 790.0, 705.0)


@dataclass
class PromptPaths:
    image: Path
    checkpoint: Path
    output_dir: Path

    @property
    def output(self) -> Path:
        return self.output_dir / "prompt.png"


@dataclass
class PromptCase:
    name: str
    point_coords: np.ndarray | None
    point_labels: np.ndarray | None
    box: np.ndarray | None
    mask_input: np.ndarray | None
    prompt: dict[str, object]
    display_box: np.ndarray | None = None


@dataclass
class PromptResult:
    case: PromptCase
    mask: np.ndarray
    score: float
    mask_shape: tuple[int, ...]
    low_res_shape: tuple[int, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed prompt cases on one image.")
    parser.add_argument("--image", default="asset/frog_target.jpg")
    parser.add_argument("--checkpoint", default="weight/sam3.1_multiplex.pt")
    parser.add_argument("--output-dir", default="outputs/prompt")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace, *, root: Path = ROOT) -> PromptPaths:
    return PromptPaths(
        image=root / args.image,
        checkpoint=root / args.checkpoint,
        output_dir=root / args.output_dir,
    )


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when --device cuda is selected.")
    return device


def build_point_prompt(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[x, y]], dtype=np.float32),
        np.array([1], dtype=np.int64),
    )


def build_points_prompt(
    x: float,
    y: float,
    neg_x: float,
    neg_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[x, y], [neg_x, neg_y]], dtype=np.float32),
        np.array([1, 0], dtype=np.int64),
    )


def build_prompt_cases(image_size: tuple[int, int]) -> list[PromptCase]:
    point_coords, point_labels = build_point_prompt(*POSITIVE_POINT)
    points_coords, points_labels = build_points_prompt(
        *POSITIVE_POINT,
        *NEGATIVE_POINT,
    )
    box = box_to_array(DEFAULT_BOX)
    mask_input = build_filled_box_mask(image_size, box)
    return [
        PromptCase(
            name="point",
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            mask_input=None,
            prompt=describe_prompt(
                prompt_type="point",
                point_coords=point_coords,
                point_labels=point_labels,
            ),
        ),
        PromptCase(
            name="points",
            point_coords=points_coords,
            point_labels=points_labels,
            box=None,
            mask_input=None,
            prompt=describe_prompt(
                prompt_type="points",
                point_coords=points_coords,
                point_labels=points_labels,
            ),
        ),
        PromptCase(
            name="box",
            point_coords=None,
            point_labels=None,
            box=box,
            mask_input=None,
            prompt=describe_prompt(prompt_type="box", box=box),
            display_box=box,
        ),
        PromptCase(
            name="point_box",
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
        ),
        PromptCase(
            name="mask",
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
        ),
    ]


def box_to_array(box: tuple[float, float, float, float]) -> np.ndarray:
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
    box: tuple[float, float, float, float] | list[float] | np.ndarray,
) -> np.ndarray:
    width, height = image_size
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float32).tolist()
    left = max(0, min(width, int(round(x0))))
    top = max(0, min(height, int(round(y0))))
    right = max(0, min(width, int(round(x1))))
    bottom = max(0, min(height, int(round(y1))))
    if right <= left or bottom <= top:
        raise ValueError("default box must define a non-empty region")

    mask = np.zeros((height, width), dtype=np.float32)
    mask[top:bottom, left:right] = 1.0
    return mask


def describe_prompt(
    *,
    prompt_type: str,
    point_coords: np.ndarray | None = None,
    point_labels: np.ndarray | None = None,
    box: np.ndarray | None = None,
    mask_input: np.ndarray | None = None,
    mask_source: str | None = None,
) -> dict[str, object]:
    prompt: dict[str, object] = {"type": prompt_type}
    if point_coords is not None and point_labels is not None:
        prompt["points"] = [
            {
                "x": float(coord[0]),
                "y": float(coord[1]),
                "label": int(label),
            }
            for coord, label in zip(point_coords.tolist(), point_labels.tolist())
        ]
    if box is not None:
        prompt["box"] = box_to_dict(box)
    if mask_input is not None:
        prompt["mask_source"] = mask_source
        prompt["mask_input_shape"] = list(mask_input.shape)
        prompt["mask_area"] = int(mask_input.astype(bool).sum())
    return prompt


def save_prompt_visualization(
    image: Image.Image,
    results: list[PromptResult],
    path: Path,
) -> None:
    panel_width = 320
    columns = 3
    gap = 12
    padding = 16
    panels = [_make_result_panel(image, result, panel_width) for result in results]
    rows = max(1, int(np.ceil(len(panels) / columns)))
    panel_height = max(panel.height for panel in panels) if panels else 1
    canvas_width = padding * 2 + columns * panel_width + (columns - 1) * gap
    canvas_height = padding * 2 + rows * panel_height + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), (246, 246, 242))
    for index, panel in enumerate(panels):
        x = padding + (index % columns) * (panel_width + gap)
        y = padding + (index // columns) * (panel_height + gap)
        canvas.paste(panel, (x, y))
    canvas.save(path)


def _make_result_panel(
    image: Image.Image,
    result: PromptResult,
    width: int,
) -> Image.Image:
    body = _resize_image(image, width)
    overlay = body.convert("RGBA")
    mask = _resize_mask(result.mask, body.size)
    mask_layer = Image.new("RGBA", body.size, (230, 57, 70, 255))
    mask_layer.putalpha(Image.fromarray(mask.astype(np.uint8) * 110, mode="L"))
    overlay = Image.alpha_composite(overlay, mask_layer)

    prompt_case = result.case
    if prompt_case.mask_input is not None:
        prompt_mask = _resize_mask(prompt_case.mask_input.astype(bool), body.size)
        prompt_layer = Image.new("RGBA", body.size, (0, 210, 255, 255))
        prompt_layer.putalpha(
            Image.fromarray(prompt_mask.astype(np.uint8) * 70, mode="L")
        )
        overlay = Image.alpha_composite(overlay, prompt_layer)

    draw = ImageDraw.Draw(overlay)
    scale_x = body.width / image.width
    scale_y = body.height / image.height
    if prompt_case.display_box is not None:
        draw.rectangle(
            _scale_bbox(prompt_case.display_box, scale_x, scale_y),
            outline=(0, 210, 255, 255),
            width=3,
        )
    if prompt_case.point_coords is not None and prompt_case.point_labels is not None:
        for (x, y), label in zip(
            prompt_case.point_coords.tolist(),
            prompt_case.point_labels.tolist(),
        ):
            color = (0, 210, 255, 255) if label == 1 else (255, 220, 0, 255)
            px = x * scale_x
            py = y * scale_y
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=color, width=3)
            draw.line((px - 8, py, px + 8, py), fill=color, width=2)
            draw.line((px, py - 8, px, py + 8), fill=color, width=2)

    return _with_header(
        overlay.convert("RGB"),
        f"{prompt_case.name} score={result.score:.3f}",
    )


def _resize_image(image: Image.Image, width: int) -> Image.Image:
    height = max(1, int(round(image.height * width / image.width)))
    return image.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                size,
                Image.Resampling.NEAREST,
            )
        )
        > 127
    )


def _scale_bbox(
    bbox: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox.tolist()
    return (
        int(round(x0 * scale_x)),
        int(round(y0 * scale_y)),
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
    )


def _with_header(body: Image.Image, title: str) -> Image.Image:
    header_height = 32
    panel = Image.new("RGB", (body.width, body.height + header_height), (30, 30, 30))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, header_height), fill=(246, 246, 242))
    draw.text((10, 9), title, fill=(30, 30, 30))
    panel.paste(body, (0, header_height))
    return panel


def summarize_result(result: PromptResult) -> dict[str, object]:
    return {
        "case": result.case.name,
        "prompt": result.case.prompt,
        "score": float(result.score),
        "mask_shape": list(result.mask_shape),
        "low_res_shape": list(result.low_res_shape),
        "mask_area": int(result.mask.sum()),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    device = resolve_device(args.device)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    from src.predict.prompted import Sam3Predictor

    image = Image.open(paths.image).convert("RGB")
    prompt_cases = build_prompt_cases(image.size)
    predictor = Sam3Predictor.from_checkpoint(paths.checkpoint, device=device)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else torch.no_grad()
    )
    started_at = time.perf_counter()
    results: list[PromptResult] = []
    with torch.inference_mode(), autocast_context:
        embedding = predictor.encode_image(image)
        for prompt_case in prompt_cases:
            masks, scores, low_res = predictor.predict_from_embedding(
                embedding,
                point_coords=prompt_case.point_coords,
                point_labels=prompt_case.point_labels,
                box=prompt_case.box,
                mask_input=prompt_case.mask_input,
                multimask_output=False,
            )
            results.append(
                PromptResult(
                    case=prompt_case,
                    mask=masks[0].astype(bool),
                    score=float(np.asarray(scores).reshape(-1)[0]),
                    mask_shape=tuple(np.asarray(masks).shape),
                    low_res_shape=tuple(np.asarray(low_res).shape),
                )
            )
    elapsed = time.perf_counter() - started_at
    save_prompt_visualization(image, results, paths.output)

    print(
        json.dumps(
            {
                "checkpoint": str(paths.checkpoint),
                "image": str(paths.image),
                "device": device,
                "cases": [prompt_case.name for prompt_case in prompt_cases],
                "elapsed_sec": round(elapsed, 3),
                "results": [summarize_result(result) for result in results],
                "output": str(paths.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
