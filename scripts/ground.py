import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict.ground import GroundPredictor  # noqa: E402
from src.predict.single import SinglePredictor  # noqa: E402

WEIGHT = ROOT / "weight" / "sam3.1_multiplex.pt"
VISUAL = ROOT / "weight" / "visual_token.pt"
REF = ROOT / "asset" / "frog_ref.jpg"
TARGET = ROOT / "asset" / "frog_tgt.jpg"
OUT = ROOT / "outputs" / "ground" / "frog_flower_ground.png"

CONCEPTS = [
    {
        "name": "frog",
        "points": np.array([[465.0, 610.0]], dtype=np.float32),
        "color": (255, 60, 60),
    },
    {
        "name": "flower",
        "points": np.array(
            [
                [20.0, 1065.0],
                [119.0, 623.0],
                [270.0, 670.0],
                [817.0, 1135.0],
            ],
            dtype=np.float32,
        ),
        "color": (255, 180, 0),
    },
]
POINT_LABEL = np.array([1], dtype=np.int32)
POINT_COLOR = (40, 120, 255)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_image = Image.open(REF).convert("RGB")
    target = Image.open(TARGET).convert("RGB")

    refs = segment_refs(ref_image, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    predictor = GroundPredictor.from_path(
        WEIGHT,
        VISUAL,
        device=device,
        top_k=1,
        sim_thr=0.75,
    )
    encoded = [
        predictor.encode_ref(ref_image, mask=ref["masks"], name=ref["name"])
        for ref in refs
    ]
    out = predictor.predict(target, encoded)
    for item in out.values():
        item.pop("raw", None)

    del encoded, predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    refine(target, out, device)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    make_sheet(ref_image, refs, target, out).save(OUT)
    print_result(device, refs, out)


def segment_refs(image, device):
    predictor = SinglePredictor.from_path(WEIGHT, {"device": device})
    refs = []
    for ref in CONCEPTS:
        masks = []
        scores = []
        for point in ref["points"]:
            out = predictor.predict(
                image,
                point_coords=point[None],
                point_labels=POINT_LABEL,
                multimask=True,
            )
            index = int(np.argmax(out["scores"]))
            masks.append(out["masks"][index])
            scores.append(float(out["scores"][index]))
        refs.append(
            {
                **ref,
                "masks": np.asarray(masks, dtype=bool),
                "scores": np.asarray(scores, dtype=np.float32),
            }
        )
    return refs


def refine(image, out, device):
    predictor = SinglePredictor.from_path(WEIGHT, {"device": device})
    for item in out.values():
        masks = []
        scores = []
        for logit in item["logits"]:
            refined = predictor.refine(image, logit)
            masks.append(refined["masks"][0])
            scores.append(float(refined["scores"][0]))
        item["refined_masks"] = np.asarray(masks, dtype=bool)
        item["refined_scores"] = np.asarray(scores, dtype=np.float32)


def make_sheet(ref_image, refs, target, out):
    panels = [
        draw_points(ref_image),
        draw_ref_masks(ref_image, refs),
        draw_target(target, out, refined=False),
        draw_target(target, out, refined=True),
    ]

    cell_w = max(panel.width for panel in panels)
    cell_h = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (cell_w * 2, cell_h * 2), "white")
    for index, panel in enumerate(panels):
        x = cell_w * (index % 2)
        y = cell_h * (index // 2)
        sheet.paste(panel, (x, y))
    return sheet


def draw_points(image):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = load_font(32)
    for ref in CONCEPTS:
        for index, point in enumerate(ref["points"]):
            x, y = point
            radius = 34
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=POINT_COLOR,
                outline=(255, 255, 255),
                width=5,
            )
            text = ref["name"] if len(ref["points"]) == 1 else f"{ref['name']} {index}"
            draw.text(
                (x + 42, y - 24),
                text,
                fill=(255, 255, 255),
                font=font,
                stroke_width=3,
                stroke_fill=(0, 0, 0),
            )
    return image


def draw_ref_masks(image, refs):
    image = image.copy()
    for ref in refs:
        for mask in ref["masks"]:
            overlay(image, mask, ref["color"])
    for ref in refs:
        for index, mask in enumerate(ref["masks"]):
            text = f"{ref['name']} {float(ref['scores'][index]):.3f}"
            if len(ref["masks"]) > 1:
                text = f"{ref['name']} {index} {float(ref['scores'][index]):.3f}"
            draw_box(image, find_box(mask), text, ref["color"])
    return image


def draw_target(image, out, refined):
    image = image.copy()
    for ref in CONCEPTS:
        item = out[ref["name"]]
        key = "refined_masks" if refined else "masks"
        for mask in item[key]:
            overlay(image, mask, ref["color"])

    for ref in CONCEPTS:
        item = out[ref["name"]]
        if refined:
            boxes = [find_box(mask) for mask in item["refined_masks"]]
            scores = item["refined_scores"]
        else:
            boxes = item["boxes"]
            scores = item.get("similarities", item["scores"])

        for index, box in enumerate(boxes):
            draw_box(
                image,
                box,
                f"{ref['name']} {float(scores[index]):.3f}",
                ref["color"],
            )
    return image


def find_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_box(image, box, text, color):
    if box is None:
        return

    draw = ImageDraw.Draw(image)
    font = load_font()
    x0, y0, x1, y1 = [float(value) for value in box]
    draw.rectangle((x0, y0, x1, y1), outline=color, width=5)

    text_box = draw.textbbox((x0, y0), text, font=font)
    pad = 6
    draw.rectangle(
        (
            text_box[0],
            text_box[1],
            text_box[2] + pad * 2,
            text_box[3] + pad * 2,
        ),
        fill=color,
    )
    draw.text((x0 + pad, y0 + pad), text, fill=(255, 255, 255), font=font)


def overlay(image, mask, color):
    base = np.asarray(image, dtype=np.float32)
    color = np.array(color, dtype=np.float32)
    base[mask] = base[mask] * 0.55 + color * 0.45
    image.paste(Image.fromarray(base.astype(np.uint8), mode="RGB"))


def load_font(size=28):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def print_result(device, refs, out):
    print(f"device: {device}")
    print(f"reference: {REF}")
    print(f"target: {TARGET}")
    print(f"output: {OUT}")

    for ref in refs:
        for index, mask in enumerate(ref["masks"]):
            print(
                f"ref {ref['name']} {index}: "
                f"score={float(ref['scores'][index]):.4f} "
                f"pixels={int(mask.sum())} "
                f"box={find_box(mask)}"
            )

    for name, item in out.items():
        print(f"target {name}: masks={len(item['masks'])}")
        for index in range(len(item["masks"])):
            print(
                f"  {index}: "
                f"score={float(item['scores'][index]):.4f} "
                f"sim={float(item['similarities'][index]):.4f} "
                f"pixels={int(item['masks'][index].sum())} "
                f"box={tuple(float(value) for value in item['boxes'][index])} "
                f"refined_score={float(item['refined_scores'][index]):.4f} "
                f"refined_pixels={int(item['refined_masks'][index].sum())}"
            )


if __name__ == "__main__":
    main()
