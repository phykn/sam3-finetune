import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict.ground import GroundPredictor  # noqa: E402
from src.predict.single import SinglePredictor  # noqa: E402
from src.data import pack  # noqa: E402
from src.data.sample import Image as DataImage  # noqa: E402
from src.data.sample import Object, Sample, load, save  # noqa: E402

WEIGHT = ROOT / "weight" / "sam3.1_multiplex.pt"
VISUAL = ROOT / "weight" / "visual_token.pt"
REF = ROOT / "asset" / "frog_ref.jpg"
TARGET = ROOT / "asset" / "frog_tgt.jpg"
OUT = ROOT / "outputs" / "ground" / "frog_flower_ground.png"
JSON = OUT.with_suffix(".json")

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
    save_result(make_result(ref_image, refs, target, out), JSON)
    result = load_result(JSON)
    make_sheet(result).save(OUT)
    print_result(device, refs, out)


@torch.inference_mode()
def segment_refs(image, device):
    predictor = SinglePredictor.from_path(WEIGHT, {"device": device})
    embed = predictor.encode(image)
    refs = []
    for ref in CONCEPTS:
        points = ref["points"]
        point_labels = np.broadcast_to(
            POINT_LABEL,
            (len(points), len(POINT_LABEL)),
        ).copy()
        out = predictor.predict_embed(
            embed,
            point_coords=points[:, None],
            point_labels=point_labels,
            multimask=True,
        )
        masks = np.asarray(out["masks"], dtype=bool)
        scores = np.asarray(out["scores"], dtype=np.float32)
        if masks.ndim == 3:
            masks = masks[None]
        if scores.ndim == 1:
            scores = scores[None]
        indices = np.argmax(scores, axis=1)
        rows = np.arange(len(points))
        refs.append(
            {
                **ref,
                "masks": masks[rows, indices],
                "scores": scores[rows, indices],
            }
        )
    return refs


@torch.inference_mode()
def refine(image, out, device):
    predictor = SinglePredictor.from_path(WEIGHT, {"device": device})
    embed = predictor.encode(image)
    for item in out.values():
        refined = predictor.predict_embed(
            embed,
            mask=item["logits"],
            multimask=False,
        )
        masks = np.asarray(refined["masks"], dtype=bool)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        item["refined_masks"] = masks
        item["refined_scores"] = np.asarray(
            refined["scores"], dtype=np.float32
        ).reshape(-1)


def make_result(ref_image, refs, target, out):
    objects = []
    for name, item in out.items():
        masks = item["refined_masks"] if "refined_masks" in item else item["masks"]
        for index, mask in enumerate(masks):
            box, roi = pack.box_roi(mask)
            metrics = {"score": float(item["scores"][index])}
            if "similarities" in item:
                metrics["similarity"] = float(item["similarities"][index])
            if "refined_scores" in item:
                metrics["refined_score"] = float(item["refined_scores"][index])
            objects.append(
                Object(
                    object_id=len(objects) + 1,
                    class_id=None,
                    box=box,
                    roi=roi,
                    metrics=metrics,
                    meta={"name": name},
                )
            )
    return Sample(
        image=DataImage(array=np.asarray(target, dtype=np.uint8)),
        objects=objects,
    )


def save_result(result, path):
    save(result, path)


def load_result(path):
    return load(path)


def pack_ref(ref):
    return {
        "name": ref["name"],
        "points": np.asarray(ref["points"], dtype=np.float32).tolist(),
        "color": list(ref["color"]),
        "masks": [pack.mask(mask) for mask in ref["masks"]],
        "scores": np.asarray(ref["scores"], dtype=np.float32).tolist(),
    }


def read_ref(ref):
    return {
        "name": ref["name"],
        "points": np.asarray(ref["points"], dtype=np.float32),
        "color": tuple(ref["color"]),
        "masks": np.asarray([pack.read_mask(mask) for mask in ref["masks"]]),
        "scores": np.asarray(ref["scores"], dtype=np.float32),
    }


def pack_target(item):
    return {
        "scores": np.asarray(item["scores"], dtype=np.float32).tolist(),
        "similarities": np.asarray(
            item.get("similarities", item["scores"]),
            dtype=np.float32,
        ).tolist(),
        "boxes": np.asarray(item["boxes"], dtype=np.float32).tolist(),
        "masks": [pack.mask(mask) for mask in item["masks"]],
        "refined_masks": [pack.mask(mask) for mask in item["refined_masks"]],
        "refined_scores": np.asarray(item["refined_scores"], dtype=np.float32).tolist(),
    }


def read_target(item):
    return {
        "scores": np.asarray(item["scores"], dtype=np.float32),
        "similarities": np.asarray(item["similarities"], dtype=np.float32),
        "boxes": np.asarray(item["boxes"], dtype=np.float32),
        "masks": np.asarray([pack.read_mask(mask) for mask in item["masks"]]),
        "refined_masks": np.asarray(
            [pack.read_mask(mask) for mask in item["refined_masks"]]
        ),
        "refined_scores": np.asarray(item["refined_scores"], dtype=np.float32),
    }


def make_sheet(result):
    image = Image.fromarray(result.image.array, mode="RGB")
    panels = [image.copy(), draw_objects(result)]

    cell_w = image.width
    cell_h = image.height
    sheet = Image.new("RGB", (cell_w * 2, cell_h), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (cell_w * index, 0))
    return sheet


def draw_objects(result):
    image = Image.fromarray(result.image.array, mode="RGB")
    for index, obj in enumerate(result.objects):
        color = concept_color(obj.meta.get("name"), index)
        overlay(image, obj.mask(result.image.shape).astype(bool), color)
    for index, obj in enumerate(result.objects):
        color = concept_color(obj.meta.get("name"), index)
        name = obj.meta.get("name", f"object {obj.object_id}")
        score = obj.metrics.get("refined_score", obj.metrics.get("score", 0.0))
        draw_box(image, obj.box, f"{name} {float(score):.3f}", color)
    return image


def concept_color(name, index):
    for ref in CONCEPTS:
        if ref["name"] == name:
            return ref["color"]
    return (40, 120, 255) if index == 0 else (255, 144, 40)


def draw_points(image, refs):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = load_font(32)
    for ref in refs:
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


def draw_target(image, refs, out, refined):
    image = image.copy()
    for ref in refs:
        item = out[ref["name"]]
        key = "refined_masks" if refined else "masks"
        for mask in item[key]:
            overlay(image, mask, ref["color"])

    for ref in refs:
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
    print(f"json: {JSON}")
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
