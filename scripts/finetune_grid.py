import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEIGHT = ROOT / "weight" / "sam3.1_multiplex.pt"
IMAGE = ROOT / "asset" / "frog_tgt.jpg"
OUT = ROOT / "outputs" / "finetune_grid"
TILES = (1, 2)
POINTS = (10, 10)
BATCH_SIZE = 4
COND = 0
COLORS = (
    (40, 120, 255),
    (255, 144, 40),
    (30, 190, 120),
    (220, 70, 220),
    (255, 210, 60),
)


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(IMAGE).convert("RGB")

    predictor = make_predictor(device)
    masks = predictor.predict(image)

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "frog_grid.png"
    result = OUT / "frog_grid.json"
    save_result(make_result(image, predictor, masks), result)
    make_sheet(load_result(result)).save(output)

    print(f"device: {device}")
    print(f"image: {IMAGE}")
    print(f"cond: {COND}")
    print(f"json: {result}")
    print(f"output: {output}")
    print(f"masks: {len(masks)}")
    print(f"tiles: {list(TILES)}")
    print(f"before_refine: {len(predictor.before)}")


def make_predictor(device):
    from src.build import build_finetune_model
    from src.predict.grid import GridPredictor
    from src.predict.single import SinglePredictor

    model = build_finetune_model(
        {
            "path": WEIGHT,
            "device": device,
            "num_conditions": 1,
            "num_experts": 4,
            "num_classes": 1,
            "lora_rank": 8,
            "feature_rank": 16,
        }
    )
    single = SinglePredictor(model, device=device, cond=COND)
    return GridPredictor(
        single,
        tiles=TILES,
        points_per_side=POINTS,
        batch_size=BATCH_SIZE,
    )


def make_result(image, predictor, items):
    from src.data.sample import Image as DataImage
    from src.data.sample import Sample

    return Sample(
        image=DataImage(array=np.asarray(image, dtype=np.uint8)),
        objects=pack_items(image.size, predictor, items),
    )


def save_result(result, path):
    from src.data.sample import save

    save(result, path)


def load_result(path):
    from src.data.sample import load

    return load(path)


def pack_items(image_size, predictor, items):
    from src.data import pack
    from src.data.sample import Object

    out = []
    for index, item in enumerate(items, start=1):
        mask = predictor.expand_mask(item, image_size)
        box, roi = pack.box_roi(mask)
        point = [float(item["point"][0]), float(item["point"][1]), 1]
        metrics = {"score": float(item["score"])}
        if "class_scores" in item:
            metrics["class_scores"] = np.asarray(
                item["class_scores"],
                dtype=float,
            ).tolist()
        out.append(
            Object(
                object_id=index,
                class_id=None,
                box=box,
                roi=roi,
                points=[point],
                metrics=metrics,
            )
        )
    return out


def make_sheet(result):
    image = Image.fromarray(result.image.array, mode="RGB")
    panels = [
        image.copy(),
        draw_masks(image, result.objects),
        draw_overlay(image, result.objects),
    ]
    sheet = Image.new("RGB", (image.width * len(panels), image.height), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (image.width * index, 0))
    return sheet


def draw_masks(image, objects):
    panel = np.full((image.height, image.width, 3), 255, dtype=np.uint8)
    for index, obj in enumerate(objects):
        panel[obj.mask(image.size[::-1]).astype(bool)] = COLORS[index % len(COLORS)]
    return Image.fromarray(panel, mode="RGB")


def draw_overlay(image, objects):
    base = np.asarray(image, dtype=np.float32)
    for index, obj in enumerate(objects):
        mask = obj.mask(image.size[::-1]).astype(bool)
        color = np.array(COLORS[index % len(COLORS)], dtype=np.float32)
        base[mask] = base[mask] * 0.55 + color * 0.45

    image = Image.fromarray(base.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    font = load_font()
    for index, item in enumerate(objects):
        color = COLORS[index % len(COLORS)]
        draw_item(draw, item, color, font)
    return image


def draw_item(draw, obj, color, font):
    x0, y0, x1, y1 = obj.box
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=4)
    px, py = obj.points[0][:2]
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)

    text = f"{obj.metrics['score']:.2f}"
    text_box = draw.textbbox((x0, y0), text, font=font)
    label = (text_box[0], text_box[1], text_box[2] + 8, text_box[3] + 8)
    draw.rectangle(label, fill=color)
    draw.text((x0 + 4, y0 + 4), text, fill=(255, 255, 255), font=font)


def load_font():
    try:
        return ImageFont.truetype("arial.ttf", 24)
    except OSError:
        return ImageFont.load_default()


if __name__ == "__main__":
    main()
