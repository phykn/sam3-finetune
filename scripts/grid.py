import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.model import Sam3ImageModel  # noqa: E402
from src.predict.grid import GridPredictor  # noqa: E402
from src.predict.grid_ops.candidates import expand_mask  # noqa: E402
from src.predict.grid_ops.points import filter_points, make_points  # noqa: E402
from src.predict.grid_ops.tiles import make_crops  # noqa: E402
from src.predict.single import SinglePredictor  # noqa: E402

WEIGHT = ROOT / "weight" / "sam3.1_multiplex.pt"
IMAGE = ROOT / "asset" / "frog_tgt.jpg"
OUT = ROOT / "outputs" / "grid"
TILES = (1, 2)
POINTS = (10, 10)
COLORS = (
    (40, 120, 255),
    (255, 144, 40),
    (30, 190, 120),
    (220, 70, 220),
    (255, 210, 60),
)
POINT_RADIUS = 8


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(IMAGE).convert("RGB")

    model = Sam3ImageModel(path=WEIGHT)
    single = SinglePredictor(model, {"device": device})
    predictor = GridPredictor(single, tiles=TILES, points_per_side=POINTS)
    masks = predictor.predict(image)

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "frog_grid.png"
    make_sheet(image, predictor.before, masks).save(output)

    print(f"device: {device}")
    print(f"image: {IMAGE}")
    print(f"output: {output}")
    print(f"masks: {len(masks)}")
    print(f"tiles: {list(TILES)}")
    print(f"before_refine: {len(predictor.before)}")


def make_sheet(image, before, after):
    panels = [
        draw_points(image.copy()),
        draw_masks(image, before),
        draw_masks(image, after),
        draw_overlay(image, after),
    ]
    sheet = Image.new("RGB", (image.width * 2, image.height * 2), "white")
    for index, panel in enumerate(panels):
        x = image.width * (index % 2)
        y = image.height * (index // 2)
        sheet.paste(panel, (x, y))
    return sheet


def draw_points(image):
    draw = ImageDraw.Draw(image)
    for tile, side in zip(TILES, POINTS):
        color = COLORS[tile % len(COLORS)]
        for crop_index, crop in enumerate(make_crops(image.size, tile, 0.25)):
            x0, y0, _x1, _y1 = crop
            points = filter_points(
                make_points((crop[2] - crop[0], crop[3] - crop[1]), side),
                crop,
                tile,
                crop_index,
                image.size,
            )
            for x, y in points:
                draw.ellipse(
                    (
                        x0 + x - POINT_RADIUS,
                        y0 + y - POINT_RADIUS,
                        x0 + x + POINT_RADIUS,
                        y0 + y + POINT_RADIUS,
                    ),
                    fill=color,
                )
    return image


def draw_masks(image, masks):
    panel = np.full((image.height, image.width, 3), 255, dtype=np.uint8)
    for index, item in enumerate(masks):
        panel[expand_mask(item, image.size)] = COLORS[index % len(COLORS)]
    return Image.fromarray(panel, mode="RGB")


def draw_overlay(image, masks):
    base = np.asarray(image, dtype=np.float32)
    for index, item in enumerate(masks):
        mask = expand_mask(item, image.size)
        color = np.array(COLORS[index % len(COLORS)], dtype=np.float32)
        base[mask] = base[mask] * 0.55 + color * 0.45

    image = Image.fromarray(base.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    font = load_font()
    for index, item in enumerate(masks):
        color = COLORS[index % len(COLORS)]
        draw_item(draw, item, color, font)
    return image


def draw_item(draw, item, color, font):
    x0, y0, x1, y1 = item["bbox"]
    draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
    px, py = item["point"]
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)

    text = f"{item['score']:.2f}"
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
