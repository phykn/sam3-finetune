import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.model import Sam3ImageModel  # noqa: E402
from src.predict.single import SinglePredictor  # noqa: E402

WEIGHT = ROOT / "weight" / "sam3.1_multiplex.pt"
IMAGE = ROOT / "asset" / "frog_tgt.jpg"
OUT = ROOT / "outputs" / "single"
POINT = np.array([[610.0, 575.0]], dtype=np.float32)
LABEL = np.array([1], dtype=np.int32)
BLUE = (40, 120, 255)


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(IMAGE).convert("RGB")

    model = Sam3ImageModel(path=WEIGHT)
    predictor = SinglePredictor(model, {"device": device})
    out = predictor.predict(
        image,
        point_coords=POINT,
        point_labels=LABEL,
        multimask=False,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    mask = out["masks"][0]
    score = float(out["scores"][0])
    output = OUT / "frog_single.png"
    make_sheet(image, mask, score).save(output)

    print(f"device: {device}")
    print(f"image: {IMAGE}")
    print(f"point: {POINT[0].tolist()}")
    print(f"output: {output}")
    print(f"score: {score:.4f}")
    print(f"mask pixels: {int(mask.sum())}")


def make_sheet(image, mask, score):
    marked = draw_point(image.copy())
    binary = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")
    overlay = make_overlay(image, mask, score)

    sheet = Image.new("RGB", (image.width * 3, image.height), "white")
    sheet.paste(marked, (0, 0))
    sheet.paste(binary, (image.width, 0))
    sheet.paste(overlay, (image.width * 2, 0))
    return sheet


def draw_point(image):
    draw = ImageDraw.Draw(image)
    x, y = POINT[0]
    radius = 28
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=BLUE,
        outline=(255, 255, 255),
        width=4,
    )
    return image


def make_overlay(image, mask, score):
    base = np.asarray(image, dtype=np.float32)
    color = np.array(BLUE, dtype=np.float32)
    base[mask] = base[mask] * 0.55 + color * 0.45
    image = Image.fromarray(base.astype(np.uint8), mode="RGB")
    draw_bbox(image, mask, score)
    return image


def draw_bbox(image, mask, score):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return

    draw = ImageDraw.Draw(image)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    draw.rectangle(box, outline=BLUE, width=5)

    text = f"{score:.3f}"
    font = load_font()
    text_box = draw.textbbox((box[0], box[1]), text, font=font)
    pad = 6
    label_box = (
        text_box[0],
        text_box[1],
        text_box[2] + pad * 2,
        text_box[3] + pad * 2,
    )
    draw.rectangle(label_box, fill=BLUE)
    draw.text((box[0] + pad, box[1] + pad), text, fill=(255, 255, 255), font=font)


def load_font():
    try:
        return ImageFont.truetype("arial.ttf", 28)
    except OSError:
        return ImageFont.load_default()


if __name__ == "__main__":
    main()
