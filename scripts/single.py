import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict.single import SinglePredictor  # noqa: E402
from src.data import pack  # noqa: E402
from src.data.sample import Image as DataImage  # noqa: E402
from src.data.sample import Object, Sample, load, save  # noqa: E402

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

    predictor = SinglePredictor.from_path(WEIGHT, device=device)
    objects = predictor.predict(
        image,
        point_coords=POINT,
        point_labels=LABEL,
        multimask=False,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    item = objects[0]
    mask = pack.full(image.size[::-1], item["box"], item["roi"])
    score = float(item["metrics"]["score"])
    output = OUT / "frog_single.png"
    result = OUT / "frog_single.json"
    save_result(make_result(image, item), result)
    make_sheet(load_result(result)).save(output)

    print(f"device: {device}")
    print(f"image: {IMAGE}")
    print(f"point: {POINT[0].tolist()}")
    print(f"json: {result}")
    print(f"output: {output}")
    print(f"score: {score:.4f}")
    print(f"mask pixels: {int(mask.sum())}")


def make_result(image, item):
    point = [float(POINT[0][0]), float(POINT[0][1]), int(LABEL[0])]
    return Sample(
        image=DataImage(array=np.asarray(image, dtype=np.uint8)),
        objects=[
            Object(
                object_id=1,
                class_id=None,
                box=item["box"],
                roi=item["roi"],
                points=[point],
                metrics=dict(item["metrics"]),
                meta={
                    "prompt_index": item["prompt_index"],
                    "candidate_index": item["candidate_index"],
                },
            )
        ],
    )


def save_result(result, path):
    save(result, path)


def load_result(path):
    return load(path)


def make_sheet(result):
    image = Image.fromarray(result.image.array, mode="RGB")
    obj = result.objects[0]
    mask = obj.mask(result.image.shape).astype(bool)
    score = float(obj.metrics["score"])
    marked = draw_point(image.copy(), obj.points)
    binary = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")
    overlay = make_overlay(image, mask, score)

    sheet = Image.new("RGB", (image.width * 3, image.height), "white")
    sheet.paste(marked, (0, 0))
    sheet.paste(binary, (image.width, 0))
    sheet.paste(overlay, (image.width * 2, 0))
    return sheet


def draw_point(image, points):
    if not points:
        return image

    draw = ImageDraw.Draw(image)
    x, y = points[0][:2]
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
