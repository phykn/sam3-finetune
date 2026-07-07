import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.model import Sam3ImageModel, Sam3VideoModel  # noqa: E402
from src.predict.single import SinglePredictor  # noqa: E402
from src.predict.video import VideoPredictor  # noqa: E402

WEIGHT = ROOT / "weight" / "sam3.1_multiplex.pt"
FRAMES = [
    ROOT / "asset" / "heli_1.jpg",
    ROOT / "asset" / "heli_2.jpg",
    ROOT / "asset" / "heli_3.jpg",
]
OUT = ROOT / "outputs" / "video"
POINT = np.array([[910.0, 345.0]], dtype=np.float32)
LABEL = np.array([1], dtype=np.int32)
POINT_COLOR = (40, 120, 255)
MASK_COLOR = (40, 120, 255)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = [Image.open(path).convert("RGB") for path in FRAMES]

    ref_mask, ref_score = segment_reference(frames[0], device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = Sam3VideoModel(path=WEIGHT)
    predictor = VideoPredictor(model, {"device": device})
    state = predictor.start(frames[0], ref_mask, obj_id=1)
    outputs = [predictor.predict(frame, state) for frame in frames[1:]]

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "heli_video.png"
    make_sheet(frames, ref_mask, ref_score, outputs).save(output)

    print(f"device: {device}")
    print(f"frames: {[str(path) for path in FRAMES]}")
    print(f"point: {POINT[0].tolist()}")
    print(f"reference score: {ref_score:.4f}")
    print(f"reference mask pixels: {int(ref_mask.sum())}")
    for index, out in enumerate(outputs, start=2):
        print(f"frame {index} score: {float(out['scores'][0]):.4f}")
        print(f"frame {index} mask pixels: {int(out['masks'][0].sum())}")
    print(f"output: {output}")


def segment_reference(image, device):
    model = Sam3ImageModel(path=WEIGHT)
    predictor = SinglePredictor(model, {"device": device})
    out = predictor.predict(
        image,
        point_coords=POINT,
        point_labels=LABEL,
        multimask=True,
    )
    index = int(np.argmax(out["scores"]))
    return out["masks"][index], float(out["scores"][index])


def make_sheet(frames, ref_mask, ref_score, outputs):
    panels = [
        draw_point(frames[0]),
        draw_mask(frames[0], ref_mask, ref_score),
        draw_mask(frames[1], outputs[0]["masks"][0], outputs[0]["scores"][0]),
        draw_mask(frames[2], outputs[1]["masks"][0], outputs[1]["scores"][0]),
    ]
    sheet = Image.new("RGB", (frames[0].width * 2, frames[0].height * 2), "white")
    for index, panel in enumerate(panels):
        x = frames[0].width * (index % 2)
        y = frames[0].height * (index // 2)
        sheet.paste(panel, (x, y))
    return sheet


def draw_point(image):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    x, y = POINT[0]
    radius = 18
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=POINT_COLOR,
        outline=(255, 255, 255),
        width=4,
    )
    return image


def draw_mask(image, mask, score):
    image = image.copy()
    overlay(image, mask, MASK_COLOR)
    draw_box(image, find_box(mask), f"{float(score):.3f}")
    return image


def find_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_box(image, box, text):
    if box is None:
        return

    draw = ImageDraw.Draw(image)
    font = load_font()
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1, y1), outline=MASK_COLOR, width=4)
    label = draw.textbbox((x0, y0), text, font=font)
    pad = 5
    draw.rectangle(
        (label[0], label[1], label[2] + pad * 2, label[3] + pad * 2),
        fill=MASK_COLOR,
    )
    draw.text((x0 + pad, y0 + pad), text, fill=(255, 255, 255), font=font)


def overlay(image, mask, color):
    base = np.asarray(image, dtype=np.float32)
    color = np.array(color, dtype=np.float32)
    base[mask] = base[mask] * 0.55 + color * 0.45
    image.paste(Image.fromarray(base.astype(np.uint8), mode="RGB"))


def load_font():
    try:
        return ImageFont.truetype("arial.ttf", 24)
    except OSError:
        return ImageFont.load_default()


if __name__ == "__main__":
    main()
