import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import pack  # noqa: E402
from src.data.sample import Image as DataImage  # noqa: E402
from src.data.sample import Object, Sample, load, save  # noqa: E402
from src.predict.ground import GroundPredictor  # noqa: E402
from src.predict.single import SinglePredictor  # noqa: E402

WEIGHT = ROOT / "weight" / "sam3.1_multiplex.pt"
VISUAL = ROOT / "weight" / "visual_token.pt"
REF = ROOT / "asset" / "frog_ref.jpg"
TARGET = ROOT / "asset" / "frog_tgt.jpg"
OUT = ROOT / "outputs" / "ground"

REFERENCE_BOXES = (
    (0, (350, 500, 580, 720)),
    (1, (0, 1010, 75, 1130)),
    (1, (80, 580, 160, 670)),
    (1, (230, 625, 320, 720)),
    (1, (770, 1090, 870, 1190)),
)
COLORS = (
    (255, 60, 60),
    (255, 180, 0),
    (40, 120, 255),
    (30, 190, 120),
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference_sample = make_reference(Image.open(REF).convert("RGB"))
    target = Image.open(TARGET).convert("RGB")
    boxes, class_ids = reference_arrays(reference_sample)

    predictor = GroundPredictor.from_path(
        WEIGHT,
        VISUAL,
        device=device,
        top_k=10,
        sim_thr=0.5,
    )
    reference = predictor.encode_reference(
        reference_sample.image.array,
        boxes,
        class_ids,
    )
    objects = predictor.predict(target, [reference])

    del reference, predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    refine(target, objects, device)

    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "frog_ground.json"
    image_path = OUT / "frog_ground.png"
    save_result(make_result(target, objects), result_path)
    result = load_result(result_path)
    make_sheet(result).save(image_path)

    print(f"device: {device}")
    print(f"reference: {REF}")
    print(f"target: {TARGET}")
    print(f"json: {result_path}")
    print(f"output: {image_path}")
    print(f"objects: {len(result.objects)}")


def make_reference(image):
    objects = []
    for object_id, (class_id, box) in enumerate(REFERENCE_BOXES, start=1):
        x0, y0, x1, y1 = box
        objects.append(
            Object(
                object_id=object_id,
                class_id=class_id,
                box=box,
                roi=np.ones((y1 - y0, x1 - x0), dtype=np.uint8),
            )
        )
    return Sample(
        image=DataImage(array=np.asarray(image, dtype=np.uint8)),
        objects=objects,
    )


def reference_arrays(sample):
    if not sample.objects:
        raise ValueError("reference sample has no objects")
    boxes = []
    class_ids = []
    for obj in sample.objects:
        if isinstance(obj.class_id, bool) or not isinstance(obj.class_id, int):
            raise ValueError("reference object class_id must be an integer")
        boxes.append(obj.box)
        class_ids.append(obj.class_id)
    return np.asarray(boxes, dtype=np.float32), np.asarray(class_ids, dtype=np.int64)


@torch.inference_mode()
def refine(image, objects, device):
    if not objects:
        return
    predictor = SinglePredictor.from_path(WEIGHT, device=device)
    embed = predictor.encode(image)
    logits = np.stack([item["logit"] for item in objects])
    refined = predictor.refine_embed(embed, logits)
    for item, result in zip(objects, refined, strict=True):
        mask = pack.full(image.size[::-1], result["box"], result["roi"])
        item["mask"] = mask
        item["box"] = result["box"]
        item["metrics"]["refined_score"] = float(result["metrics"]["score"])


def make_result(image, objects):
    packed = []
    for item in objects:
        box, roi = pack.box_roi(item["mask"])
        packed.append(
            Object(
                object_id=item["object_id"],
                class_id=item["class_id"],
                box=box,
                roi=roi,
                metrics=dict(item["metrics"]),
            )
        )
    return Sample(
        image=DataImage(array=np.asarray(image, dtype=np.uint8)),
        objects=packed,
    )


def save_result(result, path):
    save(result, path)


def load_result(path):
    return load(path)


def make_sheet(result):
    image = Image.fromarray(result.image.array, mode="RGB")
    overlay = draw_objects(result)
    sheet = Image.new("RGB", (image.width * 2, image.height), "white")
    sheet.paste(image, (0, 0))
    sheet.paste(overlay, (image.width, 0))
    return sheet


def draw_objects(result):
    image = Image.fromarray(result.image.array, mode="RGB")
    for obj in result.objects:
        overlay(image, obj.mask(result.image.shape).astype(bool), color(obj.class_id))
    for obj in result.objects:
        score = obj.metrics.get("refined_score", obj.metrics.get("score", 0.0))
        draw_box(
            image, obj.box, f"{obj.class_id} {float(score):.3f}", color(obj.class_id)
        )
    return image


def color(class_id):
    return COLORS[int(class_id) % len(COLORS)]


def draw_box(image, box, text, value):
    draw = ImageDraw.Draw(image)
    font = load_font()
    draw.rectangle(box, outline=value, width=5)
    text_box = draw.textbbox((box[0], box[1]), text, font=font)
    draw.rectangle(
        (text_box[0], text_box[1], text_box[2] + 12, text_box[3] + 12),
        fill=value,
    )
    draw.text(
        (box[0] + 6, box[1] + 6),
        text,
        fill=(255, 255, 255),
        font=font,
    )


def overlay(image, mask, value):
    base = np.asarray(image, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    base[mask] = base[mask] * 0.55 + value * 0.45
    image.paste(Image.fromarray(base.astype(np.uint8), mode="RGB"))


def load_font():
    try:
        return ImageFont.truetype("arial.ttf", 28)
    except OSError:
        return ImageFont.load_default()


if __name__ == "__main__":
    main()
