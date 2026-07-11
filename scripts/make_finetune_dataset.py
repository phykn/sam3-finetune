import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sample import Image as DataImage  # noqa: E402
from src.data.sample import Object, Sample, save  # noqa: E402
from src.predict.single import SinglePredictor  # noqa: E402

WEIGHT = ROOT / "weight" / "sam3.1_multiplex.pt"
OUT = ROOT / "finetune_dataset"

SAMPLES = {
    "train": {
        "frog_ref.jpg": {
            0: [(350, 40, 720, 300), (550, 650, 835, 785)],
            1: [(275, 420, 610, 850)],
            2: [(0, 0, 350, 310), (790, 60, 963, 430), (0, 1100, 430, 1280)],
        },
        "frog_3.png": {
            0: [(480, 120, 710, 230), (1060, 570, 1212, 780)],
            1: [(220, 230, 720, 620)],
            2: [(0, 180, 310, 690), (600, 330, 1130, 660), (690, 0, 1212, 350)],
        },
        "frog_4.jpg": {
            0: [(0, 25, 150, 155), (410, 210, 550, 370)],
            1: [(145, 40, 395, 315)],
            2: [(0, 180, 415, 382), (390, 0, 550, 215), (0, 0, 205, 75)],
        },
        "frog_5.jpg": {
            0: [(250, 160, 420, 300), (1030, 700, 1200, 890)],
            1: [(400, 230, 920, 760)],
            2: [(690, 0, 980, 190), (850, 170, 1200, 440), (40, 420, 400, 770)],
        },
    },
    "valid": {
        "frog_tgt.jpg": {
            0: [(0, 570, 190, 760), (900, 500, 1180, 650)],
            1: [(360, 170, 850, 735)],
            2: [(210, 510, 1010, 790), (540, 130, 930, 330), (970, 300, 1280, 470)],
        }
    },
}

NAMES = {0: "background", 1: "frog", 2: "leaf"}
COLORS = {0: (255, 80, 80), 1: (50, 130, 255), 2: (50, 210, 100)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SinglePredictor.from_path(WEIGHT, device=device)
    for split, samples in SAMPLES.items():
        for filename, classes in samples.items():
            make_sample(predictor, split, filename, classes)
    print(f"device: {device}")
    print(f"dataset: {OUT}")


def make_sample(predictor, split, filename, classes):
    image = Image.open(ROOT / "asset" / filename).convert("RGB")
    preview = image.copy()
    for class_id, boxes in classes.items():
        objects = predict_objects(predictor, image, boxes, class_id)
        folder = OUT / split / f"{class_id}_{NAMES[class_id]}"
        folder.mkdir(parents=True, exist_ok=True)
        save(
            Sample(
                image=DataImage(array=np.asarray(image, dtype=np.uint8), id=filename),
                objects=objects,
            ),
            folder / f"{Path(filename).stem}.json",
        )
        draw_objects(preview, objects, class_id)

    preview_dir = OUT / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview.save(preview_dir / f"{split}_{Path(filename).stem}.jpg", quality=92)


def predict_objects(predictor, image, boxes, class_id):
    results = predictor.predict(
        image,
        box=np.asarray(boxes, dtype=np.float32),
        multimask=False,
    )
    objects = []
    for object_id, (source_box, result) in enumerate(
        zip(boxes, results, strict=True),
        start=1,
    ):
        if np.asarray(result["roi"]).sum() == 0:
            raise RuntimeError(f"empty mask: class={class_id}, box={source_box}")
        objects.append(
            Object(
                object_id=object_id,
                class_id=class_id,
                box=result["box"],
                roi=np.asarray(result["roi"], dtype=np.uint8),
                metrics={"score": float(result["metrics"]["score"])},
                meta={"source_box": list(source_box)},
            )
        )
    return objects


def draw_objects(image, objects, class_id):
    color = COLORS[class_id]
    base = np.asarray(image, dtype=np.float32).copy()
    for obj in objects:
        mask = obj.mask(base.shape).astype(bool)
        base[mask] = base[mask] * 0.65 + np.asarray(color) * 0.35
    image.paste(Image.fromarray(base.astype(np.uint8), mode="RGB"))

    draw = ImageDraw.Draw(image)
    for obj in objects:
        x0, y0, x1, y1 = obj.box
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=4)
        draw.text((x0 + 5, y0 + 5), f"{class_id}:{NAMES[class_id]}", fill=color)


if __name__ == "__main__":
    main()
