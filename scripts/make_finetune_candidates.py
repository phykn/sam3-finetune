import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sample import Image as DataImage  # noqa: E402
from src.data.sample import Object, Sample, load, save  # noqa: E402

ASSET = ROOT / "asset"
SOURCE = ROOT / "finetune_dataset"
OUT = SOURCE
NAMES = {0: "background", 1: "frog", 2: "leaf"}
COLORS = {1: (255, 0, 255), 2: (40, 120, 255)}


def assign_class(
    item,
    regions,
    image_shape,
    min_overlap=0.5,
    class_masks=None,
):
    mask = np.zeros(image_shape, dtype=bool)
    x0, y0, x1, y1 = item["box"]
    mask[y0:y1, x0:x1] = np.asarray(item["roi"], dtype=bool)
    area = int(mask.sum())
    if area == 0:
        return None

    px, py = item["points"][0][:2]
    scores = []
    for class_id, boxes in regions.items():
        if class_masks is not None:
            reference = np.asarray(class_masks[class_id], dtype=bool)
            x = min(reference.shape[1] - 1, max(0, int(px)))
            y = min(reference.shape[0] - 1, max(0, int(py)))
            if not reference[y, x]:
                continue
            overlap = float(np.logical_and(mask, reference).sum()) / area
            if overlap >= min_overlap:
                scores.append((class_id, overlap))
            continue
        overlap = 0.0
        for box in boxes:
            bx0, by0, bx1, by1 = box
            if not (bx0 <= px < bx1 and by0 <= py < by1):
                continue
            overlap = max(overlap, float(mask[by0:by1, bx0:bx1].sum()) / area)
        if overlap >= min_overlap:
            scores.append((class_id, overlap))

    if not scores:
        return None
    scores.sort(key=lambda value: value[1], reverse=True)
    if len(scores) > 1 and scores[0][1] == scores[1][1]:
        return None
    return scores[0]


def select_frog(objects, regions, min_ratio=0.25):
    full = []
    for obj in objects:
        area = int(obj.roi.sum())
        if any(
            area / ((x1 - x0) * (y1 - y0)) >= min_ratio for x0, y0, x1, y1 in regions
        ):
            full.append(obj)
    return [] if not full else [max(full, key=lambda obj: int(obj.roi.sum()))]


def make_reference_masks(single, image, classes):
    boxes = []
    class_ids = []
    for class_id, regions in classes.items():
        boxes.extend(regions)
        class_ids.extend([class_id] * len(regions))
    results = single.predict(
        image,
        box=np.asarray(boxes, dtype=np.float32),
        multimask=False,
    )
    masks = {
        class_id: np.zeros((image.height, image.width), dtype=bool)
        for class_id in classes
    }
    for class_id, result in zip(class_ids, results, strict=True):
        x0, y0, x1, y1 = result["box"]
        masks[class_id][y0:y1, x0:x1] |= np.asarray(result["roi"], dtype=bool)
    return masks


def make_sample(predictor, split, filename, classes):
    image_path = ASSET / filename
    background_path = SOURCE / split / "0_background" / f"{Path(filename).stem}.json"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not background_path.is_file():
        raise FileNotFoundError(background_path)

    image = Image.open(image_path).convert("RGB")
    class_masks = make_reference_masks(predictor.single, image, classes)
    grouped = {1: [], 2: []}
    for item in predictor.predict(image):
        assigned = assign_class(
            item,
            classes,
            (image.height, image.width),
            class_masks=class_masks,
        )
        if assigned is None:
            continue
        class_id, overlap = assigned
        grouped[class_id].append(
            _object(
                item,
                class_id,
                overlap,
                len(grouped[class_id]) + 1,
                classes[class_id],
            )
        )

    grouped[1] = select_frog(grouped[1], classes[1])
    if not grouped[1]:
        frog_path = SOURCE / split / "1_frog" / f"{Path(filename).stem}.json"
        if not frog_path.is_file():
            raise FileNotFoundError(frog_path)
        grouped[1] = _fallback_objects(load(frog_path).objects)

    samples = {0: load(background_path)}
    data_image = DataImage(np.asarray(image, dtype=np.uint8), id=filename)
    for class_id, objects in grouped.items():
        samples[class_id] = Sample(image=data_image, objects=objects)

    for class_id, sample in samples.items():
        folder = OUT / split / f"{class_id}_{NAMES[class_id]}"
        folder.mkdir(parents=True, exist_ok=True)
        save(sample, folder / f"{Path(filename).stem}.json")
    preview = OUT / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    make_preview(image, grouped).save(
        preview / f"{split}_{Path(filename).stem}.jpg",
        quality=92,
    )
    return {class_id: len(sample.objects) for class_id, sample in samples.items()}


def _object(item, class_id, overlap, object_id, regions):
    point = [float(value) for value in item["points"][0][:2]]
    metrics = dict(item["metrics"])
    metrics["region_overlap"] = float(overlap)
    return Object(
        object_id=object_id,
        class_id=class_id,
        box=item["box"],
        roi=item["roi"],
        points=item["points"],
        metrics=metrics,
        meta={
            "source_point": point,
            "class_regions": [list(box) for box in regions],
        },
    )


def _fallback_objects(objects):
    out = []
    for object_id, obj in enumerate(objects, start=1):
        meta = dict(obj.meta)
        meta["candidate_source"] = "box_fallback"
        out.append(
            Object(
                object_id=object_id,
                class_id=1,
                box=obj.box,
                roi=obj.roi,
                points=obj.points,
                metrics=dict(obj.metrics),
                meta=meta,
            )
        )
    return out


def make_preview(image, grouped):
    base = np.asarray(image, dtype=np.float32).copy()
    for class_id, objects in grouped.items():
        color = np.asarray(COLORS[class_id], dtype=np.float32)
        for obj in objects:
            mask = obj.mask(base.shape).astype(bool)
            base[mask] = base[mask] * 0.55 + color * 0.45

    preview = Image.fromarray(base.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(preview)
    for class_id, objects in grouped.items():
        color = COLORS[class_id]
        for obj in objects:
            x0, y0, x1, y1 = obj.box
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=3)
            metrics = obj.metrics
            if obj.meta.get("candidate_source") == "box_fallback":
                text = f"{class_id}:{obj.object_id} fallback s={metrics['score']:.2f}"
            else:
                text = (
                    f"{class_id}:{obj.object_id} "
                    f"s={metrics['score']:.2f} "
                    f"t={metrics['stability']:.2f} "
                    f"o={metrics['region_overlap']:.2f}"
                )
            draw.text((x0 + 3, y0 + 3), text, fill=color)
    return preview


def load_regions(split, stem):
    regions = {}
    for class_id in (1, 2):
        path = SOURCE / split / f"{class_id}_{NAMES[class_id]}" / f"{stem}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        sample = load(path)
        if sample.objects and "class_regions" in sample.objects[0].meta:
            boxes = [tuple(box) for box in sample.objects[0].meta["class_regions"]]
        else:
            boxes = [tuple(obj.meta["source_box"]) for obj in sample.objects]
        if not boxes:
            raise ValueError(f"class region is empty: {path}")
        regions[class_id] = boxes
    return regions


def main():
    import torch

    from src.predict.grid import GridPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = GridPredictor.from_path(
        ROOT / "weight" / "sam3.1_multiplex.pt",
        device=device,
        tiles=(1, 2),
        points_per_side=(10, 10),
        overlap=0.25,
        batch_size=4,
        min_area=64,
        nms_thr=0.7,
        stability_thr=0.75,
    )
    for split in ("train", "valid"):
        folder = SOURCE / split / "0_background"
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.glob("*.json")):
            source = load(path)
            filename = source.image.id
            if not isinstance(filename, str):
                raise ValueError(f"image id must be a filename: {path}")
            counts = make_sample(
                predictor,
                split,
                filename,
                load_regions(split, path.stem),
            )
            print(f"{split}/{filename}: {counts}")
    print(f"device: {device}")
    print(f"dataset: {OUT}")


if __name__ == "__main__":
    main()
