import json
from pathlib import Path
from typing import Any

from . import png, rle
from .core import Image, Object, Sample

SCHEMA = "sam3.sample.v1"


def to_json(sample: Sample) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "image": _img_to_json(sample.image),
        "objects": [_obj_to_json(obj) for obj in sample.objects],
    }


def from_json(data: dict[str, Any]) -> Sample:
    img = _img_from_json(data["image"])
    objs = [_obj_from_json(obj) for obj in data["objects"]]
    return Sample(image=img, objects=objs)


def save(sample: Sample, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_json(sample), file, indent=2)


def load(path: str | Path) -> Sample:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return from_json(json.load(file))


def _img_to_json(img: Image) -> dict[str, Any]:
    return {
        "id": img.id,
        "shape": list(img.shape),
        "dtype": "uint8",
        "color": "RGB",
        "format": "png",
        "encoding": "base64",
        "data": png.pack(img.array),
    }


def _img_from_json(data: dict[str, Any]) -> Image:
    img = png.unpack(data["data"])
    return Image(array=img, shape=tuple(data["shape"]), id=data.get("id"))


def _obj_to_json(obj: Object) -> dict[str, Any]:
    return {
        "object_id": obj.object_id,
        "class_id": obj.class_id,
        "box": list(obj.box),
        "roi": rle.pack(obj.roi),
        "points": None if obj.points is None else [list(pt) for pt in obj.points],
        "metrics": obj.metrics,
        "meta": obj.meta,
    }


def _obj_from_json(data: dict[str, Any]) -> Object:
    return Object(
        object_id=data["object_id"],
        class_id=data.get("class_id"),
        box=tuple(data["box"]),
        roi=rle.unpack(data["roi"]),
        points=data.get("points"),
        metrics=data.get("metrics", {}),
        meta=data.get("meta", {}),
    )
