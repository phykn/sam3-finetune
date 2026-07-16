import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from . import png
from .core import Image, Object, Sample

SCHEMA = "sam3.sample.v2"


def to_json(sample: Sample) -> dict[str, Any]:
    _validate_sample(sample)
    return {
        "schema_version": SCHEMA,
        "image": _img_to_json(sample.image),
        "objects": [_obj_to_json(obj) for obj in sample.objects],
    }


def from_json(data: dict[str, Any]) -> Sample:
    if data.get("schema_version") != SCHEMA:
        raise ValueError(f"schema_version must be {SCHEMA}")
    img = _img_from_json(data["image"])
    objs = [_obj_from_json(obj) for obj in data["objects"]]
    sample = Sample(image=img, objects=objs)
    _validate_sample(sample)
    return sample


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
    shape = tuple(data["shape"])
    if tuple(img.shape) != shape:
        raise ValueError("decoded image shape must match declared shape")
    return Image(array=img, shape=shape, id=data.get("id"))


def _obj_to_json(obj: Object) -> dict[str, Any]:
    return {
        "object_id": obj.object_id,
        "class_id": obj.class_id,
        "box": list(obj.box),
        "roi": _roi_to_json(obj.roi),
        "points": None if obj.points is None else [list(pt) for pt in obj.points],
        "metrics": obj.metrics,
        "meta": obj.meta,
    }


def _obj_from_json(data: dict[str, Any]) -> Object:
    return Object(
        object_id=data["object_id"],
        class_id=data.get("class_id"),
        box=tuple(data["box"]),
        roi=_roi_from_json(data["roi"]),
        points=data.get("points"),
        metrics=data.get("metrics", {}),
        meta=data.get("meta", {}),
    )


def _roi_to_json(roi) -> dict[str, Any]:
    shape = tuple(roi.shape)
    return {
        "shape": list(shape),
        "dtype": "uint8",
        "format": "png",
        "encoding": "base64",
        "data": png.pack(roi, mode="L"),
    }


def _roi_from_json(data: dict[str, Any]):
    shape = tuple(data["shape"])
    roi = png.unpack(data["data"], mode="L")
    if tuple(roi.shape) != shape:
        raise ValueError("decoded ROI shape must match declared shape")
    return roi


def _validate_sample(sample: Sample) -> None:
    shape = tuple(sample.image.shape)
    if tuple(sample.image.array.shape) != shape:
        raise ValueError("image array shape must match image shape")
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError("image must have shape HxWx3")

    height, width = shape[:2]
    for obj in sample.objects:
        _validate_object(obj, height, width)


def _validate_object(obj: Object, height: int, width: int) -> None:
    if len(obj.box) != 4 or not all(
        isinstance(value, Integral) and not isinstance(value, bool) for value in obj.box
    ):
        raise ValueError("object box must contain four integers")

    x0, y0, x1, y1 = obj.box
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError("object box must be inside the image with positive area")
    if obj.roi.shape != (y1 - y0, x1 - x0):
        raise ValueError("ROI shape must match object box")

    if obj.points is not None:
        for point in obj.points:
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ValueError("each point must contain x, y, and label")
            if not all(isinstance(value, Real) for value in point):
                raise ValueError("point values must be numeric")
            _check_finite(point, "point")

    if not isinstance(obj.metrics, dict):
        raise ValueError("object metrics must be a dictionary")
    if not isinstance(obj.meta, dict):
        raise ValueError("object meta must be a dictionary")
    _check_finite(obj.metrics, "metric")


def _check_finite(value: Any, name: str) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _check_finite(child, name)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_finite(child, name)
    elif isinstance(value, Real) and not math.isfinite(float(value)):
        raise ValueError(f"{name} values must be finite")
