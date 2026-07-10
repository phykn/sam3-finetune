from math import ceil, floor

import numpy as np

from ...ops.box import nms_indices


def find_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def scale_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    crop: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    height, width = shape
    x0, y0, x1, y1 = box
    crop_x0, crop_y0, crop_x1, crop_y1 = crop
    scale_x = (crop_x1 - crop_x0) / width
    scale_y = (crop_y1 - crop_y0) / height
    return (
        crop_x0 + int(floor(x0 * scale_x)),
        crop_y0 + int(floor(y0 * scale_y)),
        crop_x0 + int(ceil(x1 * scale_x)),
        crop_y0 + int(ceil(y1 * scale_y)),
    )


def scale_area(
    area: int,
    shape: tuple[int, int],
    crop: tuple[int, int, int, int],
) -> int:
    height, width = shape
    crop_x0, crop_y0, crop_x1, crop_y1 = crop
    scale_x = (crop_x1 - crop_x0) / width
    scale_y = (crop_y1 - crop_y0) / height
    return int(round(int(area) * scale_x * scale_y))


def is_edge_cut(item: dict[str, object], atol: int = 4) -> bool:
    x0, y0, x1, y1 = item["low_box"]
    height, width = item["low_shape"]
    crop_x0, crop_y0, crop_x1, crop_y1 = item["crop"]
    image_width, image_height = item["image_size"]
    atol = int(atol)

    return (
        (crop_x0 > 0 and x0 <= atol)
        or (crop_y0 > 0 and y0 <= atol)
        or (crop_x1 < image_width and width - x1 <= atol)
        or (crop_y1 < image_height and height - y1 <= atol)
    )


def _ordered_nms(
    items: list[dict[str, object]],
    order: list[int],
    nms_thr: float,
) -> list[dict[str, object]]:
    if not items:
        return []
    boxes = np.array([item["bbox"] for item in items], dtype=np.float32)
    scores = np.empty(len(items), dtype=np.float32)
    scores[np.asarray(order)] = np.arange(len(items), 0, -1, dtype=np.float32)
    keep = nms_indices(boxes, scores, nms_thr)
    return [items[index] for index in keep]


def filter_crop(
    items: list[dict[str, object]],
    nms_thr: float,
) -> list[dict[str, object]]:
    order = sorted(
        range(len(items)),
        key=lambda index: (
            -items[index]["score"],
            -items[index]["stability_score"],
            index,
        ),
    )
    return _ordered_nms(items, order, nms_thr)


def filter_image(
    items: list[dict[str, object]],
    nms_thr: float,
) -> list[dict[str, object]]:
    def key(index):
        x0, y0, x1, y1 = items[index]["crop"]
        return (
            (x1 - x0) * (y1 - y0),
            -items[index]["score"],
            -items[index]["stability_score"],
            index,
        )

    order = sorted(range(len(items)), key=key)
    return _ordered_nms(items, order, nms_thr)
