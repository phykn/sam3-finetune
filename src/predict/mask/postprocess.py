import numpy as np

from ...data import pack


def sort_area(
    items: list[dict[str, object]],
    order: str,
) -> list[dict[str, object]]:
    if order not in ("asc", "desc"):
        raise ValueError("order must be 'asc' or 'desc'")
    out = [_copy(item) for item in items]
    out.sort(key=_area, reverse=order == "desc")
    return _renumber(out)


def make_exclusive(
    items: list[dict[str, object]],
    min_ratio: float,
) -> list[dict[str, object]]:
    min_ratio = _ratio(min_ratio, "min_ratio")
    ranked = sorted(
        enumerate(_copy(item) for item in items),
        key=lambda value: (_area(value[1]), value[0]),
    )
    kept = []
    for index, item in ranked:
        original = _area(item)
        if original == 0:
            continue
        for _, smaller in kept:
            _subtract(item, smaller)
        remaining = _area(item)
        if remaining == 0 or remaining / original < min_ratio:
            continue
        kept.append((index, _compact(item)))

    out = [item for _, item in sorted(kept)]
    return _renumber(out)


def merge_overlap(
    items: list[dict[str, object]],
    min_overlap: float,
) -> list[dict[str, object]]:
    min_overlap = _ratio(min_overlap, "min_overlap")
    active = {index: _copy(item) for index, item in enumerate(items) if _area(item) > 0}

    while True:
        order = sorted(active, key=lambda index: (-_area(active[index]), index))
        merged = False
        for position, large_index in enumerate(order):
            large = active[large_index]
            for small_index in order[position + 1 :]:
                small = active[small_index]
                overlap = _intersection(large, small)
                if overlap == 0 or overlap / _area(small) < min_overlap:
                    continue
                active[large_index] = _union(large, small)
                del active[small_index]
                merged = True
                break
            if merged:
                break
        if not merged:
            break

    return _renumber([active[index] for index in sorted(active)])


def drop_edge(
    items: list[dict[str, object]],
    image_shape: tuple[int, ...] | list[int],
) -> list[dict[str, object]]:
    height, width = tuple(image_shape)[:2]
    out = []
    for item in items:
        x0, y0, x1, y1 = item["box"]
        if x0 == 0 or y0 == 0 or x1 == width or y1 == height:
            continue
        out.append(_copy(item))
    return _renumber(out)


def _copy(item: dict[str, object]) -> dict[str, object]:
    out = dict(item)
    out["box"] = tuple(int(value) for value in item["box"])
    out["roi"] = np.asarray(item["roi"], dtype=bool).copy()
    return out


def _area(item: dict[str, object]) -> int:
    return int(np.count_nonzero(item["roi"]))


def _renumber(items: list[dict[str, object]]) -> list[dict[str, object]]:
    for index, item in enumerate(items, start=1):
        item["object_id"] = index
    return items


def _ratio(value: float, name: str) -> float:
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return value


def _views(
    first: dict[str, object],
    second: dict[str, object],
) -> tuple[np.ndarray, np.ndarray] | None:
    ax0, ay0, ax1, ay1 = first["box"]
    bx0, by0, bx1, by1 = second["box"]
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return None
    return (
        first["roi"][y0 - ay0 : y1 - ay0, x0 - ax0 : x1 - ax0],
        second["roi"][y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0],
    )


def _subtract(item: dict[str, object], occupied: dict[str, object]) -> None:
    views = _views(item, occupied)
    if views is not None:
        target, mask = views
        target &= ~mask


def _intersection(first: dict[str, object], second: dict[str, object]) -> int:
    views = _views(first, second)
    if views is None:
        return 0
    return int(np.count_nonzero(views[0] & views[1]))


def _compact(item: dict[str, object]) -> dict[str, object]:
    x0, y0, _, _ = item["box"]
    local_box, roi = pack.box_roi(item["roi"])
    lx0, ly0, lx1, ly1 = local_box
    item["box"] = (x0 + lx0, y0 + ly0, x0 + lx1, y0 + ly1)
    item["roi"] = roi.astype(bool)
    return item


def _union(
    first: dict[str, object],
    second: dict[str, object],
) -> dict[str, object]:
    ax0, ay0, ax1, ay1 = first["box"]
    bx0, by0, bx1, by1 = second["box"]
    x0, y0 = min(ax0, bx0), min(ay0, by0)
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    roi = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    roi[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0] |= first["roi"]
    roi[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0] |= second["roi"]
    out = dict(first)
    out["box"] = (x0, y0, x1, y1)
    out["roi"] = roi
    return _compact(out)
