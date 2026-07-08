import numpy as np


def make(target, union, bg_prob=0.2, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if rng.random() < bg_prob:
        ys, xs = np.where(union == 0)
        if len(xs) > 0:
            index = int(rng.integers(len(xs)))
            return {
                "points": _point(xs[index], ys[index]),
                "point_labels": np.array([1], dtype=np.int64),
                "target": np.zeros_like(target, dtype=np.uint8),
                "has_object": False,
            }

    ys, xs = np.where(target > 0)
    index = int(rng.integers(len(xs)))
    return {
        "points": _point(xs[index], ys[index]),
        "point_labels": np.array([1], dtype=np.int64),
        "target": target.astype(np.uint8, copy=False),
        "has_object": True,
    }


def _point(x, y):
    return np.array([[float(x), float(y)]], dtype=np.float32)
