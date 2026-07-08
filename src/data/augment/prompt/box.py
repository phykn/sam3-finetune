import numpy as np


def make(target, image_shape, jitter=0.1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    base = tight(target)
    if jitter <= 0:
        return base

    height, width = tuple(image_shape)[:2]
    x0, y0, x1, y1 = base
    dx = (x1 - x0) * float(jitter)
    dy = (y1 - y0) * float(jitter)
    out = np.array(
        [
            x0 + rng.uniform(-dx, dx),
            y0 + rng.uniform(-dy, dy),
            x1 + rng.uniform(-dx, dx),
            y1 + rng.uniform(-dy, dy),
        ],
        dtype=np.float32,
    )
    out[[0, 2]] = np.clip(out[[0, 2]], 0, width)
    out[[1, 3]] = np.clip(out[[1, 3]], 0, height)

    if out[2] <= out[0] or out[3] <= out[1]:
        return base
    return out


def tight(target):
    ys, xs = np.where(target > 0)
    return np.array(
        [
            float(xs.min()),
            float(ys.min()),
            float(xs.max() + 1),
            float(ys.max() + 1),
        ],
        dtype=np.float32,
    )
