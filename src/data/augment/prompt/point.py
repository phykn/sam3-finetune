import numpy as np


def sample_point_prompt(
    target: np.ndarray,
    union: np.ndarray,
    bg_prob: float = 0.2,
) -> dict[str, object]:
    if np.random.random() < bg_prob:
        ys, xs = np.where(union == 0)
        if len(xs) > 0:
            index = int(np.random.randint(len(xs)))
            return {
                "points": _make_point_array(xs[index], ys[index]),
                "point_labels": np.array([1], dtype=np.int64),
                "target": np.zeros_like(target, dtype=np.uint8),
                "has_mask": False,
                "is_auto_bg": True,
            }

    ys, xs = np.where(target > 0)
    index = int(np.random.randint(len(xs)))
    return {
        "points": _make_point_array(xs[index], ys[index]),
        "point_labels": np.array([1], dtype=np.int64),
        "target": target.astype(np.uint8, copy=False),
        "has_mask": True,
        "is_auto_bg": False,
    }


def _make_point_array(x: int, y: int) -> np.ndarray:
    return np.array([[float(x), float(y)]], dtype=np.float32)
