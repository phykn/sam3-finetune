from typing import Any

import numpy as np


def pack(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array, dtype=np.uint8)
    flat = array.reshape(-1)
    if flat.size == 0:
        return {
            "shape": list(array.shape),
            "dtype": "uint8",
            "encoding": "rle",
            "start": 0,
            "counts": [],
        }

    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    stops = np.concatenate(([0], changes, [flat.size]))
    return {
        "shape": list(array.shape),
        "dtype": "uint8",
        "encoding": "rle",
        "start": int(flat[0]),
        "counts": np.diff(stops).astype(int).tolist(),
    }


def unpack(data: dict[str, Any]) -> np.ndarray:
    counts = data["counts"]
    flat = np.empty(sum(counts), dtype=np.uint8)
    offset = 0
    value = int(data["start"])
    for count in counts:
        flat[offset : offset + count] = value
        offset += count
        value = 1 - value
    return flat.reshape(tuple(data["shape"]))
