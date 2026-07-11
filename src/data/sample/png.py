import base64
from io import BytesIO

import numpy as np
from PIL import Image


def pack(array: np.ndarray, mode: str = "RGB") -> str:
    buf = BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode=mode).save(
        buf,
        format="PNG",
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def unpack(data: str, mode: str = "RGB") -> np.ndarray:
    raw = base64.b64decode(data.encode("ascii"))
    img = Image.open(BytesIO(raw)).convert(mode)
    return np.asarray(img, dtype=np.uint8)
