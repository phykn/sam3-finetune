import base64
from io import BytesIO

import numpy as np
from PIL import Image


def pack(array: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(
        buf,
        format="PNG",
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def unpack(data: str) -> np.ndarray:
    raw = base64.b64decode(data.encode("ascii"))
    img = Image.open(BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)
