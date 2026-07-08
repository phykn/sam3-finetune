import numpy as np


def random_flip(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)
    mode = str(np.random.choice(("none", "horizontal")))
    if mode == "none":
        return image, mask

    return np.fliplr(image).copy(), np.fliplr(mask).copy()
