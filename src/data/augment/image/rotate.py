import numpy as np


def random_rotate(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)
    factor = int(np.random.choice((0, 1, 2, 3)))
    if factor == 0:
        return image, mask

    image = np.rot90(image, factor).copy()
    mask = np.rot90(mask, factor).copy()
    return image, mask
