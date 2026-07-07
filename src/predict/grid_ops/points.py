import numpy as np


def make_points(size: tuple[int, int], points_per_side: int) -> np.ndarray:
    width, height = size
    offset = 0.5 / points_per_side
    values = np.linspace(offset, 1.0 - offset, points_per_side, dtype=np.float32)
    x, y = np.meshgrid(values * width, values * height)
    return np.stack([x.reshape(-1), y.reshape(-1)], axis=1).astype(np.float32)


def filter_points(points, crop, tile, crop_index, full_size):
    if tile == 1:
        return points

    width, height = full_size
    col = crop_index % tile
    row = crop_index // tile
    x0 = width * col / tile
    x1 = width * (col + 1) / tile
    y0 = height * row / tile
    y1 = height * (row + 1) / tile

    global_x = points[:, 0] + crop[0]
    global_y = points[:, 1] + crop[1]
    keep = (global_x >= x0) & (global_y >= y0)
    if col < tile - 1:
        keep &= global_x < x1
    if row < tile - 1:
        keep &= global_y < y1
    return points[keep]
