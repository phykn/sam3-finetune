from math import ceil


def make_crops(
    size: tuple[int, int],
    tile: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    width, height = size
    if tile == 1:
        return [(0, 0, width, height)]

    overlap_w = int(round((width / tile) * overlap))
    overlap_h = int(round((height / tile) * overlap))
    crop_w = int(ceil((width + overlap_w * (tile - 1)) / tile))
    crop_h = int(ceil((height + overlap_h * (tile - 1)) / tile))
    step_w = crop_w - overlap_w
    step_h = crop_h - overlap_h

    crops = []
    for y in range(tile):
        y0 = min(y * step_h, height - crop_h)
        for x in range(tile):
            x0 = min(x * step_w, width - crop_w)
            crops.append((int(x0), int(y0), int(x0 + crop_w), int(y0 + crop_h)))
    return crops
