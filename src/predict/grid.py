import numpy as np
import torch
from PIL import Image

from .grid_ops.boxes import filter_candidates, is_edge_cut
from .grid_ops.candidates import format_logits, format_masks, make_candidate
from .grid_ops.points import filter_points, make_points
from .grid_ops.tiles import make_crops

MIN_STABILITY = 0.75


class GridPredictor:
    def __init__(
        self,
        single,
        tiles=(1, 2),
        points_per_side=(10, 10),
        overlap=0.25,
        batch_size=64,
        min_area=64,
        max_masks=40,
        nms=0.7,
    ) -> None:
        self.single = single
        self.tiles = tuple(tiles)
        self.points_per_side = _match(points_per_side, len(self.tiles))
        self.overlap = float(overlap)
        self.batch_size = int(batch_size)
        self.min_area = int(min_area)
        self.max_masks = int(max_masks)
        self.nms = float(nms)
        self.before = []
        self.after = []

    @torch.inference_mode()
    def predict(self, image: Image.Image):
        image = image.convert("RGB")
        items = []
        embeds = {}
        for tile, side in zip(self.tiles, self.points_per_side):
            for crop_index, crop in enumerate(
                make_crops(image.size, tile, self.overlap)
            ):
                key = (tile, crop_index)
                crop_items, embed = self._predict_crop(
                    image,
                    crop,
                    tile,
                    crop_index,
                    side,
                )
                embeds[key] = embed
                items.extend(crop_items)

        self.before = filter_candidates(items, self.nms, self.max_masks)
        refined = self._refine(self.before, embeds)
        self.after = filter_candidates(refined, self.nms, self.max_masks)
        return self.after

    def _predict_crop(self, image, crop, tile, crop_index, side):
        crop_image = image.crop(crop)
        embed = self.single.encode(crop_image)
        points = filter_points(
            make_points(crop_image.size, side),
            crop,
            tile,
            crop_index,
            image.size,
        )
        items = []

        for batch in _batches(points, self.batch_size):
            out = self.single.predict_embed_low(
                embed,
                point_coords=batch[:, None, :],
                point_labels=np.ones((len(batch), 1), dtype=np.int32),
                multimask=False,
            )
            masks = format_masks(out["masks"])
            logits = format_logits(out["logits"])
            scores = np.asarray(out["scores"]).reshape(-1)
            for point, mask, logit, score in zip(batch, masks, logits, scores):
                item = make_candidate(
                    mask,
                    logit,
                    score,
                    point,
                    crop,
                    tile,
                    crop_index,
                    image.size,
                )
                if item is not None:
                    item["refine_logit"] = logit.astype(np.float16, copy=True)
                if item is not None and self._keep(item):
                    items.append(item)
        return items, embed

    def _refine(self, items, embeds):
        refined = []
        for item in items:
            crop = item["crop"]
            point = np.array(
                [
                    [
                        [
                            item["point"][0] - crop[0],
                            item["point"][1] - crop[1],
                        ]
                    ]
                ],
                dtype=np.float32,
            )
            out = self.single.refine_low(
                embeds[(item["tile"], item["crop_index"])],
                item["refine_logit"],
                point_coords=point,
                point_labels=np.ones((1, 1), dtype=np.int32),
            )
            masks = format_masks(out["masks"])
            logits = format_logits(out["logits"])
            scores = np.asarray(out["scores"]).reshape(-1)
            new_item = make_candidate(
                masks[0],
                logits[0],
                scores[0],
                point[0, 0],
                crop,
                item["tile"],
                item["crop_index"],
                item["image_size"],
            )
            if new_item is not None and self._keep(new_item):
                refined.append(new_item)
        return refined

    def _keep(self, item):
        return (
            item["area"] >= self.min_area
            and item["stability_score"] >= MIN_STABILITY
            and not is_edge_cut(item)
        )


def _batches(points, batch_size):
    for start in range(0, len(points), batch_size):
        yield points[start : start + batch_size]


def _match(value, count):
    if isinstance(value, int):
        return (value,) * count
    value = tuple(value)
    if len(value) == 1:
        return value * count
    if len(value) != count:
        raise ValueError("points_per_side must have length 1 or match tiles")
    return value
