from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .grid_ops.boxes import filter_candidates, is_edge_cut
from .grid_ops.candidates import (
    expand_mask,
    format_logits,
    format_masks,
    make_candidate,
)
from .grid_ops.points import filter_points, make_points
from .grid_ops.tiles import make_crops
from .single import SinglePredictor


def _batches(points: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(points), batch_size):
        yield points[start : start + batch_size]


def _match(value: int | tuple[int, ...], count: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * count
    value = tuple(value)
    if len(value) == 1:
        return value * count
    if len(value) != count:
        raise ValueError("points_per_side must have length 1 or match tiles")
    return value


def _local_points(items: list[dict[str, object]]) -> np.ndarray:
    return np.array(
        [
            [item["point"][0] - item["crop"][0], item["point"][1] - item["crop"][1]]
            for item in items
        ],
        dtype=np.float32,
    )[:, None, :]


class GridPredictor:
    def __init__(
        self,
        single: SinglePredictor,
        tiles: tuple[int, ...] = (1, 2),
        points_per_side: int | tuple[int, ...] = (10, 10),
        overlap: float = 0.25,
        batch_size: int = 64,
        min_area: int = 64,
        nms: float = 0.7,
        min_stability: float = 0.75,
    ) -> None:
        self.single = single
        self.tiles = tuple(tiles)
        self.points_per_side = _match(points_per_side, len(self.tiles))
        self.overlap = float(overlap)
        self.batch_size = int(batch_size)
        self.min_area = int(min_area)
        self.nms = float(nms)
        self.min_stability = float(min_stability)
        self.before: list[dict[str, object]] = []
        self.after: list[dict[str, object]] = []

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        device: str | torch.device = "cuda",
        tiles: tuple[int, ...] = (1, 2),
        points_per_side: int | tuple[int, ...] = (10, 10),
        overlap: float = 0.25,
        batch_size: int = 64,
        min_area: int = 64,
        nms: float = 0.7,
        min_stability: float = 0.75,
    ) -> "GridPredictor":
        single = SinglePredictor.from_path(path, {"device": device})
        return cls(
            single,
            tiles=tiles,
            points_per_side=points_per_side,
            overlap=overlap,
            batch_size=batch_size,
            min_area=min_area,
            nms=nms,
            min_stability=min_stability,
        )

    def _keep(self, item: dict[str, object]) -> bool:
        return (
            item["area"] >= self.min_area
            and item["stability_score"] >= self.min_stability
            and not is_edge_cut(item)
        )

    @staticmethod
    def expand_mask(item: dict[str, object], image_size: tuple[int, int]) -> np.ndarray:
        return expand_mask(item, image_size)

    @staticmethod
    def _points(
        crop_size: tuple[int, int],
        crop: tuple[int, int, int, int],
        tile: int,
        crop_index: int,
        image_size: tuple[int, int],
        side: int,
    ) -> np.ndarray:
        return filter_points(
            make_points(crop_size, side),
            crop,
            tile,
            crop_index,
            image_size,
        )

    def iter_points(
        self,
        image_size: tuple[int, int],
    ) -> Iterator[tuple[int, int, tuple[int, int, int, int], np.ndarray]]:
        for tile, side in zip(self.tiles, self.points_per_side):
            crops = make_crops(image_size, tile, self.overlap)
            for crop_index, crop in enumerate(crops):
                crop_size = (crop[2] - crop[0], crop[3] - crop[1])
                yield (
                    tile,
                    crop_index,
                    crop,
                    self._points(
                        crop_size,
                        crop,
                        tile,
                        crop_index,
                        image_size,
                        side,
                    ),
                )

    def _predict_crop(
        self,
        image: Image.Image,
        crop: tuple[int, int, int, int],
        tile: int,
        crop_index: int,
        points: np.ndarray,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        crop_image = image.crop(crop)
        embed = self.single.encode(crop_image)
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
                if item is None or not self._keep(item):
                    continue
                item["refine_logit"] = logit.astype(np.float16, copy=True)
                items.append(item)
        return items, embed

    def _refine(
        self,
        items: list[dict[str, object]],
        embeds: dict[tuple[int, int], dict[str, object]],
    ) -> list[dict[str, object]]:
        groups = {}
        for index, item in enumerate(items):
            key = (item["tile"], item["crop_index"])
            groups.setdefault(key, []).append((index, item))

        refined = []
        for key, group in groups.items():
            embed = embeds[key]
            for start in range(0, len(group), self.batch_size):
                chunk = group[start : start + self.batch_size]
                chunk_items = [item for _index, item in chunk]
                points = _local_points(chunk_items)
                logits_in = np.stack([item["refine_logit"] for item in chunk_items])
                out = self.single.refine_low(
                    embed,
                    logits_in,
                    point_coords=points,
                    point_labels=np.ones((len(chunk), 1), dtype=np.int32),
                )
                masks = format_masks(out["masks"])
                logits = format_logits(out["logits"])
                scores = np.asarray(out["scores"]).reshape(-1)
                for (source_index, item), point, mask, logit, score in zip(
                    chunk,
                    points[:, 0, :],
                    masks,
                    logits,
                    scores,
                ):
                    new_item = make_candidate(
                        mask,
                        logit,
                        score,
                        point,
                        item["crop"],
                        item["tile"],
                        item["crop_index"],
                        item["image_size"],
                    )
                    if new_item is not None and self._keep(new_item):
                        refined.append((source_index, new_item))
        refined.sort(key=lambda x: x[0])
        return [item for _index, item in refined]

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> list[dict[str, object]]:
        image = image.convert("RGB")
        items = []
        embeds = {}
        for tile, crop_index, crop, points in self.iter_points(image.size):
            key = (tile, crop_index)
            crop_items, embed = self._predict_crop(
                image,
                crop,
                tile,
                crop_index,
                points,
            )
            embeds[key] = embed
            items.extend(crop_items)

        self.before = filter_candidates(items, self.nms)
        refined = self._refine(self.before, embeds)
        self.after = filter_candidates(refined, self.nms)
        return self.after
