from collections.abc import Iterator
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .grid_ops.boxes import filter_image, is_edge_cut
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


def _class_rows(out: dict[str, object], count: int) -> list[np.ndarray | None]:
    values = out.get("class_scores")
    if values is None:
        return [None] * count
    values = np.asarray(values).reshape(count, -1)
    return [row.copy() for row in values]


def _match(value: int | tuple[int, ...], count: int) -> tuple[int, ...]:
    try:
        values = (value,) if isinstance(value, Integral) else tuple(value)
    except TypeError as error:
        raise ValueError("points_per_side must contain positive integers") from error
    if not values or any(
        isinstance(item, bool) or not isinstance(item, Integral) or item <= 0
        for item in values
    ):
        raise ValueError("points_per_side must contain positive integers")
    if len(values) == 1:
        return tuple(int(values[0]) for _ in range(count))
    if len(values) != count:
        raise ValueError("points_per_side must have length 1 or match tiles")
    return tuple(int(item) for item in values)


def _positive_int(value, name, zero=False):
    minimum = 0 if zero else 1
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


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
        nms_thr: float = 0.7,
        stability_thr: float = 0.75,
    ) -> None:
        try:
            tiles = tuple(tiles)
        except TypeError as error:
            raise ValueError("tiles must contain positive integers") from error
        if not tiles or any(
            isinstance(tile, bool) or not isinstance(tile, Integral) or tile <= 0
            for tile in tiles
        ):
            raise ValueError("tiles must contain positive integers")
        if len(set(tiles)) != len(tiles):
            raise ValueError("tiles must be unique")
        overlap = float(overlap)
        nms_thr = float(nms_thr)
        stability_thr = float(stability_thr)
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be between zero and one")
        if not 0 <= nms_thr <= 1:
            raise ValueError("nms_thr must be between zero and one")
        if not 0 <= stability_thr <= 1:
            raise ValueError("stability_thr must be between zero and one")
        self.single = single
        self.tiles = tuple(int(tile) for tile in tiles)
        self.points_per_side = _match(points_per_side, len(self.tiles))
        self.overlap = overlap
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.min_area = _positive_int(min_area, "min_area", zero=True)
        self.nms_thr = nms_thr
        self.stability_thr = stability_thr
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
        nms_thr: float = 0.7,
        stability_thr: float = 0.75,
    ) -> "GridPredictor":
        single = SinglePredictor.from_path(path, device=device)
        return cls(
            single,
            tiles=tiles,
            points_per_side=points_per_side,
            overlap=overlap,
            batch_size=batch_size,
            min_area=min_area,
            nms_thr=nms_thr,
            stability_thr=stability_thr,
        )

    def _keep(self, item: dict[str, object]) -> bool:
        return (
            item["area"] >= self.min_area
            and item["stability_score"] >= self.stability_thr
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
            out = self.single._predict_low(
                embed,
                point_coords=batch[:, None, :],
                point_labels=np.ones((len(batch), 1), dtype=np.int32),
                multimask=False,
            )
            masks = format_masks(out["masks"])
            logits = format_logits(out["logits"])
            scores = np.asarray(out["scores"]).reshape(-1)
            classes = _class_rows(out, len(scores))
            for point, mask, logit, score, class_scores in zip(
                batch,
                masks,
                logits,
                scores,
                classes,
            ):
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
                if class_scores is not None:
                    item["class_scores"] = class_scores
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
                out = self.single._predict_low(
                    embed,
                    point_coords=points,
                    point_labels=np.ones((len(chunk), 1), dtype=np.int32),
                    mask=logits_in,
                    multimask=False,
                )
                masks = format_masks(out["masks"])
                logits = format_logits(out["logits"])
                scores = np.asarray(out["scores"]).reshape(-1)
                classes = _class_rows(out, len(scores))
                for (
                    (source_index, item),
                    point,
                    mask,
                    logit,
                    score,
                    class_scores,
                ) in zip(
                    chunk,
                    points[:, 0, :],
                    masks,
                    logits,
                    scores,
                    classes,
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
                        if class_scores is not None:
                            new_item["class_scores"] = class_scores
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

        self.before = filter_image(items, self.nms_thr)
        refined = self._refine(self.before, embeds)
        self.after = filter_image(refined, self.nms_thr)
        return self.after
