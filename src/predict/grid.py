from collections.abc import Iterator
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..data import image as image_data
from .grid_ops.boxes import filter_crop, filter_image, is_edge_cut
from .grid_ops.candidates import (
    format_logits,
    format_masks,
    make_candidate,
    make_objects,
)
from .grid_ops.points import filter_points, make_points
from .grid_ops.tiles import make_crops
from .mask.component import largest
from .single import SinglePredictor


def _batches(points: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(points), batch_size):
        yield points[start : start + batch_size]


def _class_rows(out: dict[str, object], count: int) -> list[dict[str, np.ndarray]]:
    rows = [{} for _ in range(count)]
    for key in ("class_logits", "class_scores"):
        if key not in out:
            continue
        values = np.asarray(out[key]).reshape(count, -1)
        for row, value in zip(rows, values, strict=True):
            row[key] = value.copy()
    return rows


def _prediction_rows(out: dict[str, object]):
    scores = np.asarray(out["scores"]).reshape(-1)
    return zip(
        format_masks(out["masks"]),
        format_logits(out["logits"]),
        scores,
        _class_rows(out, len(scores)),
        strict=True,
    )


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


def _keep_largest(mask: np.ndarray, logit: np.ndarray):
    mask = largest(mask)
    logit = np.asarray(logit).copy()
    logit[~mask] = np.minimum(logit[~mask], -1.0)
    return mask, logit


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
        iou_thr: float = 0.0,
        presence_thr: float = 0.5,
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
        iou_thr = float(iou_thr)
        presence_thr = float(presence_thr)
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be between zero and one")
        if not 0 <= nms_thr <= 1:
            raise ValueError("nms_thr must be between zero and one")
        if not 0 <= stability_thr <= 1:
            raise ValueError("stability_thr must be between zero and one")
        if not 0 <= iou_thr <= 1:
            raise ValueError("iou_thr must be between zero and one")
        if not 0 <= presence_thr <= 1:
            raise ValueError("presence_thr must be between zero and one")
        self.single = single
        self.tiles = tuple(int(tile) for tile in tiles)
        self.points_per_side = _match(points_per_side, len(self.tiles))
        self.overlap = overlap
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.min_area = _positive_int(min_area, "min_area", zero=True)
        self.nms_thr = nms_thr
        self.stability_thr = stability_thr
        self.iou_thr = iou_thr
        self.presence_thr = presence_thr

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
        iou_thr: float = 0.0,
        presence_thr: float = 0.5,
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
            iou_thr=iou_thr,
            presence_thr=presence_thr,
        )

    @classmethod
    def from_finetune(
        cls,
        base_path: str | Path,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
        cond: int = 0,
        tiles: tuple[int, ...] = (1, 2),
        points_per_side: int | tuple[int, ...] = (10, 10),
        overlap: float = 0.25,
        batch_size: int = 64,
        min_area: int = 64,
        nms_thr: float = 0.7,
        stability_thr: float = 0.75,
        iou_thr: float = 0.0,
        presence_thr: float = 0.5,
    ) -> "GridPredictor":
        single = SinglePredictor.from_finetune(
            base_path,
            checkpoint_path,
            device=device,
            cond=cond,
        )
        return cls(
            single,
            tiles=tiles,
            points_per_side=points_per_side,
            overlap=overlap,
            batch_size=batch_size,
            min_area=min_area,
            nms_thr=nms_thr,
            stability_thr=stability_thr,
            iou_thr=iou_thr,
            presence_thr=presence_thr,
        )

    def _keep(self, item: dict[str, object]) -> bool:
        class_scores = item.get("class_scores")
        if class_scores is not None:
            values = np.asarray(class_scores).reshape(-1)
            if not values.size:
                return False
            presence = float(values[0])
            if not np.isfinite(presence) or presence < self.presence_thr:
                return False
        return (
            float(item["score"]) >= self.iou_thr
            and item["area"] >= self.min_area
            and item["stability_score"] >= self.stability_thr
            and not is_edge_cut(item)
        )

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

    def _generate(
        self,
        embed: dict[str, object],
        crop: tuple[int, int, int, int],
        tile: int,
        crop_index: int,
        points: np.ndarray,
        image_size: tuple[int, int],
    ) -> list[dict[str, object]]:
        items = []

        for batch in _batches(points, self.batch_size):
            out = self.single.predict_low(
                embed,
                point_coords=batch[:, None, :],
                point_labels=np.ones((len(batch), 1), dtype=np.int32),
                multimask=False,
            )
            for point, prediction in zip(
                batch,
                _prediction_rows(out),
                strict=True,
            ):
                mask, logit, score, class_values = prediction
                item = make_candidate(
                    mask,
                    logit,
                    score,
                    point,
                    crop,
                    tile,
                    crop_index,
                    image_size,
                )
                if item is None:
                    continue
                item.update(class_values)
                if not self._keep(item):
                    continue
                items.append(item)
        return items

    def _refine_crop(
        self,
        items: list[dict[str, object]],
        embed: dict[str, object],
        largest_component: bool,
    ) -> list[dict[str, object]]:
        refined = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            points = _local_points(chunk)
            logits_in = np.stack([item["logit"] for item in chunk])
            out = self.single.predict_low(
                embed,
                point_coords=points,
                point_labels=np.ones((len(chunk), 1), dtype=np.int32),
                mask=logits_in,
                multimask=False,
            )
            for item, point, prediction in zip(
                chunk,
                points[:, 0, :],
                _prediction_rows(out),
                strict=True,
            ):
                mask, logit, score, class_values = prediction
                if largest_component:
                    mask, logit = _keep_largest(mask, logit)
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
                if new_item is None:
                    continue
                new_item.update(class_values)
                if self._keep(new_item):
                    refined.append(new_item)
        return refined

    def _predict_crop(
        self,
        image: Image.Image,
        crop: tuple[int, int, int, int],
        tile: int,
        crop_index: int,
        points: np.ndarray,
        largest_component: bool,
    ) -> list[dict[str, object]]:
        embed = self.single.encode(image.crop(crop))
        items = self._generate(
            embed,
            crop,
            tile,
            crop_index,
            points,
            image.size,
        )
        items = filter_crop(items, self.nms_thr)
        return filter_crop(
            self._refine_crop(items, embed, largest_component),
            self.nms_thr,
        )

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image | np.ndarray,
        largest_component: bool = False,
    ) -> list[dict[str, object]]:
        image = image_data.convert_rgb(image)
        items = []
        for tile, crop_index, crop, points in self.iter_points(image.size):
            items.extend(
                self._predict_crop(
                    image,
                    crop,
                    tile,
                    crop_index,
                    points,
                    largest_component,
                )
            )
        return make_objects(filter_image(items, self.nms_thr))
