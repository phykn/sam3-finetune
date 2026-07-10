import numpy as np
import pytest
import torch
from PIL import Image
from src.predict.grid import GridPredictor
from src.predict.grid_ops.boxes import filter_crop, filter_image, find_box, is_edge_cut
from src.predict.grid_ops.candidates import expand_mask, make_candidate
from src.predict.grid_ops.points import filter_points, make_points
from src.predict.grid_ops.tiles import make_crops


class FakeSingle:
    def __init__(self):
        self.crops = []
        self.batches = []
        self.masks = []
        self.refine_logits = []

    def encode(self, image):
        assert not torch.is_grad_enabled()
        self.crops.append(image.size)
        width, height = image.size
        return {"orig_hw": (height, width)}

    def _predict_low(
        self,
        embed,
        point_coords,
        point_labels,
        mask=None,
        multimask=False,
    ):
        assert not torch.is_grad_enabled()
        self.batches.append((point_coords.shape, point_labels.shape, multimask))
        self.masks.append(mask)
        if mask is not None:
            self.refine_logits.append(np.asarray(mask))
        height, width = embed["orig_hw"]
        masks = np.zeros((len(point_coords), 1, height, width), dtype=bool)
        for index, point in enumerate(point_coords[:, 0, :]):
            x = min(max(int(point[0]), 0), width - 1)
            y = min(max(int(point[1]), 0), height - 1)
            masks[index, 0, max(y - 1, 0) : y + 1, max(x - 1, 0) : x + 1] = True
        return {
            "masks": masks,
            "scores": np.ones((len(point_coords), 1), dtype=np.float32),
            "logits": np.where(masks, 2.0, -2.0).astype(np.float32),
        }


class FakeClassSingle(FakeSingle):
    def _predict_low(self, *args, **kwargs):
        out = super()._predict_low(*args, **kwargs)
        score = 0.8 if kwargs.get("mask") is not None else 0.2
        out["class_scores"] = np.tile(
            np.array([[[score, 1 - score]]], dtype=np.float32),
            (len(out["scores"]), 1, 1),
        )
        return out


def test_make_crops_splits_tile_grid():
    assert make_crops((8, 6), 1, 0.25) == [(0, 0, 8, 6)]
    assert make_crops((8, 6), 2, 0.0) == [
        (0, 0, 4, 3),
        (4, 0, 8, 3),
        (0, 3, 4, 6),
        (4, 3, 8, 6),
    ]


def test_make_points_places_centers_inside_crop():
    points = make_points((8, 4), 2)

    assert points.tolist() == [
        [2.0, 1.0],
        [6.0, 1.0],
        [2.0, 3.0],
        [6.0, 3.0],
    ]


def test_filter_points_keeps_only_owner_tile_overlap_area():
    left = filter_points(
        np.array([[3.5, 2.0], [4.5, 2.0]], dtype=np.float32),
        (0, 0, 5, 8),
        2,
        0,
        (8, 8),
    )
    right = filter_points(
        np.array([[0.5, 2.0], [2.0, 2.0]], dtype=np.float32),
        (3, 0, 8, 8),
        2,
        1,
        (8, 8),
    )

    assert left.tolist() == [[3.5, 2.0]]
    assert right.tolist() == [[2.0, 2.0]]


def test_grid_predictor_runs_tiles_and_batches_points():
    single = FakeSingle()
    predictor = GridPredictor(
        single,
        tiles=(1, 2),
        points_per_side=(2, 1),
        overlap=0.0,
        batch_size=3,
        nms_thr=1.0,
        min_area=1,
    )

    out = predictor.predict(Image.new("RGB", (64, 64)))

    assert len(single.crops) == 5
    assert len(out) == 8
    assert predictor.after == out
    assert len(predictor.before) == len(out)
    assert {item["tile"] for item in out} == {1, 2}
    assert all("mask" not in item for item in out)
    assert all("refine_logit" not in item for item in out)
    assert all(item["segmentation"].shape[0] <= 2 for item in out)
    assert all(shape[1:] == (1, 2) for shape, _labels, _multimask in single.batches)
    assert sum(mask is not None for mask in single.masks) == len(single.refine_logits)
    assert len(single.refine_logits) < len(out)
    assert any(logit.ndim == 3 and logit.shape[0] > 1 for logit in single.refine_logits)
    assert all(logit.dtype.kind == "f" for logit in single.refine_logits)


def test_grid_predictor_keeps_refined_class_scores():
    predictor = GridPredictor(
        FakeClassSingle(),
        tiles=(1,),
        points_per_side=1,
        batch_size=1,
        nms_thr=1.0,
        min_area=1,
    )

    out = predictor.predict(Image.new("RGB", (64, 64)))

    assert len(out) == 1
    np.testing.assert_allclose(out[0]["class_scores"], [0.8, 0.2])


def test_expand_mask_expands_roi_segmentation():
    item = {
        "segmentation": np.ones((2, 3), dtype=bool),
        "bbox": (2, 1, 5, 3),
    }

    mask = expand_mask(item, (8, 6))

    assert mask.shape == (6, 8)
    assert mask.sum() == 6
    assert mask[1:3, 2:5].all()


def test_expand_mask_resizes_low_res_roi_at_the_end():
    item = {
        "segmentation": np.ones((2, 2), dtype=bool),
        "bbox": (2, 1, 6, 5),
    }

    mask = expand_mask(item, (8, 6))

    assert mask.shape == (6, 8)
    assert mask.sum() == 16
    assert mask[1:5, 2:6].all()


def test_expand_mask_uses_bilinear_logit_when_available():
    item = {
        "segmentation": np.ones((2, 2), dtype=bool),
        "logit": np.array(
            [
                [2.0, -2.0],
                [-2.0, -2.0],
            ],
            dtype=np.float32,
        ),
        "bbox": (0, 0, 4, 4),
    }

    mask = expand_mask(item, (4, 4))

    assert mask.shape == (4, 4)
    assert 1 < mask.sum() < 16


def test_make_candidate_keeps_low_res_roi_and_scales_box_area():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    logit = np.where(mask, 2.0, -2.0).astype(np.float32)

    item = make_candidate(
        mask,
        logit,
        0.75,
        np.array([2.0, 2.0], dtype=np.float32),
        (10, 20, 18, 28),
        2,
        0,
        (32, 32),
    )

    assert item["bbox"] == (12, 22, 16, 26)
    assert item["area"] == 16
    assert item["segmentation"].shape == (2, 2)
    assert item["logit"].shape == (2, 2)
    assert item["low_box"] == (1, 1, 3, 3)
    assert item["low_shape"] == (4, 4)
    assert item["stability_score"] == 1.0
    assert item["point"] == (12.0, 22.0)


def test_find_box():
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:3, 2:5] = True

    box = find_box(mask)

    assert box == (2, 1, 5, 3)


def test_is_edge_cut_ignores_outer_image_edges():
    base = {
        "crop": (0, 0, 40, 40),
        "image_size": (80, 80),
        "low_shape": (20, 20),
    }

    assert is_edge_cut({**base, "low_box": (5, 5, 18, 10)})
    assert not is_edge_cut({**base, "low_box": (5, 5, 10, 10)})
    assert is_edge_cut(
        {
            "crop": (0, 0, 40, 40),
            "image_size": (80, 80),
            "low_shape": (20, 20),
            "low_box": (5, 5, 16, 10),
        },
        atol=4,
    )
    assert not is_edge_cut(
        {
            "crop": (0, 0, 80, 80),
            "image_size": (80, 80),
            "low_shape": (20, 20),
            "low_box": (0, 1, 2, 3),
        }
    )
    assert is_edge_cut(
        {
            "crop": (40, 0, 80, 40),
            "image_size": (80, 80),
            "low_shape": (20, 20),
            "low_box": (4, 5, 10, 10),
        }
    )


def test_image_nms_ranks_smaller_crop_before_higher_score():
    full = {
        "bbox": (1, 1, 7, 7),
        "score": 3.0,
        "stability_score": 1.0,
        "crop": (0, 0, 8, 8),
        "image_size": (8, 8),
    }
    small = {
        **full,
        "score": 1.0,
        "crop": (0, 0, 4, 4),
    }

    out = filter_image([full, small], nms_thr=0.5)

    assert out == [small]


def test_crop_nms_ranks_higher_score_before_stability():
    stable = {
        "bbox": (1, 1, 7, 7),
        "score": 1.0,
        "stability_score": 1.0,
        "crop": (0, 0, 8, 8),
    }
    scored = {**stable, "score": 2.0, "stability_score": 0.8}

    out = filter_crop([stable, scored], nms_thr=0.5)

    assert out == [scored]


def test_image_nms_has_no_count_limit():
    items = [
        {
            "bbox": (index * 10, 0, index * 10 + 4, 4),
            "score": 1.0,
            "stability_score": 1.0,
            "crop": (0, 0, 100, 100),
            "image_size": (100, 100),
        }
        for index in range(50)
    ]

    out = filter_image(items, nms_thr=0.5)

    assert len(out) == 50


def test_grid_predictor_filters_by_stability_threshold():
    predictor = GridPredictor(FakeSingle(), stability_thr=0.8)
    item = {
        "area": 100,
        "stability_score": 0.75,
        "low_box": (2, 2, 4, 4),
        "low_shape": (8, 8),
        "crop": (0, 0, 8, 8),
        "image_size": (8, 8),
    }

    assert not predictor._keep(item)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tiles": ()},
        {"tiles": 1},
        {"tiles": (1, 1)},
        {"tiles": (0,)},
        {"tiles": (1.5,)},
        {"points_per_side": 0},
        {"points_per_side": 1.5},
        {"points_per_side": (1, 2, 3)},
        {"overlap": -0.1},
        {"overlap": 1.0},
        {"batch_size": 0},
        {"min_area": -1},
        {"nms_thr": -0.1},
        {"nms_thr": 1.1},
        {"stability_thr": -0.1},
        {"stability_thr": 1.1},
    ],
)
def test_grid_predictor_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        GridPredictor(FakeSingle(), **kwargs)
