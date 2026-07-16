import numpy as np
import pytest
import torch
from PIL import Image
import src.predict.grid as grid_module
from src.predict.grid import GridPredictor
from src.predict.grid_ops.boxes import filter_crop, filter_image, find_box, is_edge_cut
from src.predict.grid_ops.candidates import make_candidate, make_objects
from src.predict.grid_ops.points import filter_points, make_points
from src.predict.grid_ops.tiles import make_crops


class FakeSingle:
    def __init__(self):
        self.crops = []
        self.batches = []
        self.masks = []
        self.refine_logits = []
        self.events = []

    def encode(self, image):
        assert not torch.is_grad_enabled()
        self.crops.append(image.size)
        crop_index = len(self.crops) - 1
        self.events.append(("encode", crop_index))
        width, height = image.size
        return {"orig_hw": (height, width), "crop_index": crop_index}

    def predict_low(
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
            self.events.append(("refine", embed["crop_index"]))
        else:
            self.events.append(("predict", embed["crop_index"]))
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
    def predict_low(self, *args, **kwargs):
        out = super().predict_low(*args, **kwargs)
        score = 0.8 if kwargs.get("mask") is not None else 0.2
        out["class_scores"] = np.tile(
            np.array([[[score, 1 - score]]], dtype=np.float32),
            (len(out["scores"]), 1, 1),
        )
        out["class_logits"] = np.log(out["class_scores"] / (1 - out["class_scores"]))
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
    assert not hasattr(predictor, "before")
    assert not hasattr(predictor, "after")
    assert not hasattr(predictor, "expand_mask")
    assert all(
        set(item) == {"object_id", "class_id", "box", "roi", "points", "metrics"}
        for item in out
    )
    assert all(item["roi"].dtype == np.bool_ for item in out)
    for crop_index in range(4):
        assert single.events.index(("refine", crop_index)) < single.events.index(
            ("encode", crop_index + 1)
        )
    assert all(shape[1:] == (1, 2) for shape, _labels, _multimask in single.batches)
    assert sum(mask is not None for mask in single.masks) == len(single.refine_logits)
    assert len(single.refine_logits) < len(out)
    assert any(logit.ndim == 3 and logit.shape[0] > 1 for logit in single.refine_logits)
    assert all(logit.dtype.kind == "f" for logit in single.refine_logits)


def test_grid_predictor_accepts_numpy_image():
    predictor = GridPredictor(
        FakeSingle(),
        tiles=(1,),
        points_per_side=1,
        min_area=1,
        nms_thr=1.0,
    )

    out = predictor.predict(np.zeros((64, 48, 3), dtype=np.uint8))

    assert len(out) == 1


def test_grid_predictor_keeps_refined_class_scores():
    predictor = GridPredictor(
        FakeClassSingle(),
        tiles=(1,),
        points_per_side=1,
        batch_size=1,
        nms_thr=1.0,
        min_area=1,
        presence_thr=0.0,
    )

    out = predictor.predict(Image.new("RGB", (64, 64)))

    assert len(out) == 1
    np.testing.assert_allclose(out[0]["metrics"]["class_scores"], [0.8, 0.2])
    assert len(out[0]["metrics"]["class_logits"]) == 2


def test_grid_predictor_filters_presence_before_refinement():
    single = FakeClassSingle()
    predictor = GridPredictor(
        single,
        tiles=(1,),
        points_per_side=1,
        min_area=1,
        nms_thr=1.0,
    )

    out = predictor.predict(Image.new("RGB", (64, 64)))

    assert out == []
    assert all(event[0] != "refine" for event in single.events)


def test_grid_predictor_filters_raw_iou_before_refinement():
    class LowIouSingle(FakeSingle):
        def predict_low(self, *args, **kwargs):
            out = super().predict_low(*args, **kwargs)
            out["scores"].fill(-0.2)
            return out

    single = LowIouSingle()
    predictor = GridPredictor(
        single,
        tiles=(1,),
        points_per_side=1,
        min_area=1,
        nms_thr=1.0,
        iou_thr=0.4,
    )

    out = predictor.predict(Image.new("RGB", (64, 64)))

    assert out == []
    assert all(event[0] != "refine" for event in single.events)


def test_grid_predictor_skips_presence_filter_for_base_model():
    predictor = GridPredictor(
        FakeSingle(),
        tiles=(1,),
        points_per_side=1,
        min_area=1,
        nms_thr=1.0,
        presence_thr=1.0,
    )

    assert len(predictor.predict(Image.new("RGB", (64, 64)))) == 1


def test_grid_largest_component_changes_final_boxes_and_nms(monkeypatch):
    class ArtifactSingle(FakeSingle):
        def predict_low(
            self,
            embed,
            point_coords,
            point_labels,
            mask=None,
            multimask=False,
        ):
            count = len(point_coords)
            masks = np.zeros((count, 1, 8, 8), dtype=bool)
            for index, point in enumerate(point_coords[:, 0]):
                if point[0] < 4:
                    masks[index, 0, 0:2, 0:2] = True
                    if mask is not None:
                        masks[index, 0, 0, 5] = True
                else:
                    masks[index, 0, 0:2, 5:7] = True
            return {
                "masks": masks,
                "scores": np.ones((count, 1), dtype=np.float32),
                "logits": np.where(masks, 2.0, -2.0).astype(np.float32),
            }

    predictor = GridPredictor(
        ArtifactSingle(),
        tiles=(1,),
        points_per_side=1,
        min_area=1,
        nms_thr=0.1,
    )
    points = np.array([[1.0, 1.0], [6.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(
        predictor,
        "iter_points",
        lambda _size: iter([(1, 0, (0, 0, 8, 8), points)]),
    )

    original = predictor.predict(Image.new("RGB", (8, 8)))
    calls = []
    component = grid_module.largest

    def record(mask):
        calls.append(mask.shape)
        return component(mask)

    monkeypatch.setattr(grid_module, "largest", record)
    filtered = predictor.predict(
        Image.new("RGB", (8, 8)),
        largest_component=True,
    )

    assert len(original) == 1
    assert original[0]["box"] == (0, 0, 6, 2)
    assert calls == [(8, 8), (8, 8)]
    assert [value["box"] for value in filtered] == [
        (0, 0, 2, 2),
        (5, 0, 7, 2),
    ]


def test_make_objects_resizes_logit_to_compact_roi():
    item = {
        "logit": np.array(
            [
                [2.0, -2.0],
                [-2.0, -2.0],
            ],
            dtype=np.float32,
        ),
        "bbox": (10, 20, 14, 24),
        "crop": (10, 20, 18, 28),
        "point": (13.0, 22.0),
        "score": 1.5,
        "stability_score": 0.8,
        "class_logits": np.array([2.0, -2.0], dtype=np.float32),
        "class_scores": np.array([0.88, 0.12], dtype=np.float32),
    }

    objects = make_objects([item])

    assert len(objects) == 1
    assert objects[0]["box"] == (10, 20, 14, 24)
    assert objects[0]["roi"].shape == (4, 4)
    assert 1 < objects[0]["roi"].sum() < 16
    assert objects[0]["points"] == [[13.0, 22.0, 1]]
    assert objects[0]["metrics"]["score"] == 1.5
    assert objects[0]["metrics"]["stability"] == 0.8
    assert objects[0]["metrics"]["class_logits"] == [2.0, -2.0]
    full = np.asarray(
        Image.fromarray(item["logit"], mode="F").resize(
            (8, 8),
            Image.Resampling.BILINEAR,
        )
    )
    np.testing.assert_array_equal(objects[0]["roi"], full[:4, :4] > 0)


def test_make_objects_drops_low_res_pixel_lost_during_resize():
    mask = np.zeros((288, 288), dtype=bool)
    mask[100, 100] = True
    logit = np.where(mask, 2.0, -2.0).astype(np.float32)
    item = make_candidate(
        mask,
        logit,
        0.75,
        np.array([3.0, 3.0], dtype=np.float32),
        (0, 0, 8, 8),
        1,
        0,
        (8, 8),
    )

    assert item is not None
    assert make_objects([item]) == []


def test_make_candidate_keeps_full_low_res_logit_and_scales_box_area():
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
    assert "segmentation" not in item
    assert item["logit"].shape == (4, 4)
    assert item["logit"].dtype == np.float16
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
        "score": 1.0,
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
        {"iou_thr": -0.1},
        {"iou_thr": 1.1},
        {"presence_thr": -0.1},
        {"presence_thr": 1.1},
    ],
)
def test_grid_predictor_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        GridPredictor(FakeSingle(), **kwargs)


def test_grid_predictor_from_finetune_delegates_and_keeps_grid_options(monkeypatch):
    calls = []
    single = FakeSingle()

    def make_single(cls, base_path, checkpoint_path, device="cuda", cond=0):
        calls.append((base_path, checkpoint_path, device, cond))
        return single

    monkeypatch.setattr(
        "src.predict.grid.SinglePredictor.from_finetune",
        classmethod(make_single),
    )

    predictor = GridPredictor.from_finetune(
        "base.pt",
        "last.pt",
        device="cpu",
        cond=2,
        tiles=(1,),
        points_per_side=3,
        iou_thr=0.2,
        presence_thr=0.7,
    )

    assert calls == [("base.pt", "last.pt", "cpu", 2)]
    assert predictor.single is single
    assert predictor.points_per_side == (3,)
    assert predictor.iou_thr == 0.2
    assert predictor.presence_thr == 0.7
