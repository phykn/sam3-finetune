import numpy as np
from PIL import Image

import scripts.make_finetune_candidates as candidates
from scripts.make_finetune_candidates import assign_class
from src.data.sample import Image as DataImage
from src.data.sample import Object, Sample, load, save


def test_generator_updates_single_dataset_root():
    assert candidates.OUT == candidates.SOURCE


def candidate(box, roi, point):
    return {
        "box": box,
        "roi": np.asarray(roi, dtype=np.uint8),
        "points": [[point[0], point[1], 1]],
    }


def test_assign_class_uses_point_and_mask_overlap():
    item = candidate((1, 1, 5, 5), np.ones((4, 4)), (2, 2))
    regions = {1: [(0, 0, 4, 5)], 2: [(4, 0, 6, 5)]}

    assert assign_class(item, regions, (6, 6)) == (1, 0.75)


def test_assign_class_rejects_overlap_below_half():
    item = candidate((0, 0, 4, 4), np.ones((4, 4)), (1, 1))
    regions = {1: [(0, 0, 1, 4)]}

    assert assign_class(item, regions, (4, 4)) is None


def test_assign_class_rejects_point_outside_region():
    item = candidate((0, 0, 4, 4), np.ones((4, 4)), (3, 2))
    regions = {1: [(0, 0, 3, 4)]}

    assert assign_class(item, regions, (4, 4)) is None


def test_assign_class_rejects_equal_best_classes():
    item = candidate((0, 0, 4, 4), np.ones((4, 4)), (2, 2))
    regions = {1: [(0, 0, 4, 4)], 2: [(0, 0, 4, 4)]}

    assert assign_class(item, regions, (4, 4)) is None


def test_make_sample_writes_assigned_candidates_and_preserves_background(
    monkeypatch,
    tmp_path,
):
    asset = tmp_path / "asset"
    source = tmp_path / "source"
    out = tmp_path / "out"
    asset.mkdir()
    Image.new("RGB", (8, 6), "green").save(asset / "frog.jpg")

    background = Sample(
        image=DataImage(np.zeros((6, 8, 3), dtype=np.uint8), id="frog.jpg"),
        objects=[Object(1, 0, (0, 0, 2, 2), np.ones((2, 2)))],
    )
    path = source / "train" / "0_background" / "frog.json"
    path.parent.mkdir(parents=True)
    save(background, path)
    frog_source = source / "train" / "1_frog" / "frog.json"
    frog_source.parent.mkdir(parents=True)
    save(Sample(background.image, []), frog_source)

    class FakePredictor:
        def __init__(self):
            self.calls = 0

        def predict(self, _image):
            self.calls += 1
            return [
                {
                    "object_id": 8,
                    "class_id": None,
                    "box": (0, 0, 4, 6),
                    "roi": np.ones((6, 4), dtype=np.uint8),
                    "points": [[2.0, 2.0, 1]],
                    "metrics": {"score": 0.9, "stability": 0.8},
                },
                {
                    "object_id": 9,
                    "class_id": None,
                    "box": (4, 0, 8, 6),
                    "roi": np.ones((6, 4), dtype=np.uint8),
                    "points": [[6.0, 2.0, 1]],
                    "metrics": {"score": 0.85, "stability": 0.81},
                },
            ]

    monkeypatch.setattr(candidates, "ASSET", asset)
    monkeypatch.setattr(candidates, "SOURCE", source)
    monkeypatch.setattr(candidates, "OUT", out)
    predictor = FakePredictor()

    counts = candidates.make_sample(
        predictor,
        "train",
        "frog.jpg",
        {1: [(0, 0, 4, 6)], 2: [(4, 0, 8, 6)]},
    )

    assert predictor.calls == 1
    assert counts == {0: 1, 1: 1, 2: 1}
    saved_background = load(out / "train" / "0_background" / "frog.json")
    frog = load(out / "train" / "1_frog" / "frog.json")
    leaf = load(out / "train" / "2_leaf" / "frog.json")
    assert saved_background.objects[0].mask((6, 8)).sum() == 4
    assert frog.objects[0].object_id == 1
    assert frog.objects[0].class_id == 1
    assert frog.objects[0].metrics["region_overlap"] == 1.0
    assert frog.objects[0].meta["source_point"] == [2.0, 2.0]
    assert frog.objects[0].meta["class_regions"] == [[0, 0, 4, 6]]
    assert leaf.objects[0].object_id == 1
    assert leaf.objects[0].class_id == 2


def test_make_preview_keeps_image_size_and_draws_objects():
    image = Image.new("RGB", (8, 6), "green")
    obj = Object(
        1,
        1,
        (1, 1, 5, 5),
        np.ones((4, 4)),
        metrics={"score": 0.9, "stability": 0.8, "region_overlap": 0.75},
    )

    preview = candidates.make_preview(image, {1: [obj], 2: []})

    assert preview.size == image.size
    assert np.any(np.asarray(preview) != np.asarray(image))


def test_frog_fallback_requires_a_large_candidate():
    small = Object(1, 1, (0, 0, 2, 2), np.ones((2, 2)))
    large = Object(2, 1, (0, 0, 8, 4), np.ones((4, 8)))
    region = [(0, 0, 8, 8)]

    assert candidates.needs_frog_fallback([small], region)
    assert not candidates.needs_frog_fallback([small, large], region)


def test_make_sample_falls_back_to_source_frog_mask(monkeypatch, tmp_path):
    asset = tmp_path / "asset"
    source = tmp_path / "source"
    out = tmp_path / "out"
    asset.mkdir()
    Image.new("RGB", (8, 8), "green").save(asset / "frog.jpg")
    data_image = DataImage(np.zeros((8, 8, 3), dtype=np.uint8), id="frog.jpg")
    for folder, objects in {
        "0_background": [Object(1, 0, (0, 0, 2, 2), np.ones((2, 2)))],
        "1_frog": [
            Object(
                1,
                1,
                (1, 1, 7, 7),
                np.ones((6, 6)),
                metrics={"score": 0.8},
            )
        ],
    }.items():
        path = source / "train" / folder / "frog.json"
        path.parent.mkdir(parents=True)
        save(Sample(data_image, objects), path)

    class FakePredictor:
        def predict(self, _image):
            return [
                {
                    "box": (1, 1, 3, 3),
                    "roi": np.ones((2, 2), dtype=np.uint8),
                    "points": [[2.0, 2.0, 1]],
                    "metrics": {"score": 0.9, "stability": 0.9},
                }
            ]

    monkeypatch.setattr(candidates, "ASSET", asset)
    monkeypatch.setattr(candidates, "SOURCE", source)
    monkeypatch.setattr(candidates, "OUT", out)
    candidates.make_sample(
        FakePredictor(),
        "train",
        "frog.jpg",
        {1: [(0, 0, 8, 8)], 2: [(6, 6, 8, 8)]},
    )

    frog = load(out / "train" / "1_frog" / "frog.json")
    assert frog.objects[0].roi.sum() == 36
    assert frog.objects[0].meta["candidate_source"] == "box_fallback"
