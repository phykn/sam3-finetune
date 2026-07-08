import numpy as np

from src.data.dataset import BaseDataset
from src.data.augment.prompt import box, point
from src.data.sample import Image, Object, Sample, save


def write_sample(path, objects):
    image = Image(array=np.zeros((6, 8, 3), dtype=np.uint8), id="img-1")
    sample = Sample(image=image, objects=objects)
    save(sample, path)
    return path


def test_base_dataset_loads_sample_json(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = BaseDataset([str(path)])
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample.image.shape == (6, 8, 3)
    assert sample.image.id == "img-1"
    assert sample.objects[0].object_id == 1


def test_point_prompt_samples_object_click_inside_mask():
    target = np.zeros((6, 8), dtype=np.uint8)
    target[1:4, 2:6] = 1
    union = target.copy()

    out = point.make(
        target,
        union,
        bg_prob=0.0,
        rng=np.random.default_rng(0),
    )

    x, y = out["points"][0].astype(int)
    assert out["point_labels"].tolist() == [1]
    assert target[y, x] == 1
    assert out["target"].sum() == target.sum()
    assert out["has_object"] is True


def test_point_prompt_samples_background_as_positive_click():
    target = np.zeros((6, 8), dtype=np.uint8)
    target[1:4, 2:6] = 1
    union = target.copy()

    out = point.make(
        target,
        union,
        bg_prob=1.0,
        rng=np.random.default_rng(1),
    )

    x, y = out["points"][0].astype(int)
    assert out["point_labels"].tolist() == [1]
    assert union[y, x] == 0
    assert out["target"].sum() == 0
    assert out["has_object"] is False


def test_box_prompt_uses_tight_mask_box_without_jitter():
    target = np.zeros((6, 8), dtype=np.uint8)
    target[2:5, 1:6] = 1

    out = box.make(
        target,
        image_shape=(6, 8, 3),
        jitter=0.0,
        rng=np.random.default_rng(0),
    )

    assert out.dtype == np.float32
    assert out.tolist() == [1.0, 2.0, 6.0, 5.0]


def test_box_prompt_jitter_stays_inside_image_and_valid():
    target = np.zeros((6, 8), dtype=np.uint8)
    target[1:5, 2:7] = 1

    out = box.make(
        target,
        image_shape=(6, 8, 3),
        jitter=0.5,
        rng=np.random.default_rng(2),
    )

    assert 0.0 <= out[0] < out[2] <= 8.0
    assert 0.0 <= out[1] < out[3] <= 6.0
