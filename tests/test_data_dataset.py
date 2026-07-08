import numpy as np

from src.data.dataset import BaseDataset, TrainDataset
from src.data.augment.prompt import box, mask, point
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

    out = point.sample_point_prompt(
        target,
        union,
        bg_prob=0.0,
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

    out = point.sample_point_prompt(
        target,
        union,
        bg_prob=1.0,
    )

    x, y = out["points"][0].astype(int)
    assert out["point_labels"].tolist() == [1]
    assert union[y, x] == 0
    assert out["target"].sum() == 0
    assert out["has_object"] is False


def test_box_prompt_uses_tight_mask_box_without_jitter():
    target = np.zeros((6, 8), dtype=np.uint8)
    target[2:5, 1:6] = 1

    out = box.jitter_mask_box(
        target,
        image_shape=(6, 8, 3),
        amount=0.0,
    )

    assert out.dtype == np.float32
    assert out.tolist() == [1.0, 2.0, 6.0, 5.0]


def test_box_prompt_jitter_stays_inside_image_and_valid():
    target = np.zeros((6, 8), dtype=np.uint8)
    target[1:5, 2:7] = 1

    out = box.jitter_mask_box(
        target,
        image_shape=(6, 8, 3),
        amount=0.5,
    )

    assert 0.0 <= out[0] < out[2] <= 8.0
    assert 0.0 <= out[1] < out[3] <= 6.0


def test_mask_prompt_none_returns_float_gt():
    target = np.zeros((6, 8), dtype=np.uint8)
    target[1:4, 2:6] = 1

    out = mask.degrade_mask_prompt(
        target,
        ops=("none",),
    )

    assert out.dtype == np.float32
    assert out.shape == target.shape
    assert np.array_equal(out, target.astype(np.float32))


def test_mask_prompt_each_op_keeps_shape_and_float_dtype():
    target = np.zeros((12, 16), dtype=np.uint8)
    target[3:9, 4:12] = 1

    for op in ("shift", "erode", "dilate", "blur", "resize"):
        out = mask.degrade_mask_prompt(
            target,
            ops=(op,),
        )
        assert out.dtype == np.float32
        assert out.shape == target.shape


def test_train_dataset_skips_empty_objects(tmp_path):
    full = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    empty = Object(
        object_id=2,
        class_id=2,
        box=(4, 4, 6, 6),
        roi=np.zeros((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [full, empty])

    dataset = TrainDataset([str(path)], config={"prompt": "box", "box_jitter": 0.0})

    assert len(dataset) == 1


def test_train_dataset_returns_box_prompt_item(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = TrainDataset([str(path)], config={"prompt": "box", "box_jitter": 0.0})
    item = dataset[0]

    assert set(item) == {"image", "prompt", "target", "has_object"}
    assert item["image"].shape == (6, 8, 3)
    assert item["prompt"]["type"] == "box"
    assert item["prompt"]["points"] is None
    assert item["prompt"]["point_labels"] is None
    assert item["prompt"]["box"].tolist() == [1.0, 1.0, 3.0, 3.0]
    assert item["prompt"]["mask"] is None
    assert item["target"].sum() == 4
    assert item["has_object"] is True


def test_train_dataset_returns_background_point_item(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = TrainDataset(
        [str(path)],
        config={"prompt": "point", "bg_prob": 1.0},
    )
    item = dataset[0]

    x, y = item["prompt"]["points"][0].astype(int)
    assert item["prompt"]["type"] == "point"
    assert item["prompt"]["point_labels"].tolist() == [1]
    assert item["target"].sum() == 0
    assert item["has_object"] is False
    assert obj.mask(item["image"].shape)[y, x] == 0


def test_train_dataset_returns_mask_prompt_item(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = TrainDataset([str(path)], config={"prompt": "mask", "mask_ops": ("none",)})
    item = dataset[0]

    assert item["prompt"]["type"] == "mask"
    assert item["prompt"]["points"] is None
    assert item["prompt"]["point_labels"] is None
    assert item["prompt"]["box"] is None
    assert np.array_equal(item["prompt"]["mask"], item["target"].astype(np.float32))
    assert item["has_object"] is True
