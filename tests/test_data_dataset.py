import numpy as np

import src.data.dataset as dataset_mod
from src.data.augment.image import crop as image_crop
from src.data.augment.image import flip as image_flip
from src.data.augment.image import rotate as image_rotate
from src.data.augment.image import resize as image_resize
from src.data.augment.image import zoom_out as image_zoom_out
from src.data.dataset import (
    MASK_SIZE,
    SIZE,
    BaseDataset,
    IMAGE_OPS,
    TrainDataset,
    ValidDataset,
)
from src.data.augment.prompt import box, mask, point
from src.data.sample import Image, Object, Sample, save


def write_sample(path, objects, image=None):
    array = np.zeros((6, 8, 3), dtype=np.uint8) if image is None else image
    sample = Sample(image=Image(array=array, id="img-1"), objects=objects)
    save(sample, path)
    return path


def test_base_dataset_returns_box_prompt_item(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = BaseDataset(
        [str(path)],
        prompts=["box"],
        box_jitter=0.0,
        size=8,
        mask_size=8,
    )
    item = dataset[0]

    assert len(dataset) == 1
    assert item["image"].shape == (8, 8, 3)
    assert item["prompt"]["type"] == "box"
    assert item["prompt"]["box"].tolist() == [1.0, 1.0, 3.0, 3.0]
    assert item["target"].sum() == 4
    assert item["mask_valid"] is True
    assert item["is_auto_bg"] is False


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
    assert out["has_mask"] is True
    assert out["is_auto_bg"] is False


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
    assert out["has_mask"] is False
    assert out["is_auto_bg"] is True


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

    dataset = TrainDataset([str(path)])

    assert len(dataset) == 1


def test_train_dataset_enables_augmentations_by_default(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = TrainDataset(
        [str(path)],
        bg_prob=0.3,
        box_jitter=0.4,
    )

    assert dataset.image_aug is True
    assert dataset.shape_aug is True
    assert dataset.scale == (0.5, 1.5)
    assert dataset.size == SIZE
    assert dataset.mask_size == MASK_SIZE
    assert dataset.bg_prob == 0.3
    assert dataset.box_jitter == 0.4


def test_valid_dataset_uses_point_prompt_without_aug(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = ValidDataset([str(path)])
    assert isinstance(dataset, BaseDataset)
    assert dataset.prompts == ("point",)
    assert dataset.bg_prob == 0.2
    assert dataset.box_jitter == 0.0
    assert dataset.image_aug is False
    assert dataset.shape_aug is False

    item = ValidDataset([str(path)], bg_prob=0.0)[0]
    assert item["image"].shape == (SIZE, SIZE, 3)
    assert item["target"].shape == (MASK_SIZE, MASK_SIZE)
    assert item["prompt"]["type"] == "point"
    assert item["mask_valid"] is True
    assert item["is_auto_bg"] is False


def test_base_dataset_returns_train_item_without_aug(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = BaseDataset(
        [str(path)],
        prompts=["box"],
        box_jitter=0.0,
        size=8,
        mask_size=8,
    )
    item = dataset[0]

    assert set(item) == {"image", "prompt", "target", "mask_valid", "is_auto_bg"}
    assert item["image"].shape == (8, 8, 3)
    assert item["prompt"]["type"] == "box"
    assert item["prompt"]["points"] is None
    assert item["prompt"]["point_labels"] is None
    assert item["prompt"]["box"].tolist() == [1.0, 1.0, 3.0, 3.0]
    assert item["prompt"]["mask"] is None
    assert item["target"].sum() == 4
    assert item["mask_valid"] is True
    assert item["is_auto_bg"] is False


def test_base_dataset_resizes_target_as_soft_float_mask(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])
    dataset = BaseDataset(
        [str(path)],
        prompts=["box"],
        box_jitter=0.0,
        size=8,
        mask_size=5,
    )

    item = dataset[0]

    assert item["target"].dtype == np.float32
    assert item["target"].shape == (5, 5)
    assert item["target"].min() >= 0.0
    assert item["target"].max() <= 1.0
    assert ((item["target"] > 0.0) & (item["target"] < 1.0)).any()


def test_base_dataset_adds_condition_and_label_from_sample_index(tmp_path):
    obj_a = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    obj_b = Object(
        object_id=2,
        class_id=3,
        box=(2, 2, 4, 4),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path_a = write_sample(tmp_path / "a.json", [obj_a])
    path_b = write_sample(tmp_path / "b.json", [obj_b])

    dataset = BaseDataset(
        [str(path_a), str(path_b)],
        conds=[2, 5],
        labels=[
            {"target": [1, 0, 0], "weight": [1, 1, 0]},
            {"target": [0, 1, 1], "weight": [1, 0, 1]},
        ],
        prompts=["box"],
        box_jitter=0.0,
        size=8,
        mask_size=8,
    )

    first = dataset[0]
    second = dataset[1]

    assert first["cond"] == 2
    assert first["label_target"].tolist() == [1.0, 0.0, 0.0]
    assert first["label_weight"].tolist() == [1.0, 1.0, 0.0]
    assert second["cond"] == 5
    assert second["label_target"].tolist() == [0.0, 0.0, 0.0]
    assert second["label_weight"].tolist() == [1.0, 0.0, 0.0]


def test_base_dataset_marks_object_sample_for_mask_and_full_label_loss(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = BaseDataset(
        [str(path)],
        labels=[{"target": [1, 0, 1], "weight": [1, 1, 1]}],
        prompts=["box"],
        box_jitter=0.0,
        size=8,
        mask_size=8,
    )

    item = dataset[0]

    assert item["mask_valid"] is True
    assert item["is_auto_bg"] is False
    assert item["label_target"].tolist() == [1.0, 0.0, 1.0]
    assert item["label_weight"].tolist() == [1.0, 1.0, 1.0]


def test_base_dataset_marks_background_sample_for_label_only(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = BaseDataset(
        [str(path)],
        labels=[{"target": [0, 1, 1], "weight": [1, 1, 1]}],
        prompts=["box"],
        box_jitter=0.0,
        size=8,
        mask_size=8,
    )

    item = dataset[0]

    assert item["mask_valid"] is False
    assert item["is_auto_bg"] is False
    assert item["label_target"].tolist() == [0.0, 0.0, 0.0]
    assert item["label_weight"].tolist() == [1.0, 0.0, 0.0]


def test_base_dataset_auto_background_point_uses_label_only(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = BaseDataset(
        [str(path)],
        labels=[{"target": [1, 0, 1], "weight": [1, 1, 1]}],
        prompts=["point"],
        bg_prob=1.0,
        size=8,
        mask_size=8,
    )

    item = dataset[0]

    assert item["target"].sum() == 0
    assert item["mask_valid"] is False
    assert item["is_auto_bg"] is True
    assert item["label_target"].tolist() == [0.0, 0.0, 0.0]
    assert item["label_weight"].tolist() == [1.0, 0.0, 0.0]


def test_train_dataset_inherits_base_behavior(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = TrainDataset(
        [str(path)],
    )

    assert isinstance(dataset, BaseDataset)


def test_base_dataset_returns_background_point_item(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = BaseDataset(
        [str(path)],
        prompts=["point"],
        bg_prob=1.0,
        size=8,
        mask_size=8,
    )
    item = dataset[0]

    x, y = item["prompt"]["points"][0].astype(int)
    assert item["prompt"]["type"] == "point"
    assert item["prompt"]["point_labels"].tolist() == [1]
    assert item["target"].sum() == 0
    assert item["mask_valid"] is False
    assert item["is_auto_bg"] is True
    assert obj.mask(item["image"].shape)[y, x] == 0


def test_base_dataset_returns_mask_prompt_item(tmp_path, monkeypatch):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    monkeypatch.setattr(mask.np.random, "choice", lambda ops: "none")
    dataset = BaseDataset(
        [str(path)],
        prompts=["mask"],
        size=8,
        mask_size=8,
    )
    item = dataset[0]

    assert item["prompt"]["type"] == "mask"
    assert item["prompt"]["points"] is None
    assert item["prompt"]["point_labels"] is None
    assert item["prompt"]["box"] is None
    assert np.array_equal(item["prompt"]["mask"], item["target"].astype(np.float32))
    assert item["mask_valid"] is True
    assert item["is_auto_bg"] is False


def test_base_dataset_applies_each_image_aug_op(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(10, 10, 20, 20),
        roi=np.ones((10, 10), dtype=np.uint8),
    )
    image = np.full((40, 40, 3), 120, dtype=np.uint8)
    path = write_sample(tmp_path / "sample.json", [obj], image=image)

    for op in IMAGE_OPS:
        dataset = BaseDataset(
            [str(path)],
            prompts=["box"],
            box_jitter=0.0,
            image_aug=True,
            image_ops=[op],
            size=40,
            mask_size=40,
        )
        item = dataset[0]

        assert item["image"].shape == image.shape
        assert item["image"].dtype == np.uint8
        assert item["target"].shape == image.shape[:2]
        assert item["prompt"]["box"].tolist() == [10.0, 10.0, 20.0, 20.0]


def test_base_dataset_image_dropout_adds_black_hole(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(10, 10, 20, 20),
        roi=np.ones((10, 10), dtype=np.uint8),
    )
    image = np.full((40, 40, 3), 120, dtype=np.uint8)
    path = write_sample(tmp_path / "sample.json", [obj], image=image)
    dataset = BaseDataset(
        [str(path)],
        prompts=["box"],
        image_aug=True,
        image_ops=["dropout"],
        size=40,
        mask_size=40,
    )

    item = dataset[0]

    assert (item["image"] == 0).any()


def test_resize_keeps_ratio_and_pads_right_or_bottom():
    wide_image = np.full((4, 8, 3), 100, dtype=np.uint8)
    wide_mask = np.ones((4, 8), dtype=np.uint8)
    out_image, out_mask = image_resize.resize(wide_image, wide_mask, size=(8, 8))

    assert out_image.shape == (8, 8, 3)
    assert out_mask.shape == (8, 8)
    assert (out_image[:4, :] == 100).all()
    assert (out_mask[:4, :] == 1).all()
    assert (out_image[4:, :] == 0).all()
    assert (out_mask[4:, :] == 0).all()

    tall_image = np.full((8, 4, 3), 100, dtype=np.uint8)
    tall_mask = np.ones((8, 4), dtype=np.uint8)
    out_image, out_mask = image_resize.resize(tall_image, tall_mask, size=(8, 8))

    assert (out_image[:, :4] == 100).all()
    assert (out_mask[:, :4] == 1).all()
    assert (out_image[:, 4:] == 0).all()
    assert (out_mask[:, 4:] == 0).all()


def test_random_crop_uses_square_crop_before_resize():
    image = np.full((4, 8, 3), 100, dtype=np.uint8)
    mask = np.ones((4, 8), dtype=np.uint8)

    out_image, out_mask = image_crop.random_crop(image, mask, scale=0.5)

    assert out_image.shape == (8, 8, 3)
    assert out_mask.shape == (8, 8)
    assert (out_image == 100).all()
    assert (out_mask == 1).all()


def test_random_zoom_out_uses_square_canvas_before_resize():
    image = np.full((4, 8, 3), 100, dtype=np.uint8)
    mask = np.ones((4, 8), dtype=np.uint8)

    out_image, out_mask = image_zoom_out.random_zoom_out(image, mask, scale=2.0)

    assert out_image.shape == (8, 8, 3)
    assert out_mask.shape == (8, 8)
    assert out_image.sum() > 0
    assert out_mask.sum() > 0
    assert (out_image == 0).any()
    assert (out_mask == 0).any()


def test_random_flip_uses_numpy_horizontal_flip(monkeypatch):
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    mask = np.arange(8, dtype=np.uint8).reshape(2, 4)

    monkeypatch.setattr(image_flip.np.random, "choice", lambda values: "horizontal")
    out_image, out_mask = image_flip.random_flip(image, mask)

    assert np.array_equal(out_image, np.fliplr(image))
    assert np.array_equal(out_mask, np.fliplr(mask))


def test_base_dataset_shape_aug_crops_when_scale_is_less_than_one(
    tmp_path,
    monkeypatch,
):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(10, 10, 20, 20),
        roi=np.ones((10, 10), dtype=np.uint8),
    )
    image = np.full((40, 40, 3), 120, dtype=np.uint8)
    path = write_sample(tmp_path / "sample.json", [obj], image=image)
    calls = []

    def fake_crop(
        image: np.ndarray,
        pair: np.ndarray,
        scale: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(("crop", scale))
        return image, pair

    def fake_zoom_out(
        image: np.ndarray,
        pair: np.ndarray,
        scale: float = 1.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(("zoom_out", scale))
        return image, pair

    def fake_rotate(
        image: np.ndarray,
        pair: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(("rotate", 1.0))
        return image, pair

    def fake_flip(
        image: np.ndarray,
        pair: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(("flip", 1.0))
        return image, pair

    monkeypatch.setattr(dataset_mod, "random_crop", fake_crop)
    monkeypatch.setattr(dataset_mod, "random_zoom_out", fake_zoom_out)
    monkeypatch.setattr(dataset_mod, "random_rotate", fake_rotate)
    monkeypatch.setattr(dataset_mod, "random_flip", fake_flip)
    dataset = BaseDataset(
        [str(path)],
        prompts=["box"],
        box_jitter=0.0,
        shape_aug=True,
        scale=(0.5, 0.5),
        size=40,
        mask_size=40,
    )
    item = dataset[0]

    assert calls == [("crop", 0.5), ("rotate", 1.0), ("flip", 1.0)]
    assert item["image"].shape == image.shape
    assert item["target"].shape == image.shape[:2]


def test_base_dataset_shape_aug_zooms_out_when_scale_is_greater_than_one(
    tmp_path,
    monkeypatch,
):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(10, 10, 20, 20),
        roi=np.ones((10, 10), dtype=np.uint8),
    )
    image = np.full((40, 40, 3), 120, dtype=np.uint8)
    path = write_sample(tmp_path / "sample.json", [obj], image=image)
    calls = []

    def fake_crop(
        image: np.ndarray,
        pair: np.ndarray,
        scale: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(("crop", scale))
        return image, pair

    def fake_zoom_out(
        image: np.ndarray,
        pair: np.ndarray,
        scale: float = 1.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(("zoom_out", scale))
        return image, pair

    def fake_rotate(
        image: np.ndarray,
        pair: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(("rotate", 1.0))
        return image, pair

    def fake_flip(
        image: np.ndarray,
        pair: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(("flip", 1.0))
        return image, pair

    monkeypatch.setattr(dataset_mod, "random_crop", fake_crop)
    monkeypatch.setattr(dataset_mod, "random_zoom_out", fake_zoom_out)
    monkeypatch.setattr(dataset_mod, "random_rotate", fake_rotate)
    monkeypatch.setattr(dataset_mod, "random_flip", fake_flip)
    dataset = BaseDataset(
        [str(path)],
        prompts=["box"],
        box_jitter=0.0,
        shape_aug=True,
        scale=(1.5, 1.5),
        size=40,
        mask_size=40,
    )
    item = dataset[0]

    assert calls == [("zoom_out", 1.5), ("rotate", 1.0), ("flip", 1.0)]
    assert item["image"].shape == image.shape
    assert item["target"].shape == image.shape[:2]


def test_base_dataset_shape_aug_rotates_and_flips_when_scale_is_one(
    tmp_path,
    monkeypatch,
):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(10, 10, 20, 20),
        roi=np.ones((10, 10), dtype=np.uint8),
    )
    image = np.full((40, 40, 3), 120, dtype=np.uint8)
    path = write_sample(tmp_path / "sample.json", [obj], image=image)
    calls = []

    def fake_rotate(
        image: np.ndarray,
        pair: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append("rotate")
        return image, pair

    def fake_flip(
        image: np.ndarray,
        pair: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append("flip")
        return image, pair

    monkeypatch.setattr(dataset_mod, "random_rotate", fake_rotate)
    monkeypatch.setattr(dataset_mod, "random_flip", fake_flip)
    dataset = BaseDataset(
        [str(path)],
        prompts=["box"],
        box_jitter=0.0,
        shape_aug=True,
        scale=(1.0, 1.0),
        size=40,
        mask_size=40,
    )
    item = dataset[0]

    assert calls == ["rotate", "flip"]
    assert item["image"].shape == image.shape
    assert item["target"].shape == image.shape[:2]


def test_base_dataset_background_point_uses_augmented_union(tmp_path, monkeypatch):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    def fake_flip(
        image: np.ndarray,
        pair: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        out = np.zeros_like(pair)
        out[1, 1, 0] = 1
        out[..., 1] = 1
        out[0, 0, 1] = 0
        return image, out

    monkeypatch.setattr(dataset_mod, "random_flip", fake_flip)
    dataset = BaseDataset(
        [str(path)],
        prompts=["point"],
        bg_prob=1.0,
        shape_aug=True,
        scale=(1.0, 1.0),
        size=8,
        mask_size=8,
    )

    item = dataset[0]

    assert item["prompt"]["points"].tolist() == [[0.0, 0.0]]
    assert item["target"].sum() == 0
    assert item["mask_valid"] is False
    assert item["is_auto_bg"] is True


def test_random_rotate_keeps_edge_mask_non_empty(monkeypatch):
    image = np.zeros((6, 10, 3), dtype=np.uint8)
    mask = np.zeros((6, 10, 2), dtype=np.uint8)
    mask[4:6, 0:2, 0] = 1
    mask[4:6, 0:2, 1] = 1

    monkeypatch.setattr(image_rotate.np.random, "choice", lambda values: 3)
    out_image, out_mask = image_rotate.random_rotate(image, mask)

    assert out_image.shape == (10, 6, 3)
    assert out_mask.shape == (10, 6, 2)
    assert out_mask[..., 0].sum() > 0


def test_base_dataset_shape_aug_keeps_target_non_empty(tmp_path, monkeypatch):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    def fake_rotate(
        image: np.ndarray,
        pair: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        out = pair.copy()
        out[..., 0] = 0
        return image, out

    monkeypatch.setattr(dataset_mod, "random_rotate", fake_rotate)
    dataset = BaseDataset(
        [str(path)],
        prompts=["box"],
        box_jitter=0.0,
        shape_aug=True,
        scale=(1.0, 1.0),
        size=8,
        mask_size=8,
    )
    item = dataset[0]

    assert item["target"].sum() == 4
    assert item["prompt"]["box"].tolist() == [1.0, 1.0, 3.0, 3.0]
