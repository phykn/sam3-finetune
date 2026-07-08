import numpy as np

from src.data.sample import (
    Image,
    Object,
    Sample,
    from_json,
    load,
    save,
    to_json,
)


def test_image_keeps_shape_and_optional_id():
    array = np.zeros((4, 6, 3), dtype=np.uint8)

    image = Image(array=array, id="img-1")

    assert image.id == "img-1"
    assert image.shape == (4, 6, 3)
    assert image.array.dtype == np.uint8


def test_object_keeps_roi_and_full_mask_as_uint8():
    roi = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    obj = Object(
        object_id=7,
        class_id=3,
        box=(2, 1, 5, 3),
        roi=roi,
        points=[(3, 2, 1)],
        metrics={"stability": 0.97},
        meta={"source": "manual"},
    )

    mask = obj.mask((4, 6, 3))

    assert mask.shape == (4, 6)
    assert obj.roi.dtype == np.uint8
    assert mask.dtype == np.uint8
    assert np.array_equal(mask[1:3, 2:5], roi)
    assert mask.sum() == roi.sum()
    assert obj.metrics["stability"] == 0.97
    assert obj.meta["source"] == "manual"


def test_sample_groups_image_and_objects():
    image = Image(array=np.zeros((4, 6, 3), dtype=np.uint8))
    obj = Object(
        object_id=1,
        class_id=None,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )

    sample = Sample(image=image, objects=[obj])

    assert sample.image.shape == (4, 6, 3)
    assert sample.objects[0].object_id == 1


def test_sample_json_round_trip_embeds_image_and_roi(tmp_path):
    image = Image(
        id="img-1",
        array=np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3),
    )
    roi = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
        ],
        dtype=np.uint8,
    )
    obj = Object(
        object_id=7,
        class_id=3,
        box=(2, 1, 5, 3),
        roi=roi,
        points=[(3, 2, 1)],
        metrics={"stability": 0.97},
        meta={"source": "manual"},
    )
    sample = Sample(image=image, objects=[obj])

    data = to_json(sample)
    loaded = from_json(data)

    assert data["schema_version"] == "sam3.sample.v1"
    assert data["image"]["format"] == "png"
    assert data["image"]["encoding"] == "base64"
    assert isinstance(data["image"]["data"], str)
    assert data["objects"][0]["roi"]["encoding"] == "rle"
    assert loaded.image.id == "img-1"
    assert loaded.image.shape == (4, 6, 3)
    assert np.array_equal(loaded.image.array, image.array)
    assert loaded.objects[0].box == (2, 1, 5, 3)
    assert np.array_equal(loaded.objects[0].roi, roi)
    assert loaded.objects[0].points == [[3, 2, 1]]
    assert loaded.objects[0].metrics["stability"] == 0.97
    assert loaded.objects[0].meta["source"] == "manual"

    path = tmp_path / "sample.json"
    save(sample, path)
    from_file = load(path)

    assert '\n  "schema_version"' in path.read_text(encoding="utf-8")
    assert np.array_equal(from_file.image.array, image.array)
    assert np.array_equal(from_file.objects[0].roi, roi)
