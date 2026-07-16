import numpy as np
import pytest

from src.data import pack
from src.predict.mask.component import largest
from src.predict.mask.postprocess import (
    drop_edge,
    make_exclusive,
    merge_overlap,
    sort_area,
)


def item(score, box, roi):
    return {
        "object_id": int(score),
        "class_id": None,
        "box": box,
        "roi": np.asarray(roi, dtype=bool),
        "metrics": {"score": float(score)},
    }


def test_largest_keeps_only_largest_eight_connected_component():
    mask = np.zeros((2, 1, 6, 7), dtype=bool)
    mask[0, 0, 1, 1] = True
    mask[0, 0, 2, 2] = True
    mask[0, 0, 4, 5] = True

    out = largest(mask)

    expected = np.zeros_like(mask)
    expected[0, 0, 1, 1] = True
    expected[0, 0, 2, 2] = True
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.bool_


def test_largest_rejects_values_without_image_axes():
    with pytest.raises(ValueError, match="two dimensions"):
        largest(np.ones(3, dtype=bool))


def test_sort_area_supports_ascending_and_descending_order():
    items = [
        item(1, (0, 0, 3, 1), np.ones((1, 3))),
        item(2, (0, 0, 2, 3), np.ones((3, 2))),
        item(3, (0, 0, 1, 1), np.ones((1, 1))),
    ]

    asc = sort_area(items, "asc")
    desc = sort_area(items, "desc")

    assert [value["metrics"]["score"] for value in asc] == [3.0, 1.0, 2.0]
    assert [value["metrics"]["score"] for value in desc] == [2.0, 1.0, 3.0]
    assert [value["object_id"] for value in asc] == [1, 2, 3]
    assert [value["object_id"] for value in desc] == [1, 2, 3]
    assert [value["object_id"] for value in items] == [1, 2, 3]


def test_sort_area_rejects_unknown_order():
    with pytest.raises(ValueError, match="asc"):
        sort_area([], "largest")


def test_make_exclusive_keeps_small_masks_and_cuts_large_masks():
    large = item(1, (0, 0, 4, 4), np.ones((4, 4)))
    small = item(2, (1, 1, 3, 3), np.ones((2, 2)))

    out = make_exclusive([large, small], min_ratio=0.7)

    masks = [pack.full((4, 4), value["box"], value["roi"]) for value in out]
    assert [int(mask.sum()) for mask in masks] == [12, 4]
    assert not (masks[0] & masks[1]).any()
    assert [value["object_id"] for value in out] == [1, 2]
    assert large["roi"].all()


def test_make_exclusive_drops_masks_below_remaining_ratio():
    large = item(1, (0, 0, 4, 4), np.ones((4, 4)))
    small = item(2, (1, 1, 3, 3), np.ones((2, 2)))

    out = make_exclusive([large, small], min_ratio=0.8)

    assert len(out) == 1
    assert out[0]["metrics"]["score"] == 2.0
    assert out[0]["object_id"] == 1


def test_merge_overlap_uses_fraction_of_the_smaller_mask():
    small = item(1, (3, 1, 5, 3), np.ones((2, 2)))
    large = item(2, (0, 0, 4, 4), np.ones((4, 4)))

    merged = merge_overlap([small, large], min_overlap=0.5)
    separate = merge_overlap([small, large], min_overlap=0.51)

    assert len(merged) == 1
    assert merged[0]["box"] == (0, 0, 5, 4)
    assert int(merged[0]["roi"].sum()) == 18
    assert merged[0]["metrics"]["score"] == 2.0
    assert len(separate) == 2
    assert [value["object_id"] for value in separate] == [1, 2]


def test_merge_overlap_repeats_for_nested_masks():
    large = item(1, (0, 0, 6, 6), np.ones((6, 6)))
    middle = item(2, (1, 1, 5, 5), np.ones((4, 4)))
    small = item(3, (2, 2, 4, 4), np.ones((2, 2)))

    out = merge_overlap([small, middle, large], min_overlap=1.0)

    assert len(out) == 1
    assert out[0]["box"] == (0, 0, 6, 6)
    assert out[0]["metrics"]["score"] == 1.0


def test_overlap_uses_mask_pixels_instead_of_box_intersection():
    first = item(1, (0, 0, 2, 2), [[1, 0], [0, 0]])
    second = item(2, (0, 0, 2, 2), [[0, 0], [0, 1]])

    exclusive = make_exclusive([first, second], min_ratio=1.0)
    merged = merge_overlap([first, second], min_overlap=0.0)

    assert len(exclusive) == 2
    assert [int(value["roi"].sum()) for value in exclusive] == [1, 1]
    assert len(merged) == 2


def test_drop_edge_removes_masks_touching_any_image_side():
    items = [
        item(1, (1, 0, 3, 2), np.ones((2, 2))),
        item(2, (1, 3, 3, 5), np.ones((2, 2))),
        item(3, (0, 1, 2, 3), np.ones((2, 2))),
        item(4, (4, 1, 6, 3), np.ones((2, 2))),
        item(5, (2, 1, 4, 3), np.ones((2, 2))),
    ]

    out = drop_edge(items, image_shape=(5, 6, 3))

    assert len(out) == 1
    assert out[0]["metrics"]["score"] == 5.0
    assert out[0]["object_id"] == 1


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_overlap_functions_reject_invalid_ratios(value):
    with pytest.raises(ValueError):
        make_exclusive([], value)
    with pytest.raises(ValueError):
        merge_overlap([], value)
