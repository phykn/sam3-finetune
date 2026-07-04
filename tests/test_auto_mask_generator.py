import numpy as np

from src.auto_mask_generator import (
    batched,
    box_area,
    box_iou,
    build_point_grid,
    calculate_stability_score,
    mask_to_box,
    nms_boxes,
)


def test_build_point_grid_centers_points_inside_unit_cells():
    grid = build_point_grid(2)

    assert grid.shape == (4, 2)
    np.testing.assert_allclose(
        grid,
        np.array(
            [
                [0.25, 0.25],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.75, 0.75],
            ],
            dtype=np.float32,
        ),
    )


def test_build_point_grid_rejects_invalid_size():
    try:
        build_point_grid(0)
    except ValueError as exc:
        assert "points_per_side" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_mask_to_box_returns_inclusive_exclusive_xyxy():
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:5, 3:7] = True

    assert mask_to_box(mask) == (3, 2, 7, 5)
    assert mask_to_box(np.zeros((3, 4), dtype=bool)) is None


def test_calculate_stability_score_uses_offset_thresholds():
    logits = np.array([[-2.0, -0.5, 0.5, 2.0]], dtype=np.float32)

    score = calculate_stability_score(logits, mask_threshold=0.0, offset=1.0)

    assert score == 1.0 / 3.0


def test_box_iou_and_nms_boxes_remove_lower_scoring_duplicate():
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [20, 20, 30, 30],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    assert box_area((0, 0, 10, 10)) == 100
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert nms_boxes(boxes, scores, iou_threshold=0.6) == [0, 2]


def test_batched_splits_sequence_without_dropping_items():
    chunks = list(batched(np.arange(5), 2))

    assert [chunk.tolist() for chunk in chunks] == [[0, 1], [2, 3], [4]]
