import numpy as np
from PIL import Image

from src.auto_mask_generator import (
    MaskProposal,
    Sam3AutomaticMaskGenerator,
    batched,
    box_area,
    box_iou,
    build_point_grid,
    calculate_stability_score,
    count_proposals_by_crop_grid,
    generate_crop_boxes,
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


def test_generate_crop_boxes_full_image_grid():
    crops = generate_crop_boxes(100, 80, grid_size=1, overlap_ratio=0.25)

    assert crops == [(0, 0, 100, 80)]


def test_generate_crop_boxes_two_by_two_with_overlap_cover_edges():
    crops = generate_crop_boxes(100, 80, grid_size=2, overlap_ratio=0.25)

    assert len(crops) == 4
    assert crops[0] == (0, 0, 56, 45)
    assert crops[-1] == (44, 35, 100, 80)
    assert min(crop[0] for crop in crops) == 0
    assert min(crop[1] for crop in crops) == 0
    assert max(crop[2] for crop in crops) == 100
    assert max(crop[3] for crop in crops) == 80
    assert all(x0 < x1 and y0 < y1 for x0, y0, x1, y1 in crops)


def test_generate_crop_boxes_rejects_invalid_config():
    for grid_size in (0, -1):
        try:
            generate_crop_boxes(100, 80, grid_size=grid_size, overlap_ratio=0.25)
        except ValueError as exc:
            assert "grid_size" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    for overlap_ratio in (-0.1, 0.5):
        try:
            generate_crop_boxes(100, 80, grid_size=2, overlap_ratio=overlap_ratio)
        except ValueError as exc:
            assert "overlap_ratio" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


def test_mask_proposal_defaults_crop_metadata():
    proposal = MaskProposal(
        segmentation=np.zeros((4, 4), dtype=bool),
        bbox=(0, 0, 1, 1),
        area=1,
        predicted_iou=0.9,
        stability_score=1.0,
        point_coords=(0.5, 0.5),
        crop_box=(0, 0, 4, 4),
    )

    assert proposal.crop_grid == 1
    assert proposal.crop_index == 0


class FakePredictor:
    def __init__(self):
        self.set_image_calls = 0
        self.predict_batches = []

    def set_image(self, image):
        self.set_image_calls += 1

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        self.predict_batches.append((point_coords.copy(), point_labels.copy()))
        batch = point_coords.shape[0]
        masks = np.zeros((batch, 1, 8, 8), dtype=bool)
        low_res = np.zeros((batch, 1, 8, 8), dtype=np.float32)
        scores = np.zeros((batch, 1), dtype=np.float32)
        for i in range(batch):
            x = min(int(point_coords[i, 0, 0] // 4), 6)
            y = min(int(point_coords[i, 0, 1] // 4), 6)
            masks[i, 0, y : y + 2, x : x + 2] = True
            low_res[i, 0] = np.where(masks[i, 0], 2.0, -2.0)
            scores[i, 0] = 1.0 - (i * 0.01)
        return masks, scores, low_res


def test_generator_batches_grid_points_and_returns_sorted_proposals():
    predictor = FakePredictor()
    generator = Sam3AutomaticMaskGenerator(
        predictor,
        points_per_side=2,
        points_per_batch=3,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        box_nms_thresh=1.0,
    )

    proposals = generator.generate(Image.new("RGB", (8, 8), color=(0, 0, 0)))

    assert predictor.set_image_calls == 1
    assert [batch[0].shape[0] for batch in predictor.predict_batches] == [3, 1]
    assert len(proposals) == 4
    assert proposals[0].predicted_iou >= proposals[-1].predicted_iou
    assert proposals[0].segmentation.shape == (8, 8)
    assert proposals[0].crop_box == (0, 0, 8, 8)


def test_generator_filters_by_score_stability_area_and_max_masks():
    predictor = FakePredictor()
    generator = Sam3AutomaticMaskGenerator(
        predictor,
        points_per_side=2,
        points_per_batch=4,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.5,
        min_mask_region_area=4,
        box_nms_thresh=1.0,
        max_masks=2,
    )

    proposals = generator.generate(np.zeros((8, 8, 3), dtype=np.uint8))

    assert len(proposals) == 2
    assert all(proposal.area >= 4 for proposal in proposals)


class CropAwareFakePredictor:
    def __init__(self):
        self.images = []
        self.predict_batches = []

    def set_image(self, image):
        self.images.append(image.size)

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        self.predict_batches.append((point_coords.copy(), point_labels.copy()))
        batch = point_coords.shape[0]
        crop_w, crop_h = self.images[-1]
        masks = np.zeros((batch, 1, crop_h, crop_w), dtype=bool)
        low_res = np.zeros((batch, 1, crop_h, crop_w), dtype=np.float32)
        scores = np.ones((batch, 1), dtype=np.float32)
        for i in range(batch):
            x = min(max(int(point_coords[i, 0, 0]), 0), crop_w - 1)
            y = min(max(int(point_coords[i, 0, 1]), 0), crop_h - 1)
            x0 = max(x - 1, 0)
            y0 = max(y - 1, 0)
            x1 = min(x0 + 2, crop_w)
            y1 = min(y0 + 2, crop_h)
            masks[i, 0, y0:y1, x0:x1] = True
            low_res[i, 0] = np.where(masks[i, 0], 2.0, -2.0)
            scores[i, 0] = 1.0 - (i * 0.01)
        return masks, scores, low_res


class EdgeTouchingFakePredictor:
    def __init__(self):
        self.images = []

    def set_image(self, image):
        self.images.append(image.size)

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        batch = point_coords.shape[0]
        crop_w, crop_h = self.images[-1]
        masks = np.ones((batch, 1, crop_h, crop_w), dtype=bool)
        low_res = np.full((batch, 1, crop_h, crop_w), 2.0, dtype=np.float32)
        scores = np.ones((batch, 1), dtype=np.float32)
        return masks, scores, low_res


def test_generator_rejects_mismatched_crop_lists():
    try:
        Sam3AutomaticMaskGenerator(
            CropAwareFakePredictor(),
            crop_grids=[1, 2],
            crop_points_per_side=[4],
        )
    except ValueError as exc:
        assert "crop_grids" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_generator_runs_explicit_crop_grids_and_maps_to_full_image():
    predictor = CropAwareFakePredictor()
    generator = Sam3AutomaticMaskGenerator(
        predictor,
        points_per_batch=8,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        box_nms_thresh=1.0,
        crop_grids=[1, 2],
        crop_points_per_side=[1, 1],
        crop_overlap_ratio=0.0,
        filter_crop_edge_masks=False,
    )

    proposals = generator.generate(Image.new("RGB", (8, 8), color=(0, 0, 0)))

    assert predictor.images == [(8, 8), (4, 4), (4, 4), (4, 4), (4, 4)]
    assert len(proposals) == 5
    assert {proposal.crop_grid for proposal in proposals} == {1, 2}
    assert sorted(
        proposal.crop_index for proposal in proposals if proposal.crop_grid == 2
    ) == [0, 1, 2, 3]
    assert all(proposal.segmentation.shape == (8, 8) for proposal in proposals)
    assert any(proposal.crop_box == (4, 4, 8, 8) for proposal in proposals)
    assert all(0.0 <= proposal.point_coords[0] <= 8.0 for proposal in proposals)
    assert all(0.0 <= proposal.point_coords[1] <= 8.0 for proposal in proposals)


def test_generator_filters_internal_crop_edge_masks():
    predictor = EdgeTouchingFakePredictor()
    generator = Sam3AutomaticMaskGenerator(
        predictor,
        points_per_batch=8,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        box_nms_thresh=1.0,
        crop_grids=[2],
        crop_points_per_side=[1],
        crop_overlap_ratio=0.0,
        filter_crop_edge_masks=True,
    )

    proposals = generator.generate(Image.new("RGB", (8, 8), color=(0, 0, 0)))

    assert proposals == []


def test_count_proposals_by_crop_grid():
    proposals = [
        MaskProposal(
            segmentation=np.zeros((2, 2), dtype=bool),
            bbox=(0, 0, 1, 1),
            area=1,
            predicted_iou=1.0,
            stability_score=1.0,
            point_coords=(0.5, 0.5),
            crop_box=(0, 0, 2, 2),
            crop_grid=1,
        ),
        MaskProposal(
            segmentation=np.zeros((2, 2), dtype=bool),
            bbox=(0, 0, 1, 1),
            area=1,
            predicted_iou=1.0,
            stability_score=1.0,
            point_coords=(0.5, 0.5),
            crop_box=(0, 0, 1, 1),
            crop_grid=2,
        ),
        MaskProposal(
            segmentation=np.zeros((2, 2), dtype=bool),
            bbox=(1, 1, 2, 2),
            area=1,
            predicted_iou=1.0,
            stability_score=1.0,
            point_coords=(1.5, 1.5),
            crop_box=(1, 1, 2, 2),
            crop_grid=2,
        ),
    ]

    assert count_proposals_by_crop_grid(proposals) == {1: 1, 2: 2}
