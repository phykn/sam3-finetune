import numpy as np
from PIL import Image
from src.ops.box import box_area_xyxy, box_iou_xyxy, nms_boxes_xyxy
from src.predict.masks.generator import AutomaticMaskGenerator
from src.predict.masks.geometry import (
    batched,
    build_point_grid,
    calculate_stability_score,
    generate_crop_boxes,
    mask_to_box,
)
from src.predict.masks.proposals import (
    count_proposals_by_crop_grid,
    MaskProposal,
    proposal_mask_image,
    proposal_to_full_mask,
    save_proposal_grid,
    save_proposal_overlay,
)
from src.types import MaskInstance


def test_mask_generator_lives_under_masks_package() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "predict" / "masks" / "generator.py").is_file()
    assert not (root / "src" / "masks").exists()
    assert not (root / "src" / "auto_mask_generator.py").exists()
    assert AutomaticMaskGenerator.__module__ == "src.predict.masks.generator"


def test_mask_helpers_are_split_by_responsibility() -> None:
    assert build_point_grid.__module__ == "src.predict.masks.geometry"
    assert MaskProposal.__module__ == "src.types"


def test_masks_package_exports_user_facing_api() -> None:
    import src.predict.masks as masks

    assert masks.AutomaticMaskGenerator is AutomaticMaskGenerator
    assert masks.MaskInstance is MaskInstance
    assert masks.MaskProposal is MaskProposal
    assert hasattr(masks, "ReferenceExample")
    assert hasattr(masks, "mask_instance_from_proposal")
    assert hasattr(masks, "mask_instances_from_proposals")
    assert not hasattr(masks, "__all__")


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

    assert box_area_xyxy((0, 0, 10, 10)) == 100
    assert box_iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert nms_boxes_xyxy(boxes, scores, iou_threshold=0.6) == [0, 2]


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


def test_proposal_to_full_mask_reconstructs_roi_mask():
    roi_mask = np.array([[True, False], [True, True]], dtype=bool)
    proposal = MaskProposal(
        segmentation=roi_mask,
        bbox=(2, 1, 4, 3),
        area=3,
        predicted_iou=0.9,
        stability_score=1.0,
        point_coords=(2.5, 1.5),
        crop_box=(0, 0, 5, 4),
        image_size=(5, 4),
    )

    full_mask = proposal_to_full_mask(proposal)

    expected = np.zeros((4, 5), dtype=bool)
    expected[1:3, 2:4] = roi_mask
    np.testing.assert_array_equal(full_mask, expected)


def test_proposal_to_full_mask_rejects_missing_image_size():
    proposal = MaskProposal(
        segmentation=np.ones((1, 1), dtype=bool),
        bbox=(0, 0, 1, 1),
        area=1,
        predicted_iou=0.9,
        stability_score=1.0,
        point_coords=(0.5, 0.5),
        crop_box=(0, 0, 1, 1),
    )

    try:
        proposal_to_full_mask(proposal)
    except ValueError as exc:
        assert "image_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_proposal_to_full_mask_rejects_shape_mismatch():
    proposal = MaskProposal(
        segmentation=np.ones((1, 2), dtype=bool),
        bbox=(0, 0, 3, 1),
        area=2,
        predicted_iou=0.9,
        stability_score=1.0,
        point_coords=(0.5, 0.5),
        crop_box=(0, 0, 3, 1),
        image_size=(3, 1),
    )

    try:
        proposal_to_full_mask(proposal)
    except ValueError as exc:
        assert "segmentation shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_proposal_mask_image_returns_roi_alpha_mask():
    proposal = MaskProposal(
        segmentation=np.array([[True, False]], dtype=bool),
        bbox=(1, 2, 3, 3),
        area=1,
        predicted_iou=0.9,
        stability_score=1.0,
        point_coords=(1.5, 2.5),
        crop_box=(0, 0, 4, 4),
        image_size=(4, 4),
    )

    mask_image = proposal_mask_image(proposal, alpha=7)

    assert mask_image.mode == "L"
    assert mask_image.size == (2, 1)
    np.testing.assert_array_equal(
        np.array(mask_image),
        np.array([[7, 0]], dtype=np.uint8),
    )


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
    generator = AutomaticMaskGenerator(
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
    first = proposals[0]
    x0, y0, x1, y1 = first.bbox
    assert first.segmentation.shape == (y1 - y0, x1 - x0)
    assert first.segmentation.sum() == first.area
    assert first.image_size == (8, 8)
    assert proposal_to_full_mask(first).shape == (8, 8)
    assert proposals[0].crop_box == (0, 0, 8, 8)


def test_generator_can_return_mask_instances_with_concept_and_object_ids():
    predictor = FakePredictor()
    generator = AutomaticMaskGenerator(
        predictor,
        points_per_side=2,
        points_per_batch=4,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        box_nms_thresh=1.0,
        max_masks=2,
    )

    instances = generator.generate_instances(
        Image.new("RGB", (8, 8), color=(0, 0, 0)),
        concept_id=5,
        object_id_start=20,
    )

    assert len(instances) == 2
    assert all(isinstance(instance, MaskInstance) for instance in instances)
    assert [instance.concept_id for instance in instances] == [5, 5]
    assert [instance.object_id for instance in instances] == [20, 21]
    assert [instance.source for instance in instances] == ["auto", "auto"]
    assert instances[0].score >= instances[1].score
    assert instances[0].to_full_mask().shape == (8, 8)


def test_generator_filters_by_score_stability_area_and_max_masks():
    predictor = FakePredictor()
    generator = AutomaticMaskGenerator(
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
        self.encoded_batches = []

    def set_image(self, image):
        self.images.append(image.size)

    def encode_image_batch(self, images):
        self.encoded_batches.append([image.size for image in images])
        return [{"image": image, "size": image.size} for image in images]

    def predict_from_embedding(
        self,
        embedding,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        self.images.append(embedding["size"])
        return self.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

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


class BatchDecodeCropAwareFakePredictor(CropAwareFakePredictor):
    def __init__(self):
        super().__init__()
        self.prompt_decode_batches = []

    def predict_from_embedding_batches(
        self,
        prompt_batches,
        multimask_output=True,
        return_logits=False,
    ):
        self.prompt_decode_batches.append(
            [
                (batch.embedding["size"], batch.point_coords.shape[0])
                for batch in prompt_batches
            ]
        )
        results = []
        for batch in prompt_batches:
            self.images.append(batch.embedding["size"])
            results.append(
                self.predict(
                    point_coords=batch.point_coords,
                    point_labels=batch.point_labels,
                    multimask_output=multimask_output,
                    return_logits=return_logits,
                )
            )
        return results


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
        AutomaticMaskGenerator(
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
    generator = AutomaticMaskGenerator(
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
    for proposal in proposals:
        x0, y0, x1, y1 = proposal.bbox
        assert proposal.segmentation.shape == (y1 - y0, x1 - x0)
        assert proposal.segmentation.sum() == proposal.area
        assert proposal.image_size == (8, 8)
        assert proposal_to_full_mask(proposal).shape == (8, 8)
    assert any(proposal.crop_box == (4, 4, 8, 8) for proposal in proposals)
    assert all(0.0 <= proposal.point_coords[0] <= 8.0 for proposal in proposals)
    assert all(0.0 <= proposal.point_coords[1] <= 8.0 for proposal in proposals)


def test_generator_filters_internal_crop_edge_masks():
    predictor = EdgeTouchingFakePredictor()
    generator = AutomaticMaskGenerator(
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


def test_generator_rejects_invalid_crop_encode_batch_size():
    try:
        AutomaticMaskGenerator(CropAwareFakePredictor(), crop_encode_batch_size=0)
    except ValueError as exc:
        assert "crop_encode_batch_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_generator_batches_crop_encoding_without_changing_outputs():
    predictor = CropAwareFakePredictor()
    generator = AutomaticMaskGenerator(
        predictor,
        points_per_batch=8,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        box_nms_thresh=1.0,
        crop_grids=[2],
        crop_points_per_side=[1],
        crop_overlap_ratio=0.0,
        filter_crop_edge_masks=False,
        image_batch_size=2,
    )

    proposals = generator.generate(Image.new("RGB", (8, 8), color=(0, 0, 0)))

    assert predictor.encoded_batches == [[(4, 4), (4, 4)], [(4, 4), (4, 4)]]
    assert len(proposals) == 4
    assert sorted(proposal.crop_index for proposal in proposals) == [0, 1, 2, 3]


def test_generator_batches_prompt_decoding_without_changing_outputs():
    predictor = BatchDecodeCropAwareFakePredictor()
    generator = AutomaticMaskGenerator(
        predictor,
        points_per_batch=2,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        box_nms_thresh=1.0,
        crop_grids=[2],
        crop_points_per_side=[2],
        crop_overlap_ratio=0.0,
        filter_crop_edge_masks=False,
        image_batch_size=2,
        prompt_batch_size=3,
    )

    proposals = generator.generate(Image.new("RGB", (8, 8), color=(0, 0, 0)))

    assert predictor.encoded_batches == [[(4, 4), (4, 4)], [(4, 4), (4, 4)]]
    assert predictor.prompt_decode_batches == [
        [((4, 4), 2), ((4, 4), 2)],
        [((4, 4), 2), ((4, 4), 2)],
        [((4, 4), 2), ((4, 4), 2)],
        [((4, 4), 2), ((4, 4), 2)],
    ]
    assert len(proposals) == 16
    assert sorted({proposal.crop_index for proposal in proposals}) == [0, 1, 2, 3]


def test_generator_can_cross_crop_prompt_decode_when_enabled():
    predictor = BatchDecodeCropAwareFakePredictor()
    generator = AutomaticMaskGenerator(
        predictor,
        points_per_batch=2,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        box_nms_thresh=1.0,
        crop_grids=[2],
        crop_points_per_side=[2],
        crop_overlap_ratio=0.0,
        filter_crop_edge_masks=False,
        image_batch_size=2,
        prompt_batch_size=3,
        allow_cross_crop_prompt_decode=True,
    )

    proposals = generator.generate(Image.new("RGB", (8, 8), color=(0, 0, 0)))

    assert predictor.prompt_decode_batches == [
        [((4, 4), 2), ((4, 4), 2), ((4, 4), 2)],
        [((4, 4), 2)],
        [((4, 4), 2), ((4, 4), 2), ((4, 4), 2)],
        [((4, 4), 2)],
    ]
    assert len(proposals) == 16


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


def test_save_proposal_overlay_accepts_roi_masks(tmp_path):
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    proposal = MaskProposal(
        segmentation=np.ones((2, 2), dtype=bool),
        bbox=(1, 1, 3, 3),
        area=4,
        predicted_iou=1.0,
        stability_score=1.0,
        point_coords=(2.0, 2.0),
        crop_box=(0, 0, 4, 4),
        image_size=(4, 4),
    )
    path = tmp_path / "overlay.png"

    save_proposal_overlay(image, [proposal], path)

    output = Image.open(path).convert("RGBA")
    assert output.size == (4, 4)
    assert output.getpixel((1, 1)) != (0, 0, 0, 255)
    assert output.getpixel((0, 0)) == (0, 0, 0, 255)


def test_save_proposal_grid_accepts_roi_masks(tmp_path):
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    proposal = MaskProposal(
        segmentation=np.ones((2, 2), dtype=bool),
        bbox=(1, 1, 3, 3),
        area=4,
        predicted_iou=1.0,
        stability_score=1.0,
        point_coords=(2.0, 2.0),
        crop_box=(0, 0, 4, 4),
        image_size=(4, 4),
    )
    path = tmp_path / "grid.png"

    save_proposal_grid(image, [proposal], path, max_masks=1, columns=1)

    output = Image.open(path)
    assert output.size[0] == 160
    assert output.size[1] == 160
