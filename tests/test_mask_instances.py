import numpy as np
from src.predict.masks.proposals import MaskProposal
from src.predict.masks.types import (
    mask_instance_from_proposal,
    mask_instances_from_proposals,
    MaskInstance,
)
from src.predict.reference.types import ReferenceExample


def _proposal(
    bbox: tuple[int, int, int, int] = (2, 1, 5, 3),
    predicted_iou: float = 0.83,
) -> MaskProposal:
    mask = np.ones((bbox[3] - bbox[1], bbox[2] - bbox[0]), dtype=bool)
    return MaskProposal(
        segmentation=mask,
        bbox=bbox,
        area=int(mask.sum()),
        predicted_iou=predicted_iou,
        stability_score=0.91,
        point_coords=(3.5, 2.0),
        crop_box=(0, 0, 8, 6),
        crop_grid=2,
        crop_index=1,
        image_size=(8, 6),
    )


def test_mask_instance_from_proposal_preserves_roi_metadata_and_ids():
    proposal = _proposal()

    instance = mask_instance_from_proposal(
        proposal,
        concept_id=2,
        object_id=7,
        source="auto",
    )

    assert isinstance(instance, MaskInstance)
    assert instance.segmentation.shape == (2, 3)
    assert instance.bbox == (2, 1, 5, 3)
    assert instance.area == 6
    assert instance.score == 0.83
    assert instance.predicted_iou == 0.83
    assert instance.stability_score == 0.91
    assert instance.point_coords == (3.5, 2.0)
    assert instance.crop_box == (0, 0, 8, 6)
    assert instance.crop_grid == 2
    assert instance.crop_index == 1
    assert instance.image_size == (8, 6)
    assert instance.concept_id == 2
    assert instance.object_id == 7
    assert instance.source == "auto"

    full_mask = instance.to_full_mask()
    assert full_mask.shape == (6, 8)
    assert int(full_mask.sum()) == 6
    np.testing.assert_array_equal(full_mask[1:3, 2:5], proposal.segmentation)


def test_mask_instances_from_proposals_assigns_sequential_object_ids():
    proposals = [_proposal(predicted_iou=0.9), _proposal(predicted_iou=0.7)]

    instances = mask_instances_from_proposals(
        proposals,
        concept_id=4,
        object_id_start=10,
    )

    assert [instance.concept_id for instance in instances] == [4, 4]
    assert [instance.object_id for instance in instances] == [10, 11]
    assert [instance.score for instance in instances] == [0.9, 0.7]


def test_reference_example_derives_bbox_and_area_from_mask():
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 2:6] = 1

    reference = ReferenceExample(concept_id=3, mask=mask, object_id=1)

    assert reference.mask.dtype == bool
    assert reference.box_xyxy == (2, 1, 6, 4)
    assert reference.area == 12
    assert reference.concept_id == 3
    assert reference.object_id == 1
    assert reference.weight == 1.0


def test_reference_example_rejects_invalid_ids_and_weight():
    mask = np.ones((2, 2), dtype=bool)

    for kwargs in (
        {"concept_id": -1, "mask": mask},
        {"concept_id": 0, "mask": mask, "object_id": -1},
        {"concept_id": 0, "mask": mask, "weight": 0.0},
    ):
        try:
            ReferenceExample(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")
