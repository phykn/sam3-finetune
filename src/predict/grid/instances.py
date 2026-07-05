from collections.abc import Sequence

from ...types import MaskInstance, MaskProposal


def mask_instance_from_proposal(
    proposal: MaskProposal,
    *,
    concept_id: int | None = None,
    object_id: int | None = None,
    source: str = "auto",
) -> MaskInstance:
    return MaskInstance(
        segmentation=proposal.segmentation,
        bbox=proposal.bbox,
        area=proposal.area,
        score=float(proposal.predicted_iou),
        source=source,
        concept_id=concept_id,
        object_id=object_id,
        predicted_iou=proposal.predicted_iou,
        stability_score=proposal.stability_score,
        point_coords=proposal.point_coords,
        crop_box=proposal.crop_box,
        crop_grid=proposal.crop_grid,
        crop_index=proposal.crop_index,
        image_size=proposal.image_size,
    )


def mask_instances_from_proposals(
    proposals: Sequence[MaskProposal],
    *,
    concept_id: int | None = None,
    object_id_start: int | None = None,
    source: str = "auto",
) -> list[MaskInstance]:
    _validate_optional_id(concept_id, "concept_id")
    _validate_optional_id(object_id_start, "object_id_start")
    instances: list[MaskInstance] = []
    for offset, proposal in enumerate(proposals):
        object_id = None if object_id_start is None else object_id_start + offset
        instances.append(
            mask_instance_from_proposal(
                proposal,
                concept_id=concept_id,
                object_id=object_id,
                source=source,
            )
        )
    return instances


def _validate_optional_id(value: int | None, name: str) -> None:
    if value is not None and int(value) < 0:
        raise ValueError(f"{name} must be non-negative")
