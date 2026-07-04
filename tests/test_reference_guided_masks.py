import numpy as np
import torch

from src.data.prediction import Sam3ImageEmbedding
from src.masks.types import MaskInstance, ReferenceExample


def _embedding_from_feature_map(feature_map: torch.Tensor) -> Sam3ImageEmbedding:
    return Sam3ImageEmbedding(
        image_embed=feature_map.unsqueeze(0),
        high_res_features=(),
        orig_hw=(40, 40),
    )


class FakeReferencePredictor:
    def encode_image_batch(self, images):
        return [_embedding_from_feature_map(image) for image in images]


def _candidate(
    bbox: tuple[int, int, int, int],
    score: float,
    concept_id: int | None = None,
    object_id: int | None = None,
) -> MaskInstance:
    x0, y0, x1, y1 = bbox
    mask = np.ones((y1 - y0, x1 - x0), dtype=bool)
    return MaskInstance(
        segmentation=mask,
        bbox=bbox,
        area=int(mask.sum()),
        score=score,
        source="auto",
        concept_id=concept_id,
        object_id=object_id,
        predicted_iou=score,
        image_size=(40, 40),
    )


def test_reference_guided_rerank_prefers_candidates_matching_reference_features():
    from src.context.reference_guided import ReferenceGuidedMaskGenerator

    reference_features = torch.zeros(2, 4, 4, dtype=torch.float32)
    reference_features[0, 1:3, 1:3] = 3.0
    target_features = torch.zeros(2, 4, 4, dtype=torch.float32)
    target_features[0, 1, 2] = 3.0
    target_features[1, 3, 0] = 3.0

    reference_mask = np.zeros((40, 40), dtype=bool)
    reference_mask[10:30, 10:30] = True
    references = [
        ReferenceExample(
            concept_id=6,
            object_id=0,
            image=reference_features,
            mask=reference_mask,
        )
    ]
    candidates = [
        _candidate((0, 30, 10, 40), score=0.95, object_id=10),
        _candidate((20, 10, 30, 20), score=0.10, object_id=11),
    ]
    generator = ReferenceGuidedMaskGenerator(
        FakeReferencePredictor(),
        base_score_weight=0.0,
        negative_context_mode="none",
    )

    ranked = generator.rerank(
        target_features,
        candidates,
        references,
    )

    assert [instance.bbox for instance in ranked] == [
        (20, 10, 30, 20),
        (0, 30, 10, 40),
    ]
    assert ranked[0].concept_id == 6
    assert ranked[0].object_id == 11
    assert ranked[0].source == "reference_guided"
    assert ranked[0].context_score > ranked[1].context_score
    assert ranked[0].base_score == 0.10


def test_reference_guided_generator_wraps_base_automatic_generator():
    from src.context.reference_guided import ReferenceGuidedMaskGenerator

    class FakeBaseGenerator:
        def __init__(self) -> None:
            self.image = None

        def generate_instances(self, image):
            self.image = image
            return [
                _candidate((20, 10, 30, 20), score=0.10),
                _candidate((0, 30, 10, 40), score=0.95),
            ]

    reference_features = torch.zeros(2, 4, 4, dtype=torch.float32)
    reference_features[0, 1:3, 1:3] = 3.0
    target_features = torch.zeros(2, 4, 4, dtype=torch.float32)
    target_features[0, 1, 2] = 3.0
    reference_mask = np.zeros((40, 40), dtype=bool)
    reference_mask[10:30, 10:30] = True
    base_generator = FakeBaseGenerator()
    generator = ReferenceGuidedMaskGenerator(
        FakeReferencePredictor(),
        base_generator=base_generator,
        base_score_weight=0.0,
        negative_context_mode="none",
    )

    ranked = generator.generate(
        target_features,
        [
            ReferenceExample(
                concept_id=2,
                image=reference_features,
                mask=reference_mask,
            )
        ],
        max_masks=1,
    )

    assert base_generator.image is target_features
    assert len(ranked) == 1
    assert ranked[0].bbox == (20, 10, 30, 20)
    assert ranked[0].concept_id == 2


def test_reference_guided_rerank_rejects_mixed_reference_concepts():
    from src.context.reference_guided import ReferenceGuidedMaskGenerator

    mask = np.ones((40, 40), dtype=bool)
    features = torch.zeros(2, 4, 4, dtype=torch.float32)
    generator = ReferenceGuidedMaskGenerator(FakeReferencePredictor())

    try:
        generator.rerank(
            features,
            [_candidate((0, 0, 10, 10), score=1.0)],
            [
                ReferenceExample(concept_id=0, image=features, mask=mask),
                ReferenceExample(concept_id=1, image=features, mask=mask),
            ],
        )
    except ValueError as exc:
        assert "same concept_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
