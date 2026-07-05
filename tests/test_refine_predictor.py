import numpy as np
import torch
from src.types import ContextPrediction, MaskInstance, Sam3ImageEmbedding


def _instance(score: float, *, bbox=(1, 1, 3, 3)) -> MaskInstance:
    x0, y0, x1, y1 = bbox
    return MaskInstance(
        segmentation=np.ones((y1 - y0, x1 - x0), dtype=bool),
        bbox=bbox,
        area=(y1 - y0) * (x1 - x0),
        score=score,
        predicted_iou=score,
        point_coords=(float(x0 + 1), float(y0 + 1)),
        crop_box=(0, 0, 8, 8),
        crop_grid=1,
        crop_index=0,
        image_size=(8, 8),
    )


def _prediction(score: float) -> ContextPrediction:
    return ContextPrediction(
        segmentation=np.ones((2, 2), dtype=bool),
        bbox=(2, 2, 4, 4),
        area=4,
        point_coords=(3.0, 3.0),
        context_score=score,
        predicted_iou=0.8,
        stability_score=0.7,
        score=score,
        image_size=(8, 8),
    )


def test_context_grid_refiner_sends_all_grid_masks_as_same_image_context():
    from src.predict.refine import ContextGridRefiner

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    base_instances = [
        _instance(0.4, bbox=(1, 1, 3, 3)),
        _instance(0.8, bbox=(4, 4, 7, 7)),
    ]
    refined_predictions = [_prediction(1.1)]

    class FakeBaseGenerator:
        def __init__(self) -> None:
            self.image = None

        def generate_instances(self, target_image):
            self.image = target_image
            return base_instances

    class FakeMatcher:
        def __init__(self) -> None:
            self.calls = []

        def predict(self, target_image, references, *, max_masks=None):
            self.calls.append(
                {
                    "target_image": target_image,
                    "references": references,
                    "max_masks": max_masks,
                }
            )
            return refined_predictions

    base_generator = FakeBaseGenerator()
    matcher = FakeMatcher()
    refiner = ContextGridRefiner(
        base_generator=base_generator,
        matcher=matcher,
    )

    result = refiner.refine(image, max_masks=1)

    assert base_generator.image is image
    assert matcher.calls[0]["target_image"] is image
    assert matcher.calls[0]["max_masks"] == 1
    references = matcher.calls[0]["references"]
    assert len(references) == 2
    for reference, instance in zip(references, base_instances):
        assert reference.image is image
        assert reference.weight == 1.0
        np.testing.assert_array_equal(reference.mask, instance.to_full_mask())
    assert result.base_instances == base_instances
    assert result.context_references == references
    assert result.refined_predictions == refined_predictions


def test_context_grid_refiner_returns_empty_when_grid_finds_no_context_masks():
    from src.predict.refine import ContextGridRefiner

    class FakeBaseGenerator:
        def generate_instances(self, target_image):
            return []

    class FakeMatcher:
        def __init__(self) -> None:
            self.called = False

        def predict(self, target_image, references, *, max_masks=None):
            self.called = True
            return []

    matcher = FakeMatcher()
    refiner = ContextGridRefiner(
        base_generator=FakeBaseGenerator(),
        matcher=matcher,
    )

    result = refiner.refine(np.zeros((8, 8, 3), dtype=np.uint8))

    assert matcher.called is False
    assert result.base_instances == []
    assert result.context_references == []
    assert result.refined_predictions == []


def test_refine_package_exports_context_grid_refiner_api():
    import src.predict.refine as refine
    from src.predict.refine.grid import ContextGridRefiner, ContextGridRefineResult

    assert refine.ContextGridRefiner is ContextGridRefiner
    assert refine.ContextGridRefineResult is ContextGridRefineResult
    assert hasattr(refine, "MaskRefiner")
    assert not hasattr(refine, "__all__")


def test_mask_refiner_accepts_explicit_embedding_without_set_image():
    from src.predict.refine import MaskRefiner

    class FakePredictor:
        def __init__(self) -> None:
            self.embedding = None

        def predict(self, **_kwargs):
            raise AssertionError("explicit embedding should bypass predict")

        def predict_from_embedding(self, embedding, **kwargs):
            self.embedding = embedding
            self.kwargs = kwargs
            masks = np.zeros((1, 1, 8, 8), dtype=bool)
            masks[0, 0, 2:6, 2:6] = True
            scores = np.array([[0.9]], dtype=np.float32)
            low_res = np.zeros((1, 1, 4, 4), dtype=np.float32)
            return masks, scores, low_res

    fake = FakePredictor()
    refiner = MaskRefiner(fake)
    embedding = Sam3ImageEmbedding(
        image_embed=torch.zeros(1, 1, 1, 1),
        high_res_features=(),
        orig_hw=(8, 8),
    )

    result = refiner.refine(
        embedding=embedding,
        mask_input=np.zeros((4, 4), dtype=np.float32),
    )

    assert fake.embedding is embedding
    assert fake.kwargs["multimask_output"] is False
    np.testing.assert_allclose(result.score, 0.9)
    assert result.mask.sum() == 16


def test_mask_refiner_refine_embedding_delegates_to_refine():
    from src.predict.refine import MaskRefiner

    class FakePredictor:
        def predict_from_embedding(self, embedding, **kwargs):
            self.embedding = embedding
            self.kwargs = kwargs
            masks = np.zeros((1, 1, 8, 8), dtype=bool)
            masks[0, 0, 1:4, 2:6] = True
            scores = np.array([[0.8]], dtype=np.float32)
            low_res = np.zeros((1, 1, 4, 4), dtype=np.float32)
            return masks, scores, low_res

    fake = FakePredictor()
    refiner = MaskRefiner(fake)
    embedding = Sam3ImageEmbedding(
        image_embed=torch.zeros(1, 1, 1, 1),
        high_res_features=(),
        orig_hw=(8, 8),
    )

    result = refiner.refine_embedding(
        embedding,
        box=np.array([1, 1, 6, 5], dtype=np.float32),
        mask_input=np.zeros((4, 4), dtype=np.float32),
    )

    assert fake.embedding is embedding
    np.testing.assert_array_equal(
        fake.kwargs["box"],
        np.array([1, 1, 6, 5], dtype=np.float32),
    )
    assert fake.kwargs["multimask_output"] is False
    np.testing.assert_allclose(result.score, 0.8)
    assert result.mask.sum() == 12


def test_mask_refiner_refine_image_encodes_image_then_refines_embedding():
    from src.predict.refine import MaskRefiner

    class FakePredictor:
        def __init__(self) -> None:
            self.embedding = Sam3ImageEmbedding(
                image_embed=torch.zeros(1, 1, 1, 1),
                high_res_features=(),
                orig_hw=(8, 8),
            )
            self.encoded_image = None

        def encode_image(self, image):
            self.encoded_image = image
            return self.embedding

        def predict_from_embedding(self, embedding, **kwargs):
            self.refined_embedding = embedding
            self.kwargs = kwargs
            masks = np.zeros((1, 1, 8, 8), dtype=bool)
            masks[0, 0, 2:5, 1:5] = True
            scores = np.array([[0.7]], dtype=np.float32)
            low_res = np.zeros((1, 1, 4, 4), dtype=np.float32)
            return masks, scores, low_res

    fake = FakePredictor()
    refiner = MaskRefiner(fake)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    result = refiner.refine_image(
        image,
        point_coords=np.array([[3, 4]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int64),
        mask_input=np.zeros((4, 4), dtype=np.float32),
    )

    assert fake.encoded_image is image
    assert fake.refined_embedding is fake.embedding
    np.testing.assert_array_equal(
        fake.kwargs["point_coords"],
        np.array([[3, 4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        fake.kwargs["point_labels"],
        np.array([1], dtype=np.int64),
    )
    assert fake.kwargs["multimask_output"] is False
    np.testing.assert_allclose(result.score, 0.7)
    assert result.mask.sum() == 12
