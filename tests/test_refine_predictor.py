import numpy as np
from src.types import MaskInstance


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


def test_grid_mask_refiner_uses_every_grid_mask_as_mask_prompt():
    from src.predict.refine import GridMaskRefiner

    target = np.zeros((8, 8, 3), dtype=np.uint8)
    base_instances = [
        _instance(0.4, bbox=(1, 1, 3, 3)),
        _instance(0.8, bbox=(4, 4, 7, 7)),
    ]

    class FakeBaseGenerator:
        def __init__(self) -> None:
            self.target_image = None

        def generate_instances(self, target_image):
            self.target_image = target_image
            return base_instances

    class FakePredictor:
        def __init__(self) -> None:
            self.mask_inputs = []

        def encode_image(self, image):
            return object()

        def predict_from_embedding(self, embedding, **kwargs):
            mask_input = kwargs["mask_input"]
            self.mask_inputs.append(mask_input)
            batch_size = mask_input.shape[0]
            masks = np.zeros((batch_size, 1, 8, 8), dtype=bool)
            scores = np.zeros((batch_size, 1), dtype=np.float32)
            low_res = np.zeros((batch_size, 1, 4, 4), dtype=np.float32)
            for index in range(batch_size):
                masks[index, 0] = mask_input[index] > 0
                scores[index, 0] = 0.9 - index * 0.1
                low_res[index, 0] = 1.0
            return masks, scores, low_res

    base_generator = FakeBaseGenerator()
    predictor = FakePredictor()
    refiner = GridMaskRefiner(
        predictor=predictor,
        base_generator=base_generator,
        batch_size=4,
        mask_foreground=4.0,
        mask_background=-4.0,
    )

    result = refiner.refine(target)

    assert base_generator.target_image is target
    assert len(predictor.mask_inputs) == 1
    assert predictor.mask_inputs[0].shape == (2, 8, 8)
    np.testing.assert_array_equal(
        predictor.mask_inputs[0][0],
        np.where(base_instances[0].to_full_mask(), 4.0, -4.0).astype(np.float32),
    )
    np.testing.assert_array_equal(
        predictor.mask_inputs[0][1],
        np.where(base_instances[1].to_full_mask(), 4.0, -4.0).astype(np.float32),
    )
    assert result.base_instances == base_instances
    assert len(result.refined_instances) == 2
    assert result.refined_instances[0].source == "grid_refined"
    assert result.refined_instances[0].base_score == 0.4
    assert result.refined_instances[0].score == np.float32(0.9)


def test_grid_mask_refiner_returns_empty_when_grid_finds_no_masks():
    from src.predict.refine import GridMaskRefiner

    class FakeBaseGenerator:
        def generate_instances(self, target_image):
            return []

    class FakePredictor:
        def encode_image(self, image):
            raise AssertionError("encode_image should not be called")

    refiner = GridMaskRefiner(
        predictor=FakePredictor(),
        base_generator=FakeBaseGenerator(),
    )

    result = refiner.refine(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result.base_instances == []
    assert result.refined_instances == []


def test_refine_package_exports_grid_mask_refiner_api():
    import src.predict.refine as refine
    from src.predict.refine.grid import GridMaskRefiner, GridRefineResult

    assert refine.GridMaskRefiner is GridMaskRefiner
    assert refine.GridRefineResult is GridRefineResult
    assert hasattr(refine, "MaskRefiner")
    assert not hasattr(refine, "__all__")
