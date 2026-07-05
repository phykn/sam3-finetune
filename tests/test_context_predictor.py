import numpy as np
import torch
from PIL import Image
from src.types import Sam3ImageEmbedding


def _embedding_from_feature_map(feature_map: torch.Tensor) -> Sam3ImageEmbedding:
    return Sam3ImageEmbedding(
        image_embed=feature_map.unsqueeze(0),
        high_res_features=(),
        orig_hw=(40, 40),
    )


_FEATURES_BY_IMAGE_ID: dict[int, torch.Tensor] = {}


def _image_from_feature_map(feature_map: torch.Tensor) -> np.ndarray:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    _FEATURES_BY_IMAGE_ID[id(image)] = feature_map
    return image


class FakeContextPredictor:
    def __init__(self) -> None:
        self.decode_batches = []
        self.mask_inputs = []

    def encode_image_batch(self, images):
        return [
            _embedding_from_feature_map(_FEATURES_BY_IMAGE_ID[id(image)])
            for image in images
        ]

    def predict_from_embedding(
        self,
        embedding,
        point_coords=None,
        point_labels=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        self.decode_batches.append(point_coords.copy())
        self.mask_inputs.append(None if mask_input is None else mask_input.copy())
        batch = point_coords.shape[0]
        masks = np.zeros((batch, 1, 40, 40), dtype=bool)
        low_res = np.full((batch, 1, 8, 8), -2.0, dtype=np.float32)
        scores = np.ones((batch, 1), dtype=np.float32)
        for index in range(batch):
            x = int(round(float(point_coords[index, 0, 0])))
            y = int(round(float(point_coords[index, 0, 1])))
            x0 = max(x - 3, 0)
            y0 = max(y - 3, 0)
            x1 = min(x + 3, 40)
            y1 = min(y + 3, 40)
            masks[index, 0, y0:y1, x0:x1] = True
            low_res[
                index,
                0,
                y0 // 5 : max(y1 // 5, y0 // 5 + 1),
                x0 // 5 : max(x1 // 5, x0 // 5 + 1),
            ] = 2.0
            scores[index, 0] = 0.8 - index * 0.01
        return masks, scores, low_res


def test_context_predictor_selects_target_points_from_reference_mask_similarity():
    from src.predict.reference.matcher import ContextMatcher
    from src.types import ContextReference

    reference_features = torch.zeros(2, 4, 4, dtype=torch.float32)
    reference_features[0, 1:3, 1:3] = 3.0
    reference_features[1] = 0.1
    target_features = torch.zeros(2, 4, 4, dtype=torch.float32)
    target_features[0, 1, 2] = 3.0
    target_features[0, 3, 0] = 1.0
    target_features[1] = 0.1
    reference_image = _image_from_feature_map(reference_features)
    target_image = _image_from_feature_map(target_features)

    reference_mask = np.zeros((40, 40), dtype=bool)
    reference_mask[10:30, 10:30] = True
    fake = FakeContextPredictor()
    predictor = ContextMatcher(
        fake,
        candidate_count=4,
        decode_batch_size=2,
        max_masks=2,
        mask_nms_thresh=0.0,
        min_cell_distance=1,
    )

    predictions = predictor.predict(
        target_image=target_image,
        references=[
            ContextReference(
                image=reference_image,
                mask=reference_mask,
            )
        ],
    )

    assert len(predictions) == 2
    assert fake.decode_batches[0].shape == (2, 1, 2)
    np.testing.assert_allclose(fake.decode_batches[0][0, 0], np.array([25.0, 15.0]))
    selected = next(
        prediction
        for prediction in predictions
        if prediction.point_coords == (25.0, 15.0)
    )
    assert selected.bbox == (22, 12, 28, 18)
    assert selected.segmentation.shape == (6, 6)
    assert selected.image_size == (40, 40)


def test_context_matcher_lives_under_context_package() -> None:
    from pathlib import Path

    from src.predict.reference.matcher import ContextMatcher
    from src.predict.reference.postprocess import context_prediction_to_full_mask
    from src.predict.reference.scoring import area_ratio_score
    from src.types import ContextPrediction, ContextReference

    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "predict" / "reference" / "matcher.py").is_file()
    assert (root / "src" / "predict" / "reference" / "postprocess.py").is_file()
    assert (root / "src" / "predict" / "reference" / "prototype.py").is_file()
    assert (root / "src" / "predict" / "reference" / "scoring.py").is_file()
    assert not (root / "src" / "context").exists()
    assert not (root / "src" / "context_predictor.py").exists()
    assert ContextMatcher.__module__ == "src.predict.reference.matcher"
    assert ContextReference.__module__ == "src.types"
    assert ContextPrediction.__module__ == "src.types"
    assert (
        context_prediction_to_full_mask.__module__
        == "src.predict.reference.postprocess"
    )
    assert area_ratio_score.__module__ == "src.predict.reference.scoring"


def test_context_package_exports_user_facing_api() -> None:
    import src.predict.reference as context
    from src.predict.reference.guided import ReferenceGuidedMaskGenerator
    from src.predict.reference.matcher import ContextMatcher
    from src.types import ContextPrediction, ContextReference

    assert context.ContextMatcher is ContextMatcher
    assert context.ReferenceGuidedMaskGenerator is ReferenceGuidedMaskGenerator
    assert context.ContextPrediction is ContextPrediction
    assert context.ContextReference is ContextReference
    assert not hasattr(context, "__all__")


def test_contrastive_context_penalizes_reference_background_like_candidates():
    from src.predict.reference.matcher import ContextMatcher
    from src.types import ContextReference

    reference_features = torch.zeros(2, 4, 4, dtype=torch.float32)
    reference_features[:, :, :] = torch.tensor([1.0, 0.0])[:, None, None]
    reference_features[:, 1:3, 1:3] = torch.tensor([1.0, 1.0])[:, None, None]
    target_features = torch.zeros(2, 4, 4, dtype=torch.float32)
    target_features[:, 1, 2] = torch.tensor([1.0, 1.0])
    target_features[:, 3, 0] = torch.tensor([1.0, 0.0])
    reference_image = _image_from_feature_map(reference_features)
    target_image = _image_from_feature_map(target_features)

    reference_mask = np.zeros((40, 40), dtype=bool)
    reference_mask[10:30, 10:30] = True
    fake = FakeContextPredictor()
    predictor = ContextMatcher(
        fake,
        candidate_count=16,
        decode_batch_size=16,
        max_masks=16,
        mask_nms_thresh=0.0,
        min_cell_distance=1,
        negative_context_mode="image",
        negative_context_weight=1.0,
        predicted_iou_weight=0.0,
        stability_score_weight=0.0,
    )

    predictions = predictor.predict(
        target_image=target_image,
        references=[
            ContextReference(
                image=reference_image,
                mask=reference_mask,
            )
        ],
    )

    part_like = next(
        prediction
        for prediction in predictions
        if prediction.point_coords == (25.0, 15.0)
    )
    background_like = next(
        prediction
        for prediction in predictions
        if prediction.point_coords == (5.0, 35.0)
    )
    assert part_like.context_score > background_like.context_score
    assert background_like.context_score < 0.0


def test_shape_candidate_scoring_prefers_distributed_reference_match():
    from src.predict.reference.matcher import ContextMatcher
    from src.types import ContextReference

    reference_features = torch.zeros(2, 5, 5, dtype=torch.float32)
    reference_features[0, 1:4, 1:4] = 1.0
    target_features = torch.zeros(2, 5, 5, dtype=torch.float32)
    target_features[:, 0, 0] = torch.tensor([1.0, 0.0])
    target_features[:, 2:5, 2:5] = torch.tensor([0.8, 0.6])[:, None, None]
    reference_image = _image_from_feature_map(reference_features)
    target_image = _image_from_feature_map(target_features)

    reference_mask = np.zeros((40, 40), dtype=bool)
    reference_mask[8:32, 8:32] = True
    fake = FakeContextPredictor()
    predictor = ContextMatcher(
        fake,
        candidate_count=1,
        decode_batch_size=1,
        max_masks=1,
        candidate_score_mode="shape",
        negative_context_mode="none",
        predicted_iou_weight=0.0,
        stability_score_weight=0.0,
    )

    predictor.predict(
        target_image=target_image,
        references=[
            ContextReference(
                image=reference_image,
                mask=reference_mask,
            )
        ],
    )

    selected_point = fake.decode_batches[0][0, 0]
    assert selected_point[0] > 16
    assert selected_point[1] > 16


def test_context_predictor_uses_explicit_target_points_as_candidates():
    from src.predict.reference.matcher import ContextMatcher
    from src.types import ContextReference

    reference_features = torch.zeros(2, 4, 4, dtype=torch.float32)
    reference_features[0, 1:3, 1:3] = 3.0
    target_features = torch.zeros(2, 4, 4, dtype=torch.float32)
    reference_image = _image_from_feature_map(reference_features)
    target_image = _image_from_feature_map(target_features)
    reference_mask = np.zeros((40, 40), dtype=bool)
    reference_mask[10:30, 10:30] = True
    fake = FakeContextPredictor()
    predictor = ContextMatcher(
        fake,
        candidate_count=4,
        decode_batch_size=2,
        max_masks=1,
    )

    predictor.predict(
        target_image=target_image,
        references=[
            ContextReference(
                image=reference_image,
                mask=reference_mask,
            )
        ],
        target_point_coords=np.asarray([[12.0, 34.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(fake.decode_batches[0][0, 0], np.array([12.0, 34.0]))


def test_context_predictor_rejects_non_positive_predict_max_masks():
    from src.predict.reference.matcher import ContextMatcher
    from src.types import ContextReference

    fake = FakeContextPredictor()
    predictor = ContextMatcher(fake)
    image = _image_from_feature_map(torch.zeros(2, 4, 4))

    try:
        predictor.predict(
            target_image=image,
            references=[
                ContextReference(
                    image=image,
                    mask=np.ones((40, 40), dtype=bool),
                )
            ],
            max_masks=0,
        )
    except ValueError as exc:
        assert "max_masks" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_context_predictor_rejects_target_points_on_image_boundary():
    from src.predict.reference.matcher import ContextMatcher
    from src.types import ContextReference

    fake = FakeContextPredictor()
    predictor = ContextMatcher(fake)
    image = _image_from_feature_map(torch.zeros(2, 4, 4))

    try:
        predictor.predict(
            target_image=image,
            references=[
                ContextReference(
                    image=image,
                    mask=np.ones((40, 40), dtype=bool),
                )
            ],
            target_point_coords=np.asarray([[40.0, 10.0]], dtype=np.float32),
        )
    except ValueError as exc:
        assert "target_point_coords" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_build_context_prototype_rejects_embedding_length_mismatch():
    from src.predict.reference.prototype import build_context_prototype
    from src.types import ContextReference

    try:
        build_context_prototype(
            [
                ContextReference(
                    image=Image.new("RGB", (40, 40)),
                    mask=np.ones((40, 40), dtype=bool),
                )
            ],
            [],
            feature_layer="image_embed",
            negative_context_mode="local",
            negative_context_scale=2.0,
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_context_predictor_rejects_empty_reference_mask():
    from src.predict.reference.matcher import ContextMatcher
    from src.types import ContextReference

    fake = FakeContextPredictor()
    predictor = ContextMatcher(fake)
    image = _image_from_feature_map(torch.zeros(2, 4, 4))

    try:
        predictor.predict(
            target_image=image,
            references=[
                ContextReference(
                    image=image,
                    mask=np.zeros((40, 40), dtype=bool),
                )
            ],
        )
    except ValueError as exc:
        assert "reference mask" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_context_reference_accepts_pil_images_for_public_api():
    from src.types import ContextReference

    reference = ContextReference(
        image=Image.new("RGB", (4, 4), color=(0, 0, 0)),
        mask=np.ones((4, 4), dtype=bool),
        weight=2.0,
    )

    assert reference.weight == 2.0


def test_context_reference_rejects_tensor_images():
    from src.types import ContextReference

    try:
        ContextReference(
            image=torch.zeros(3, 4, 4),
            mask=np.ones((4, 4), dtype=bool),
        )
    except TypeError as exc:
        assert "image" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


def test_context_smoke_script_loads_binary_reference_mask(tmp_path):
    from scripts.context_prompt_smoke_test import load_reference_mask_image

    mask_path = tmp_path / "mask.png"
    Image.fromarray(
        np.array([[0, 255, 255], [0, 0, 255]], dtype=np.uint8),
        mode="L",
    ).save(mask_path)

    mask = load_reference_mask_image(mask_path, expected_size=(3, 2))

    assert mask.dtype == bool
    np.testing.assert_array_equal(
        mask,
        np.array([[False, True, True], [False, False, True]], dtype=bool),
    )


def test_reference_prompt_points_encode_positive_and_negative_labels():
    from scripts.video_memory_reference import make_point_prompt_arrays

    coords, labels = make_point_prompt_arrays(
        positive_points=[(10, 20)],
        negative_points=[(30, 40), (50, 60)],
    )

    np.testing.assert_allclose(
        coords,
        np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        labels,
        np.array([1, 0, 0], dtype=np.int64),
    )


def test_reference_prompt_refinement_keeps_original_prompts(monkeypatch):
    from scripts.video_memory_reference import predict_sam_mask_from_prompts
    from src.predict import Sam3Predictor

    class FakeSam3Predictor:
        def __init__(self) -> None:
            self.predict_calls = []

        def set_image(self, image):
            self.image = image

        def predict(self, **kwargs):
            self.predict_calls.append(kwargs)
            if len(self.predict_calls) == 1:
                masks = np.zeros((2, 8, 8), dtype=bool)
                masks[1, 2:6, 2:6] = True
                scores = np.array([0.1, 0.9], dtype=np.float32)
                low_res = np.zeros((2, 4, 4), dtype=np.float32)
                low_res[1, 1:3, 1:3] = 2.0
                return masks, scores, low_res
            masks = np.zeros((1, 8, 8), dtype=bool)
            masks[0, 2:5, 2:5] = True
            scores = np.array([0.8], dtype=np.float32)
            low_res = np.zeros((1, 4, 4), dtype=np.float32)
            return masks, scores, low_res

    fake = FakeSam3Predictor()
    monkeypatch.setattr(Sam3Predictor, "from_checkpoint", lambda *args, **kwargs: fake)

    result = predict_sam_mask_from_prompts(
        Image.new("RGB", (8, 8)),
        checkpoint="unused.pt",
        device="cpu",
        positive_points=[(2, 2)],
        negative_points=[(6, 6)],
        box=[1, 1, 7, 7],
    )

    assert result.source == "sam_prompt"
    refinement_call = fake.predict_calls[1]
    np.testing.assert_allclose(
        refinement_call["point_coords"],
        np.array([[2, 2], [6, 6]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        refinement_call["point_labels"],
        np.array([1, 0], dtype=np.int64),
    )
    np.testing.assert_allclose(
        refinement_call["box"],
        np.array([1, 1, 7, 7], dtype=np.float32),
    )
    assert refinement_call["mask_input"].shape == (4, 4)


def test_context_predictor_can_send_reference_shape_as_mask_prior():
    from src.predict.reference.matcher import ContextMatcher
    from src.types import ContextReference

    reference_features = torch.zeros(2, 4, 4, dtype=torch.float32)
    reference_features[0, 1:3, 1:3] = 3.0
    reference_features[1] = 0.1
    target_features = torch.zeros(2, 4, 4, dtype=torch.float32)
    target_features[0, 1, 2] = 3.0
    target_features[1] = 0.1
    reference_image = _image_from_feature_map(reference_features)
    target_image = _image_from_feature_map(target_features)

    reference_mask = np.zeros((40, 40), dtype=bool)
    reference_mask[10:30, 12:28] = True
    fake = FakeContextPredictor()
    predictor = ContextMatcher(
        fake,
        candidate_count=2,
        decode_batch_size=2,
        max_masks=1,
        min_cell_distance=1,
        use_reference_mask_prior=True,
    )

    predictor.predict(
        target_image=target_image,
        references=[
            ContextReference(
                image=reference_image,
                mask=reference_mask,
            )
        ],
    )

    mask_prior = fake.mask_inputs[0]
    assert mask_prior.shape == (2, 40, 40)
    assert mask_prior[0, 15, 25] > 0
    assert mask_prior[0, 0, 0] < 0


def test_context_prediction_to_full_mask_reconstructs_roi():
    from src.predict.reference.postprocess import context_prediction_to_full_mask
    from src.types import ContextPrediction

    prediction = ContextPrediction(
        segmentation=np.array([[True, False], [True, True]], dtype=bool),
        bbox=(2, 1, 4, 3),
        area=3,
        point_coords=(3.0, 2.0),
        context_score=0.5,
        predicted_iou=0.8,
        stability_score=1.0,
        score=0.63,
        image_size=(5, 4),
    )

    full_mask = context_prediction_to_full_mask(prediction)

    expected = np.zeros((4, 5), dtype=bool)
    expected[1:3, 2:4] = prediction.segmentation
    np.testing.assert_array_equal(full_mask, expected)


def test_area_ratio_score_penalizes_masks_with_different_relative_size():
    from src.predict.reference.scoring import area_ratio_score

    same = area_ratio_score(candidate_ratio=0.05, reference_ratio=0.05)
    larger = area_ratio_score(candidate_ratio=0.20, reference_ratio=0.05)
    smaller = area_ratio_score(candidate_ratio=0.0125, reference_ratio=0.05)

    assert same == 1.0
    assert larger < same
    assert smaller < same
    np.testing.assert_allclose(larger, smaller)
