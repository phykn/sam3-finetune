import importlib

import numpy as np
import pytest
import torch
from PIL import Image
from src.predict.prompted import Sam3Predictor
from src.types import Sam3ImageEmbedding, Sam3PromptBatch


class FakePromptEncoder(torch.nn.Module):
    mask_input_size = (288, 288)

    def __init__(self):
        super().__init__()
        self.get_dense_pe_calls = 0
        self.forward_calls = 0

    def forward(self, points=None, boxes=None, masks=None):
        self.forward_calls += 1
        self.last_points = points
        batch_size = points[0].shape[0] if points is not None else masks.shape[0]
        sparse = torch.zeros(batch_size, 3, 256)
        dense = torch.zeros(batch_size, 256, 72, 72)
        return sparse, dense

    def get_dense_pe(self):
        self.get_dense_pe_calls += 1
        return torch.zeros(1, 256, 72, 72)


class FakeMaskDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        repeat_image,
        high_res_features,
    ):
        self.forward_calls += 1
        self.last_repeat_image = repeat_image
        self.last_image_embeddings_shape = tuple(image_embeddings.shape)
        self.last_high_res_shapes = [
            tuple(feature.shape) for feature in high_res_features
        ]
        batch_size = sparse_prompt_embeddings.shape[0]
        return (
            torch.ones(batch_size, 1, 288, 288),
            torch.full((batch_size, 1), 0.9),
            torch.zeros(batch_size, 1, 256),
            torch.ones(batch_size, 1),
        )


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_encoder = FakePromptEncoder()
        self.mask_decoder = FakeMaskDecoder()

    def encode_image(self, images):
        self.last_encoded_shape = tuple(images.shape)
        batch_size = images.shape[0]
        return {
            "image_embed": torch.arange(
                batch_size * 256 * 72 * 72,
                dtype=torch.float32,
            ).reshape(batch_size, 256, 72, 72),
            "high_res_features": [
                torch.zeros(batch_size, 32, 288, 288),
                torch.zeros(batch_size, 64, 144, 144),
            ],
        }


def test_package_public_surface_requires_workflow_imports():
    import src
    import src.predict.prompted as prompted
    import src.predict.prompted.predictor as predictor_module

    predict_root = importlib.import_module("src.predict")

    assert prompted.Sam3Predictor is Sam3Predictor
    assert not hasattr(src, "Sam3Predictor")
    assert not hasattr(src, "Sam3ImageEmbedding")
    assert not hasattr(src, "Sam3PromptBatch")
    assert not hasattr(predict_root, "Sam3Predictor")
    assert not hasattr(predict_root, "Sam3ImageEmbedding")
    assert not hasattr(predict_root, "Sam3PromptBatch")
    assert not hasattr(predictor_module, "Sam3ImageEmbedding")
    assert not hasattr(predictor_module, "Sam3PromptBatch")
    assert not hasattr(src, "__all__")
    assert not hasattr(predict_root, "__all__")
    for name in (
        "Sam3PromptBatch",
        "AutomaticMaskGenerator",
        "ContextMatcher",
        "NextFramePredictor",
        "VideoMemoryInference",
        "GroundingInference",
        "VisualLanguageCache",
        "build_model",
        "filter_grounding_prediction",
    ):
        assert not hasattr(src, name)

    for old_module in (
        "src.predict.image",
        "src.predict.masks",
        "src.predict.reference",
        "src.predict.video",
    ):
        with pytest.raises(ModuleNotFoundError) as exc:
            importlib.import_module(old_module)
        assert exc.value.name == old_module


def test_predictor_accepts_box_and_returns_numpy_outputs():
    predictor = Sam3Predictor(FakeModel(), device=torch.device("cpu"))
    embedding = predictor.encode_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    masks, scores, low_res = predictor.predict_from_embedding(
        embedding, box=np.array([2, 1, 18, 9], dtype=np.float32)
    )

    assert masks.shape == (1, 10, 20)
    np.testing.assert_allclose(scores, np.array([0.9], dtype=np.float32))
    assert low_res.shape == (1, 288, 288)
    assert predictor.model.mask_decoder.last_repeat_image is True


def test_predictor_adds_dummy_negative_point_for_mask_only_prompt():
    model = FakeModel()
    predictor = Sam3Predictor(model, device=torch.device("cpu"))
    embedding = predictor.encode_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    predictor.predict_from_embedding(
        embedding,
        mask_input=np.ones((288, 288), dtype=np.float32),
    )

    coords, labels = model.prompt_encoder.last_points
    assert coords.shape == (1, 1, 2)
    assert labels.tolist() == [[-1]]


def test_predictor_accepts_batched_boxes():
    model = FakeModel()
    predictor = Sam3Predictor(model, device=torch.device("cpu"))
    embedding = predictor.encode_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    masks, scores, low_res = predictor.predict_from_embedding(
        embedding,
        box=np.array(
            [
                [2, 1, 18, 9],
                [4, 2, 10, 8],
            ],
            dtype=np.float32,
        ),
    )

    coords, labels = model.prompt_encoder.last_points
    assert coords.shape == (2, 2, 2)
    assert labels.tolist() == [[2, 3], [2, 3]]
    assert masks.shape == (2, 1, 10, 20)
    assert scores.shape == (2, 1)
    assert low_res.shape == (2, 1, 288, 288)


def test_predictor_does_not_expose_stateful_image_api():
    predictor = Sam3Predictor(FakeModel(), device=torch.device("cpu"))

    assert not hasattr(predictor, "set_image")
    assert not hasattr(predictor, "set_image_embedding")
    assert not hasattr(predictor, "predict")


def test_encode_image_tensor_batch_returns_one_embedding_per_tensor():
    model = FakeModel()
    predictor = Sam3Predictor(model, device=torch.device("cpu"))
    input_tensor = torch.zeros(2, 3, 1008, 1008)

    embeddings = predictor.encode_image_tensor_batch(
        input_tensor,
        [(10, 20), (30, 40)],
    )

    assert len(embeddings) == 2
    assert isinstance(embeddings[0], Sam3ImageEmbedding)
    assert embeddings[0].image_embed.shape[0] == 1
    assert embeddings[0].orig_hw == (10, 20)
    assert embeddings[1].orig_hw == (30, 40)
    assert model.last_encoded_shape == (2, 3, 1008, 1008)


def test_encode_image_tensor_batch_rejects_empty_batch():
    predictor = Sam3Predictor(FakeModel(), device=torch.device("cpu"))

    try:
        predictor.encode_image_tensor_batch(torch.zeros(0, 3, 1008, 1008), [])
    except ValueError as exc:
        assert "batch" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_encode_image_batch_stacks_preprocessed_images():
    model = FakeModel()
    predictor = Sam3Predictor(model, device=torch.device("cpu"))

    embeddings = predictor.encode_image_batch(
        [
            Image.new("RGB", (20, 10), color=(0, 0, 0)),
            Image.new("RGB", (40, 30), color=(0, 0, 0)),
        ]
    )

    assert len(embeddings) == 2
    assert model.last_encoded_shape == (2, 3, 1008, 1008)
    assert [embedding.orig_hw for embedding in embeddings] == [(10, 20), (30, 40)]


def test_encode_image_batch_uses_batch_preprocessing():
    class TrackingTransforms:
        def __init__(self) -> None:
            self.batch_calls = 0
            self.device = None

        def preprocess_images(self, images, device):
            self.batch_calls += 1
            self.device = device
            return torch.zeros(len(images), 3, 1008, 1008), [(10, 20), (30, 40)]

        def preprocess_image(self, image, device):
            raise AssertionError("encode_image_batch should preprocess as a batch")

    model = FakeModel()
    predictor = Sam3Predictor(model, device=torch.device("cpu"))
    transforms = TrackingTransforms()
    predictor.transforms = transforms

    embeddings = predictor.encode_image_batch(
        [
            Image.new("RGB", (20, 10), color=(0, 0, 0)),
            Image.new("RGB", (40, 30), color=(0, 0, 0)),
        ]
    )

    assert transforms.batch_calls == 1
    assert transforms.device == torch.device("cpu")
    assert len(embeddings) == 2


def test_predict_from_embedding_does_not_require_set_image():
    predictor = Sam3Predictor(FakeModel(), device=torch.device("cpu"))
    embedding = predictor.encode_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    masks, scores, low_res = predictor.predict_from_embedding(
        embedding,
        point_coords=np.array([[[10, 5]]], dtype=np.float32),
        point_labels=np.array([[1]], dtype=np.int64),
    )

    assert masks.shape == (1, 10, 20)
    assert scores.shape == (1,)
    assert low_res.shape == (1, 288, 288)


def test_decode_low_res_from_embedding_returns_tensors_without_postprocess():
    predictor = Sam3Predictor(FakeModel(), device=torch.device("cpu"))
    embedding = predictor.encode_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    low_res, scores = predictor.decode_low_res_from_embedding(
        embedding,
        point_coords=np.array([[[10, 5]]], dtype=np.float32),
        point_labels=np.array([[1]], dtype=np.int64),
    )

    assert isinstance(low_res, torch.Tensor)
    assert isinstance(scores, torch.Tensor)
    assert tuple(low_res.shape) == (1, 1, 288, 288)
    assert tuple(scores.shape) == (1, 1)


def test_predictor_caches_dense_position_encoding():
    model = FakeModel()
    predictor = Sam3Predictor(model, device=torch.device("cpu"))
    embedding = predictor.encode_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    for _ in range(2):
        predictor.predict_from_embedding(
            embedding,
            point_coords=np.array([[[10, 5]]], dtype=np.float32),
            point_labels=np.array([[1]], dtype=np.int64),
        )

    assert model.prompt_encoder.get_dense_pe_calls == 1


def test_predict_from_embedding_batches_decodes_multiple_prompt_batches_once():
    model = FakeModel()
    predictor = Sam3Predictor(model, device=torch.device("cpu"))
    embedding_a = predictor.encode_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))
    embedding_b = predictor.encode_image(Image.new("RGB", (40, 30), color=(0, 0, 0)))

    results = predictor.predict_from_embedding_batches(
        [
            Sam3PromptBatch(
                embedding=embedding_a,
                point_coords=np.array(
                    [
                        [[10, 5]],
                        [[12, 6]],
                    ],
                    dtype=np.float32,
                ),
                point_labels=np.ones((2, 1), dtype=np.int64),
            ),
            Sam3PromptBatch(
                embedding=embedding_b,
                point_coords=np.array([[[20, 15]]], dtype=np.float32),
                point_labels=np.ones((1, 1), dtype=np.int64),
            ),
        ],
        multimask_output=True,
    )

    assert model.mask_decoder.forward_calls == 1
    assert model.prompt_encoder.forward_calls == 1
    assert model.mask_decoder.last_repeat_image is False
    assert model.mask_decoder.last_image_embeddings_shape[0] == 3
    assert model.mask_decoder.last_high_res_shapes[0][0] == 3
    assert len(results) == 2
    masks_a, scores_a, low_res_a = results[0]
    masks_b, scores_b, low_res_b = results[1]
    assert masks_a.shape == (2, 1, 10, 20)
    assert scores_a.shape == (2, 1)
    assert low_res_a.shape == (2, 1, 288, 288)
    assert masks_b.shape == (1, 1, 30, 40)
    assert scores_b.shape == (1, 1)
    assert low_res_b.shape == (1, 1, 288, 288)


def test_predict_from_embedding_batches_reuses_features_for_same_embedding():
    model = FakeModel()
    predictor = Sam3Predictor(model, device=torch.device("cpu"))
    embedding = predictor.encode_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    predictor.predict_from_embedding_batches(
        [
            Sam3PromptBatch(
                embedding=embedding,
                point_coords=np.array([[[10, 5]], [[12, 6]]], dtype=np.float32),
                point_labels=np.ones((2, 1), dtype=np.int64),
            ),
            Sam3PromptBatch(
                embedding=embedding,
                point_coords=np.array([[[14, 7]]], dtype=np.float32),
                point_labels=np.ones((1, 1), dtype=np.int64),
            ),
        ],
        multimask_output=True,
    )

    assert model.mask_decoder.forward_calls == 1
    assert model.prompt_encoder.forward_calls == 1
    assert model.mask_decoder.last_repeat_image is True
    assert model.mask_decoder.last_image_embeddings_shape[0] == 1
    assert model.mask_decoder.last_high_res_shapes[0][0] == 1
