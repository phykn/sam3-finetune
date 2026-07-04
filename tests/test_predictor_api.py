import numpy as np
import torch
from PIL import Image

from src.predictor import Sam3Predictor


class FakePromptEncoder(torch.nn.Module):
    mask_input_size = (288, 288)

    def forward(self, points=None, boxes=None, masks=None):
        self.last_points = points
        batch_size = points[0].shape[0] if points is not None else masks.shape[0]
        sparse = torch.zeros(batch_size, 3, 256)
        dense = torch.zeros(batch_size, 256, 72, 72)
        return sparse, dense

    def get_dense_pe(self):
        return torch.zeros(1, 256, 72, 72)


class FakeMaskDecoder(torch.nn.Module):
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
        self.last_repeat_image = repeat_image
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
        return {
            "image_embed": torch.zeros(1, 256, 72, 72),
            "high_res_features": [
                torch.zeros(1, 32, 288, 288),
                torch.zeros(1, 64, 144, 144),
            ],
        }


def test_predictor_accepts_box_and_returns_numpy_outputs():
    predictor = Sam3Predictor(FakeModel(), device=torch.device("cpu"))
    predictor.set_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    masks, scores, low_res = predictor.predict(
        box=np.array([2, 1, 18, 9], dtype=np.float32)
    )

    assert masks.shape == (1, 10, 20)
    np.testing.assert_allclose(scores, np.array([0.9], dtype=np.float32))
    assert low_res.shape == (1, 288, 288)
    assert predictor.model.mask_decoder.last_repeat_image is True


def test_predictor_adds_dummy_negative_point_for_mask_only_prompt():
    model = FakeModel()
    predictor = Sam3Predictor(model, device=torch.device("cpu"))
    predictor.set_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    predictor.predict(mask_input=np.ones((288, 288), dtype=np.float32))

    coords, labels = model.prompt_encoder.last_points
    assert coords.shape == (1, 1, 2)
    assert labels.tolist() == [[-1]]


def test_predictor_accepts_batched_boxes():
    model = FakeModel()
    predictor = Sam3Predictor(model, device=torch.device("cpu"))
    predictor.set_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    masks, scores, low_res = predictor.predict(
        box=np.array(
            [
                [2, 1, 18, 9],
                [4, 2, 10, 8],
            ],
            dtype=np.float32,
        )
    )

    coords, labels = model.prompt_encoder.last_points
    assert coords.shape == (2, 2, 2)
    assert labels.tolist() == [[2, 3], [2, 3]]
    assert masks.shape == (2, 1, 10, 20)
    assert scores.shape == (2, 1)
    assert low_res.shape == (2, 1, 288, 288)
