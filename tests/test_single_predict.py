import numpy as np
import torch
from PIL import Image
from src.predict.result import SingleResult
from src.predict.single import SinglePredictor


class FakePromptEncoder:
    mask_input_size = (288, 288)

    def __init__(self):
        self.pe_calls = 0

    def get_dense_pe(self):
        self.pe_calls += 1
        return torch.zeros(1, 256, 72, 72)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_encoder = FakePromptEncoder()
        self.images = []
        self.prompts = []
        self.decodes = []

    def encode_image(self, images):
        self.images.append(tuple(images.shape))
        batch = images.shape[0]
        return {
            "image_embed": torch.zeros(batch, 256, 72, 72),
            "high_res_features": (
                torch.zeros(batch, 32, 288, 288),
                torch.zeros(batch, 64, 144, 144),
            ),
        }

    def encode_prompt(self, points=None, boxes=None, masks=None):
        self.prompts.append((points, boxes, masks))
        batch = points[0].shape[0] if points is not None else masks.shape[0]
        return torch.zeros(batch, 3, 256), torch.zeros(batch, 256, 72, 72)

    def decode_masks(
        self,
        image_embed,
        high_res_features,
        prompt,
        image_pe,
        multimask=True,
        repeat_image=False,
    ):
        self.decodes.append(
            {
                "image_embed": tuple(image_embed.shape),
                "high_res": [tuple(x.shape) for x in high_res_features],
                "image_pe": tuple(image_pe.shape),
                "multimask": multimask,
                "repeat_image": repeat_image,
            }
        )
        batch = prompt[0].shape[0]
        return (
            torch.ones(batch, 1, 288, 288),
            torch.full((batch, 1), 0.75),
            torch.zeros(batch, 1, 256),
            torch.ones(batch, 1),
        )


def test_single_predictor_predicts_from_box():
    model = FakeModel()
    predictor = SinglePredictor(model, {"device": "cpu", "image_size": 1008})

    out = predictor.predict(
        Image.new("RGB", (20, 10), color=(0, 0, 0)),
        box=np.array([2, 1, 18, 9], dtype=np.float32),
        multimask=False,
    )

    assert isinstance(out, SingleResult)
    assert out.masks.shape == (1, 10, 20)
    assert out.logits.shape == (1, 288, 288)
    np.testing.assert_allclose(out.scores, np.array([0.75], dtype=np.float32))
    coords, labels = model.prompts[0][0]
    assert coords.shape == (1, 2, 2)
    assert labels.tolist() == [[2, 3]]
    assert model.decodes[0]["repeat_image"] is True
    assert model.decodes[0]["multimask"] is False


def test_single_predictor_uses_dummy_point_for_mask_only_prompt():
    model = FakeModel()
    predictor = SinglePredictor(model, {"device": "cpu", "image_size": 1008})

    predictor.predict(
        Image.new("RGB", (20, 10), color=(0, 0, 0)),
        mask=np.ones((288, 288), dtype=np.float32),
    )

    coords, labels = model.prompts[0][0]
    assert coords.shape == (1, 1, 2)
    assert labels.tolist() == [[-1]]


def test_single_predictor_rejects_empty_prompt():
    predictor = SinglePredictor(FakeModel(), {"device": "cpu"})

    try:
        predictor.predict(Image.new("RGB", (20, 10), color=(0, 0, 0)))
    except ValueError as exc:
        assert "prompt" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
