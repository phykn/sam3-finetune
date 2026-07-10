import numpy as np
import torch
from PIL import Image
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

    @property
    def mask_input_size(self):
        return self.prompt_encoder.mask_input_size

    def get_image_position_encoding(self, device=None):
        pe = self.prompt_encoder.get_dense_pe()
        return pe if device is None else pe.to(device)

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
        cond=None,
        prompt_type=None,
    ):
        self.decodes.append(
            {
                "image_embed": tuple(image_embed.shape),
                "high_res": [tuple(x.shape) for x in high_res_features],
                "image_pe": tuple(image_pe.shape),
                "multimask": multimask,
                "repeat_image": repeat_image,
                "cond": cond,
                "prompt_type": prompt_type,
            }
        )
        batch = prompt[0].shape[0]
        return (
            torch.ones(batch, 1, 288, 288),
            torch.full((batch, 1), 0.75),
            torch.zeros(batch, 1, 256),
            torch.ones(batch, 1),
        )


class FakeModelWithClasses(FakeModel):
    def decode_masks(self, *args, **kwargs):
        out = super().decode_masks(*args, **kwargs)
        batch = out[0].shape[0]
        classes = torch.tensor([[[2.0, -2.0]]]).expand(batch, -1, -1)
        return (*out, classes)


def test_single_predictor_predicts_from_box():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu")

    out = predictor.predict(
        Image.new("RGB", (20, 10), color=(0, 0, 0)),
        box=np.array([2, 1, 18, 9], dtype=np.float32),
        multimask=False,
    )

    assert out["masks"].shape == (1, 10, 20)
    assert out["logits"].shape == (1, 288, 288)
    np.testing.assert_allclose(out["scores"], np.array([0.75], dtype=np.float32))
    coords, labels = model.prompts[0][0]
    assert coords.shape == (1, 2, 2)
    assert labels.tolist() == [[2, 3]]
    assert model.decodes[0]["repeat_image"] is True
    assert model.decodes[0]["multimask"] is False


def test_finetune_prediction_adds_per_mask_class_scores():
    predictor = SinglePredictor(FakeModelWithClasses(), device="cpu")

    out = predictor.predict(
        Image.new("RGB", (20, 10), color=(0, 0, 0)),
        point_coords=np.array([[10.0, 5.0]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask=False,
    )

    assert out["class_logits"].shape == (1, 2)
    np.testing.assert_allclose(
        out["class_scores"],
        1 / (1 + np.exp(-out["class_logits"])),
        rtol=1e-6,
    )


def test_plain_prediction_does_not_add_class_keys():
    predictor = SinglePredictor(FakeModel(), device="cpu")

    out = predictor.predict(
        Image.new("RGB", (20, 10), color=(0, 0, 0)),
        point_coords=np.array([[10.0, 5.0]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask=False,
    )

    assert "class_logits" not in out
    assert "class_scores" not in out


def test_single_predictor_uses_dummy_point_for_mask_only_prompt():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu")

    predictor.predict(
        Image.new("RGB", (20, 10), color=(0, 0, 0)),
        mask=np.ones((288, 288), dtype=np.float32),
    )

    coords, labels = model.prompts[0][0]
    assert coords.shape == (1, 1, 2)
    assert labels.tolist() == [[-1]]


def test_single_predictor_refines_from_logit():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu")
    logit = np.ones((288, 288), dtype=np.float32)

    out = predictor.refine(
        Image.new("RGB", (20, 10), color=(0, 0, 0)),
        logit,
    )

    assert out["masks"].shape == (1, 10, 20)
    assert model.prompts[0][2].dtype == torch.float32
    assert model.decodes[0]["multimask"] is False


def test_single_predictor_refines_low_from_logit():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu")
    embed = predictor.encode(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    out = predictor.refine_low(
        embed,
        np.ones((288, 288), dtype=np.float32),
        point_coords=np.array([[[10.0, 5.0]]], dtype=np.float32),
        point_labels=np.array([[1]], dtype=np.int32),
    )

    assert out["masks"].shape == (1, 288, 288)
    assert model.prompts[0][2].dtype == torch.float32
    assert model.decodes[0]["multimask"] is False


def test_single_predictor_can_keep_low_res_masks():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu")
    embed = predictor.encode(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    out = predictor.predict_embed_low(
        embed,
        point_coords=np.array([[[10.0, 5.0]]], dtype=np.float32),
        point_labels=np.array([[1]], dtype=np.int32),
        multimask=False,
    )

    assert out["masks"].shape == (1, 288, 288)
    assert out["logits"].shape == (1, 288, 288)
    assert model.decodes[0]["multimask"] is False


def test_single_predictor_passes_condition_and_prompt_type():
    model = FakeModel()
    predictor = SinglePredictor(model, device="cpu", cond=2)
    embed = predictor.encode(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    predictor.predict_embed_low(
        embed,
        point_coords=np.array([[[10.0, 5.0]]], dtype=np.float32),
        point_labels=np.array([[1]], dtype=np.int32),
    )
    predictor.predict_embed_low(
        embed,
        box=np.array([2, 1, 18, 9], dtype=np.float32),
        cond=1,
    )
    predictor.refine_low(
        embed,
        np.ones((288, 288), dtype=np.float32),
        point_coords=np.array([[[10.0, 5.0]]], dtype=np.float32),
        point_labels=np.array([[1]], dtype=np.int32),
    )

    assert model.decodes[0]["cond"] == 2
    assert model.decodes[0]["prompt_type"] == "point"
    assert model.decodes[1]["cond"] == 1
    assert model.decodes[1]["prompt_type"] == "box"
    assert model.decodes[2]["cond"] == 2
    assert model.decodes[2]["prompt_type"] == "mask"


def test_single_predictor_rejects_empty_prompt():
    predictor = SinglePredictor(FakeModel(), device="cpu")

    try:
        predictor.predict(Image.new("RGB", (20, 10), color=(0, 0, 0)))
    except ValueError as exc:
        assert "prompt" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_single_predictor_keeps_fixed_image_options():
    predictor = SinglePredictor(FakeModel(), device="cpu")
    assert not hasattr(predictor, "mask_threshold")

    for kwargs in ({"image_size": 512}, {"mask_threshold": 0.5}):
        try:
            SinglePredictor(FakeModel(), device="cpu", **kwargs)
        except TypeError:
            pass
        else:
            raise AssertionError(f"Expected TypeError for {kwargs}")
