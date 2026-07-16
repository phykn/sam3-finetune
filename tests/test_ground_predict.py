import numpy as np
import pytest
import torch
from PIL import Image

from src.predict.ground import GroundPredictor


class FakeGroundModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_calls = 0
        self.image_grad_enabled = []
        self.prompt_calls = []
        self.decode_batches = []

    def encode_image(self, _image):
        self.image_calls += 1
        self.image_grad_enabled.append(torch.is_grad_enabled())
        features = torch.zeros(1, 2, 2, 2)
        features[:, 0] = 1
        return {"backbone_fpn": (features,)}

    def encode_box_prompts(self, image, boxes, labels, box_mask):
        self.prompt_calls.append((image, boxes, labels, box_mask))
        batch = boxes.shape[1]
        return {
            "features": torch.zeros(3, batch, 2),
            "mask": torch.zeros(batch, 3, dtype=torch.bool),
        }

    def decode(self, _image, prompt):
        batch = prompt["features"].shape[1]
        self.decode_batches.append(batch)
        return {
            "pred_logits": torch.full((batch, 1, 1), 4.0),
            "pred_boxes": torch.tensor([0.5, 0.5, 0.5, 0.5]).repeat(batch, 1, 1),
            "pred_masks": torch.full((batch, 1, 2, 2), 2.0),
            "raw": {"unused": torch.ones(10)},
        }


def test_encode_reference_groups_boxes_by_class_and_encodes_image_once():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")

    reference = predictor.encode_reference(
        Image.new("RGB", (8, 8)),
        [[0, 0, 4, 4], [4, 4, 8, 8], [1, 1, 3, 3]],
        [2, 1, 2],
    )

    assert model.image_calls == 1
    assert reference["prompt_classes"].tolist() == [1, 2]
    assert reference["feature_classes"].tolist() == [2, 1, 2]
    _image, boxes, _labels, box_mask = model.prompt_calls[0]
    assert boxes.shape == (2, 2, 4)
    assert box_mask.tolist() == [[False, True], [False, False]]


def test_encode_uses_inference_mode():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")

    predictor.encode(Image.new("RGB", (8, 8)))

    assert model.image_grad_enabled == [False]


def test_predict_encodes_target_once_and_decodes_prompt_batches():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu", prompt_batch_size=2)
    boxes = [[index, 0, index + 1, 2] for index in range(5)]
    reference = predictor.encode_reference(
        Image.new("RGB", (8, 8)),
        boxes,
        [0, 1, 2, 3, 4],
    )

    objects = predictor.predict(Image.new("RGB", (8, 8)), [reference])

    assert model.image_calls == 2
    assert model.decode_batches == [2, 2, 1]
    assert [item["class_id"] for item in objects] == [0, 1, 2, 3, 4]
    assert all("raw" not in item for item in objects)
    assert all(isinstance(item["roi"], np.ndarray) for item in objects)
    assert all("mask" not in item for item in objects)


def test_predict_embed_reuses_encoded_target():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")
    reference = predictor.encode_reference(
        Image.new("RGB", (8, 8)),
        [[0, 0, 4, 4]],
        [1],
    )
    target = predictor.encode(Image.new("RGB", (8, 8)))

    first = predictor.predict_embed(target, [reference])
    second = predictor.predict_embed(target, [reference])

    assert model.image_calls == 2
    assert len(first) == len(second) == 1
    assert first[0]["class_id"] == second[0]["class_id"] == 1


def test_predict_merges_same_class_features_across_references():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")
    first = predictor.encode_reference(
        Image.new("RGB", (8, 8)),
        [[0, 0, 4, 4]],
        [3],
    )
    second = predictor.encode_reference(
        Image.new("RGB", (8, 8)),
        [[4, 4, 8, 8]],
        [3],
    )

    objects = predictor.predict(Image.new("RGB", (8, 8)), [first, second])

    assert len(objects) == 1
    assert objects[0]["class_id"] == 3


def test_predict_requires_non_empty_reference_list():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")
    image = Image.new("RGB", (8, 8))

    with pytest.raises(TypeError, match="list"):
        predictor.predict(image, {})
    with pytest.raises(ValueError, match="empty"):
        predictor.predict(image, [])
    assert model.image_calls == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"score_thr": -0.1},
        {"score_thr": 1.1},
        {"nms_thr": -0.1},
        {"nms_thr": 1.1},
        {"sim_thr": -1.1},
        {"sim_thr": 1.1},
        {"prompt_batch_size": 0},
        {"prompt_batch_size": -1},
        {"prompt_batch_size": 1.5},
        {"prompt_batch_size": True},
    ],
)
def test_predictor_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        GroundPredictor(FakeGroundModel(), device="cpu", **kwargs)


def test_predictor_removes_old_reference_options():
    predictor = GroundPredictor(FakeGroundModel(), device="cpu")

    assert not hasattr(predictor, "encode_ref")
    with pytest.raises(TypeError):
        GroundPredictor(FakeGroundModel(), device="cpu", score_thresh=0.5)
