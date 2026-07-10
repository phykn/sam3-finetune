import numpy as np
import pytest
import torch
from torch import nn

from src.data import ground
from src.ml.blocks.grounding.decoder import GroundingDecoder
from src.ml.blocks.grounding.image import GroundingImage
from src.ml.model import Sam3GroundingModel
from src.ml.structures import NestedTensor


def make_image(batch=1):
    return {
        "vision_features": torch.ones(batch, 2, 3, 3),
        "vision_mask": None,
        "vision_pos_enc": (torch.ones(batch, 2, 3, 3),),
        "backbone_fpn": (
            NestedTensor(
                torch.ones(batch, 2, 3, 3),
                torch.zeros(batch, 3, 3, dtype=torch.bool),
            ),
        ),
        "feat_sizes": ((3, 3),),
    }


def test_build_box_batch_pads_class_prompts():
    groups = [
        np.array([[0, 0, 2, 2]], dtype=np.float32),
        np.array([[0, 0, 2, 2], [2, 2, 4, 4]], dtype=np.float32),
    ]

    boxes, labels, box_mask = ground.build_box_batch(groups, (4, 4), "cpu")

    assert boxes.shape == (2, 2, 4)
    assert labels.shape == (2, 2)
    assert box_mask.shape == (2, 2)
    assert box_mask.tolist() == [[False, True], [False, False]]
    torch.testing.assert_close(boxes[0, 0], torch.tensor([0.25, 0.25, 0.5, 0.5]))
    torch.testing.assert_close(boxes[1, 1], torch.tensor([0.75, 0.75, 0.5, 0.5]))


def test_grounding_image_expands_encoded_batch():
    image = make_image()

    out = GroundingImage.expand(image, 4)

    assert out["vision_features"].shape[0] == 4
    assert out["vision_mask"] is None
    assert out["vision_pos_enc"][0].shape[0] == 4
    assert out["backbone_fpn"][0].tensors.shape[0] == 4
    assert out["backbone_fpn"][0].mask.shape[0] == 4
    assert out["feat_sizes"] == ((3, 3),)


def test_grounding_image_keeps_matching_batch_and_rejects_other_batch():
    image = make_image(batch=2)

    assert GroundingImage.expand(image, 2) is image
    with pytest.raises(ValueError, match="batch"):
        GroundingImage.expand(image, 3)


def test_prompt_inputs_expands_cached_visual_tokens():
    cond = {
        "language_features": torch.ones(2, 1, 4),
        "language_mask": torch.zeros(1, 2, dtype=torch.bool),
    }
    prompt = {
        "features": torch.zeros(3, 3, 4),
        "mask": torch.zeros(3, 3, dtype=torch.bool),
    }

    features, mask = GroundingDecoder.prompt_inputs(cond, prompt)

    assert features.shape == (5, 3, 4)
    assert mask.shape == (3, 5)
    torch.testing.assert_close(features[:2], torch.ones(2, 3, 4))


class FakePrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image, **kwargs):
        self.calls.append((image, kwargs))
        batch = image["vision_features"].shape[0]
        return {
            "features": torch.zeros(3, batch, 2),
            "mask": torch.zeros(batch, 3, dtype=torch.bool),
        }


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, image, _cond, _prompt):
        self.batches.append(image["vision_features"].shape[0])
        return {"batch": self.batches[-1]}


def make_model():
    model = Sam3GroundingModel.__new__(Sam3GroundingModel)
    nn.Module.__init__(model)
    model.ground_prompt = FakePrompt()
    model.ground_dec = FakeDecoder()
    model.cond = lambda: {"language": "cached"}
    return model


def test_model_encodes_box_prompt_batch_against_one_reference_image():
    model = make_model()
    boxes = torch.zeros(2, 3, 4)
    labels = torch.ones(2, 3, dtype=torch.long)
    box_mask = torch.zeros(3, 2, dtype=torch.bool)

    out = model.encode_box_prompts(make_image(), boxes, labels, box_mask)

    image, kwargs = model.ground_prompt.calls[0]
    assert image["vision_features"].shape[0] == 3
    assert kwargs == {
        "boxes": boxes,
        "box_labels": labels,
        "box_mask": box_mask,
    }
    assert out["features"].shape[1] == 3


def test_model_decodes_prompt_batch_against_one_target_image():
    model = make_model()
    prompt = {
        "features": torch.zeros(3, 4, 2),
        "mask": torch.zeros(4, 3, dtype=torch.bool),
    }

    out = model.decode(make_image(), prompt)

    assert out == {"batch": 4}
    assert model.ground_dec.batches == [4]
