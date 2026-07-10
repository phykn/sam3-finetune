import pytest
import torch
from torch import nn

from src.finetune.model import FinetuneModel


class FakeImageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.keep = nn.Linear(3, 3)
        self.sam_mask = nn.Module()
        self.sam_mask.mask_decoder = nn.Module()
        self.sam_mask.mask_decoder.transformer = nn.Module()
        self.sam_mask.mask_decoder.transformer.q_proj = nn.Linear(256, 256)

    @property
    def mask_input_size(self):
        return (288, 288)

    def get_image_position_encoding(self, device=None):
        value = torch.zeros(1, 256, 2, 2)
        return value if device is None else value.to(device)

    def decode_masks(
        self,
        image_embed,
        high_res_features,
        prompt,
        image_pe,
        multimask=True,
        repeat_image=False,
        mix=None,
    ):
        batch = prompt[0].shape[0]
        count = 3 if multimask else 1
        return (
            torch.ones(batch, count, 8, 8),
            torch.ones(batch, count),
            torch.ones(batch, count, 256),
            torch.ones(batch, 1),
        )


def make_model(num_classes=3):
    base = FakeImageModel()
    model = FinetuneModel(
        base,
        num_conditions=2,
        num_experts=2,
        num_labels=num_classes,
        lora_rank=2,
        feature_rank=2,
    )
    return model, base


def test_decode_returns_one_class_vector_per_mask():
    model, base = make_model(num_classes=3)
    prompt = (
        torch.zeros(1, 1, 256),
        torch.zeros(1, 256, 2, 2),
    )

    out = model.decode_masks(
        torch.ones(1, 256, 2, 2),
        (torch.ones(1, 32, 8, 8), torch.ones(1, 64, 4, 4)),
        prompt,
        base.get_image_position_encoding(),
        multimask=True,
        cond=0,
        prompt_type="point",
    )

    assert out[0].shape[1] == 3
    assert out[4].shape == (1, 3, 3)


def test_train_keeps_frozen_base_in_eval_mode():
    model, base = make_model(num_classes=2)

    model.train()

    assert model.training is True
    assert base.training is False
    assert model.router.training is True
    assert model.class_head.training is True


def test_lora_modules_are_not_registered_twice():
    model, _base = make_model(num_classes=2)

    assert not hasattr(model, "linear_layers")
    names = [name for name, _param in model.named_parameters() if ".down." in name]
    assert names
    assert all(not name.startswith("linear_layers.") for name in names)


def test_build_prompt_merges_box_and_point_inputs():
    from src.finetune.prompt import build_prompt

    points, mask = build_prompt(
        {
            "points": [[4.0, 4.0]],
            "point_labels": [1],
            "box": [1.0, 1.0, 3.0, 3.0],
            "mask": None,
        },
        image_size=8,
        mask_size=(2, 2),
        device=torch.device("cpu"),
    )

    assert points[0].shape == (1, 3, 2)
    assert points[1].tolist() == [[2, 3, 1]]
    assert mask is None


def test_model_rejects_empty_class_head():
    with pytest.raises(ValueError, match="num_labels"):
        make_model(num_classes=0)


def test_decode_rejects_condition_outside_router_range():
    model, base = make_model(num_classes=2)
    prompt = (
        torch.zeros(1, 1, 256),
        torch.zeros(1, 256, 2, 2),
    )

    with pytest.raises(ValueError, match="condition"):
        model.decode_masks(
            torch.ones(1, 256, 2, 2),
            (torch.ones(1, 32, 8, 8), torch.ones(1, 64, 4, 4)),
            prompt,
            base.get_image_position_encoding(),
            multimask=False,
            cond=2,
            prompt_type="point",
        )
