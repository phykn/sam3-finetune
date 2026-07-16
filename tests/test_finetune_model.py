import copy

import pytest
import torch
from torch import nn

from src.finetune.loss import finetune_loss
from src.finetune.model import FinetuneModel


class FakeImageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decode_calls = 0
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

    def encode_image(self, image):
        batch = image.shape[0]
        return {
            "image_embed": torch.ones(batch, 256, 2, 2, device=image.device),
            "high_res_features": (
                torch.ones(batch, 32, 8, 8, device=image.device),
                torch.ones(batch, 64, 4, 4, device=image.device),
            ),
        }

    def encode_prompt(self, points=None, boxes=None, masks=None):
        batch = points[0].shape[0]
        return (
            torch.zeros(batch, 1, 256, device=points[0].device),
            torch.zeros(batch, 256, 2, 2, device=points[0].device),
        )

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
        self.decode_calls += 1
        count = 3 if multimask else 1
        pooled = image_embed.mean(dim=(2, 3))
        token = self.sam_mask.mask_decoder.transformer.q_proj(pooled, mix)
        token = token + high_res_features[0].mean(dim=(1, 2, 3))[:, None]
        token = token + high_res_features[1].mean(dim=(1, 2, 3))[:, None]
        tokens = token[:, None].expand(-1, count, -1)
        masks = tokens[:, :, :1, None].expand(-1, -1, 8, 8)
        return masks, tokens.mean(-1), tokens, tokens[:, :1, 0]


def make_model(num_classes=3):
    base = FakeImageModel()
    model = FinetuneModel(
        base,
        num_conditions=2,
        num_experts=2,
        num_classes=num_classes,
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
    with pytest.raises(ValueError, match="num_classes"):
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


def test_batch_forward_loss_backward_reaches_every_trainable_parameter():
    model, _base = make_model(num_classes=2)
    batch = {
        "image": torch.zeros(2, 3, 8, 8),
        "cond": torch.tensor([0, 1]),
        "prompt": [
            {
                "type": "point",
                "points": [[2.0, 2.0]],
                "point_labels": [1],
                "box": None,
                "mask": None,
            },
            {
                "type": "point",
                "points": [[6.0, 6.0]],
                "point_labels": [1],
                "box": None,
                "mask": None,
            },
        ],
        "target": torch.tensor(
            [
                [[[1.0] * 8] * 8],
                [[[0.0] * 8] * 8],
            ]
        ),
        "mask_valid": torch.tensor([1.0, 0.0]),
        "is_auto_bg": torch.tensor([0.0, 0.0]),
        "label_target": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        "label_weight": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
    }

    loss, _stats = finetune_loss(batch, model(batch))
    loss.backward()

    gradients = [param.grad for param in model.parameters() if param.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_forward_batches_prompts_by_type():
    model, base = make_model(num_classes=2)
    reference = copy.deepcopy(model)
    batch = {
        "image": torch.zeros(4, 3, 8, 8),
        "cond": torch.tensor([0, 1, 0, 0]),
        "prompt": [
            {
                "type": "point",
                "points": [[2.0, 2.0]],
                "point_labels": [1],
                "box": None,
                "mask": None,
            },
            {
                "type": "box",
                "points": None,
                "point_labels": None,
                "box": [1.0, 1.0, 6.0, 6.0],
                "mask": None,
            },
            {
                "type": "mask",
                "points": None,
                "point_labels": None,
                "box": None,
                "mask": [[1.0, 0.0], [0.0, 1.0]],
            },
            {
                "type": "point",
                "points": [[4.0, 4.0]],
                "point_labels": [1],
                "box": None,
                "mask": None,
            },
        ],
    }

    out = model(batch)
    single = [
        reference(
            {
                "image": batch["image"][index : index + 1],
                "cond": batch["cond"][index : index + 1],
                "prompt": [batch["prompt"][index]],
            }
        )
        for index in range(4)
    ]

    assert base.decode_calls == 3
    assert out["mask_logits"].shape[0] == 4
    assert out["class_logits"].shape == (4, 1, 2)
    for key in out:
        expected = torch.cat([item[key] for item in single])
        assert torch.allclose(out[key], expected)
