from pathlib import Path

import pytest
import torch
from torch import nn

from src.finetune.checkpoint import (
    FORMAT,
    load_checkpoint,
    save_checkpoint,
    trainable_state,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Parameter(torch.tensor(3.0), requires_grad=False)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Linear(1, 1)

    def forward(self, value):
        return self.head(value) * self.scale


def make_optimizer(model):
    return torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=1e-3,
    )


def train_once(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()


def test_checkpoint_restores_trainable_state_optimizer_and_step(tmp_path):
    model = TinyModel()
    optimizer = make_optimizer(model)
    train_once(model, optimizer)
    expected = {name: value.clone() for name, value in trainable_state(model).items()}
    expected_optimizer = optimizer.state_dict()
    path = tmp_path / "last.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        step=7,
        config={"train": {"steps": 9}},
    )
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.zero_()
    optimizer = make_optimizer(model)
    step, config = load_checkpoint(path, model, optimizer)

    assert step == 7
    assert config == {"train": {"steps": 9}}
    assert "frozen" not in trainable_state(model)
    for name, value in trainable_state(model).items():
        assert torch.equal(value, expected[name])
    assert optimizer.state_dict()["state"].keys() == expected_optimizer["state"].keys()
    assert not Path(f"{path}.tmp").exists()


def write_bad_checkpoint(tmp_path, change):
    model = TinyModel()
    optimizer = make_optimizer(model)
    path = tmp_path / "bad.pt"
    data = {
        "format": FORMAT,
        "step": 1,
        "model": trainable_state(model),
        "optimizer": optimizer.state_dict(),
        "config": {},
    }
    change(data)
    torch.save(data, path)
    return path, model, optimizer


def test_checkpoint_rejects_missing_trainable_key(tmp_path):
    def remove_key(data):
        data["model"].pop(next(iter(data["model"])))

    path, model, optimizer = write_bad_checkpoint(tmp_path, remove_key)
    with pytest.raises(ValueError, match="trainable keys"):
        load_checkpoint(path, model, optimizer)


def test_checkpoint_rejects_extra_trainable_key(tmp_path):
    def add_key(data):
        data["model"]["extra"] = torch.ones(1)

    path, model, optimizer = write_bad_checkpoint(tmp_path, add_key)
    with pytest.raises(ValueError, match="trainable keys"):
        load_checkpoint(path, model, optimizer)


def test_checkpoint_rejects_wrong_shape(tmp_path):
    def change_shape(data):
        name = next(iter(data["model"]))
        data["model"][name] = torch.ones(9)

    path, model, optimizer = write_bad_checkpoint(tmp_path, change_shape)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_checkpoint(path, model, optimizer)


def test_checkpoint_rejects_unsupported_format(tmp_path):
    def change_format(data):
        data["format"] = "sam3.finetune.old"

    path, model, optimizer = write_bad_checkpoint(tmp_path, change_format)
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(path, model, optimizer)
