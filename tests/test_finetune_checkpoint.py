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
    step, config = load_checkpoint(
        path,
        model,
        optimizer,
        {"train": {"steps": 12}},
    )

    assert step == 7
    assert config == {"train": {"steps": 9}}
    assert "frozen" not in trainable_state(model)
    for name, value in trainable_state(model).items():
        assert torch.equal(value, expected[name])
    assert optimizer.state_dict()["state"].keys() == expected_optimizer["state"].keys()
    assert not Path(f"{path}.tmp").exists()


def test_checkpoint_allows_runtime_resume_overrides(tmp_path):
    model = TinyModel()
    optimizer = make_optimizer(model)
    path = tmp_path / "last.pt"
    saved = {
        "model": {"path": "base.pt", "device": "cuda", "num_conditions": 2},
        "data": {"train": {"paths": ["sample.json"]}},
        "train": {
            "steps": 10,
            "valid_steps": 2,
            "save_every": 5,
            "run_root": "old-run",
            "learning_rate": 1e-3,
        },
    }
    save_checkpoint(path, model, optimizer, step=4, config=saved)
    current = {
        "model": {"path": "base.pt", "device": "cpu", "num_conditions": 2},
        "data": {"train": {"paths": ["sample.json"]}},
        "train": {
            "steps": 20,
            "valid_steps": 3,
            "save_every": 7,
            "run_root": "new-run",
            "learning_rate": 1e-3,
        },
    }

    step, _config = load_checkpoint(path, model, optimizer, current)

    assert step == 4


def test_checkpoint_rejects_steps_before_state_restore(tmp_path):
    model = TinyModel()
    optimizer = make_optimizer(model)
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

    with pytest.raises(ValueError, match="checkpoint step"):
        load_checkpoint(
            path,
            model,
            optimizer,
            {"train": {"steps": 6}},
        )

    assert all(
        torch.count_nonzero(param) == 0
        for param in model.parameters()
        if param.requires_grad
    )


@pytest.mark.parametrize(
    ("section", "change", "message"),
    [
        ("model", {"num_conditions": 3}, "model config"),
        ("data", {"train": {"paths": ["other.json"]}}, "data config"),
        ("train", {"learning_rate": 2e-3}, "train config"),
    ],
)
def test_checkpoint_rejects_resume_contract_mismatch(
    tmp_path,
    section,
    change,
    message,
):
    model = TinyModel()
    optimizer = make_optimizer(model)
    path = tmp_path / "last.pt"
    saved = {
        "model": {"path": "base.pt", "num_conditions": 2},
        "data": {"train": {"paths": ["sample.json"]}},
        "train": {"steps": 10, "learning_rate": 1e-3},
    }
    save_checkpoint(path, model, optimizer, step=4, config=saved)
    current = {
        "model": dict(saved["model"]),
        "data": dict(saved["data"]),
        "train": dict(saved["train"]),
    }
    current[section].update(change)

    with pytest.raises(ValueError, match=message):
        load_checkpoint(path, model, optimizer, current)


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
        load_checkpoint(path, model, optimizer, {})


def test_checkpoint_rejects_extra_trainable_key(tmp_path):
    def add_key(data):
        data["model"]["extra"] = torch.ones(1)

    path, model, optimizer = write_bad_checkpoint(tmp_path, add_key)
    with pytest.raises(ValueError, match="trainable keys"):
        load_checkpoint(path, model, optimizer, {})


def test_checkpoint_rejects_wrong_shape(tmp_path):
    def change_shape(data):
        name = next(iter(data["model"]))
        data["model"][name] = torch.ones(9)

    path, model, optimizer = write_bad_checkpoint(tmp_path, change_shape)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_checkpoint(path, model, optimizer, {})


def test_checkpoint_rejects_unsupported_format(tmp_path):
    def change_format(data):
        data["format"] = "sam3.finetune.old"

    path, model, optimizer = write_bad_checkpoint(tmp_path, change_format)
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(path, model, optimizer, {})
