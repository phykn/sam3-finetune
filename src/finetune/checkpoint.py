from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch import nn

FORMAT = "sam3.finetune.v1"
RESUME_TRAIN_OVERRIDES = {"run_root", "save_every", "steps", "valid_steps"}


def unwrap(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module
    return model


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in unwrap(model).named_parameters()
        if param.requires_grad
    }


def load_trainable_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    expected = {
        name: param
        for name, param in unwrap(model).named_parameters()
        if param.requires_grad
    }
    if set(state) != set(expected):
        raise ValueError("checkpoint trainable keys do not match model")

    with torch.no_grad():
        for name, param in expected.items():
            value = state[name]
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(f"checkpoint shape mismatch: {name}")
            param.copy_(value.to(device=param.device, dtype=param.dtype))


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    checkpoint = {
        "format": FORMAT,
        "step": int(step),
        "model": trainable_state(model),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    try:
        torch.save(checkpoint, temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"format", "step", "model", "optimizer", "config"}
    if not required.issubset(checkpoint):
        raise ValueError("checkpoint fields are incomplete")
    if checkpoint["format"] != FORMAT:
        raise ValueError(f"unsupported checkpoint format: {checkpoint['format']}")
    validate_resume_config(checkpoint["config"], config)
    step = checkpoint["step"]
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    current_train = _config_section(config, "train")
    saved_train = _config_section(checkpoint["config"], "train")
    if "steps" in saved_train or "steps" in current_train:
        steps = current_train.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < step:
            raise ValueError("resume train.steps must include the checkpoint step")
    load_trainable_state(model, checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(step), checkpoint["config"]


def validate_resume_config(
    saved: dict[str, Any],
    current: dict[str, Any],
) -> None:
    if not isinstance(saved, dict) or not isinstance(current, dict):
        raise ValueError("resume configs must be dictionaries")

    saved_model = _config_section(saved, "model")
    current_model = _config_section(current, "model")
    saved_model.pop("device", None)
    current_model.pop("device", None)
    if saved_model != current_model:
        raise ValueError("resume model config does not match checkpoint")

    if _config_section(saved, "data") != _config_section(current, "data"):
        raise ValueError("resume data config does not match checkpoint")

    saved_train = _config_section(saved, "train")
    current_train = _config_section(current, "train")
    for key in RESUME_TRAIN_OVERRIDES:
        saved_train.pop(key, None)
        current_train.pop(key, None)
    if saved_train != current_train:
        raise ValueError("resume train config does not match checkpoint")


def _config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"resume {name} config must be a dictionary")
    return dict(value)
