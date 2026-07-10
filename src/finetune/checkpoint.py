from pathlib import Path
from typing import Any

import torch
from torch import nn

FORMAT = "sam3.finetune.v1"


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
) -> tuple[int, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"format", "step", "model", "optimizer", "config"}
    if not required.issubset(checkpoint):
        raise ValueError("checkpoint fields are incomplete")
    if checkpoint["format"] != FORMAT:
        raise ValueError(f"unsupported checkpoint format: {checkpoint['format']}")
    load_trainable_state(model, checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"]), checkpoint["config"]
