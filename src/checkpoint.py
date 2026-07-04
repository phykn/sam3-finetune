from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


@dataclass(frozen=True)
class LoadReport:
    checkpoint_path: Path
    loaded_keys: int
    ignored_keys: int
    missing_keys: list[str]
    unexpected_keys: list[str]
    ignored_key_examples: list[str]


def _unwrap_state_dict(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"]
    return checkpoint


def filter_and_remap_state_dict(
    checkpoint: Mapping,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    state = _unwrap_state_dict(checkpoint)
    remapped: dict[str, torch.Tensor] = {}
    ignored: list[str] = []

    prefix_map = {
        "tracker.model.interactivity_no_mem_embed": "interactivity_no_mem_embed",
        "tracker.model.interactive_sam_prompt_encoder.": "prompt_encoder.",
        "tracker.model.interactive_sam_mask_decoder.": "mask_decoder.",
        "detector.backbone.vision_backbone.": "image_encoder.vision_backbone.",
    }

    for key, value in state.items():
        target_key = None
        for source_prefix, local_prefix in prefix_map.items():
            if key.startswith(source_prefix):
                target_key = local_prefix + key[len(source_prefix) :]
                break
        if target_key is None:
            ignored.append(key)
            continue
        remapped[target_key] = value

    return remapped, ignored


def load_local_checkpoint(path: str | Path) -> Mapping:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def load_weights(
    model: torch.nn.Module,
    path: str | Path,
    strict: bool = False,
) -> LoadReport:
    checkpoint_path = Path(path)
    checkpoint = load_local_checkpoint(checkpoint_path)
    remapped, ignored = filter_and_remap_state_dict(checkpoint)
    result = model.load_state_dict(remapped, strict=strict)
    return LoadReport(
        checkpoint_path=checkpoint_path,
        loaded_keys=len(remapped),
        ignored_keys=len(ignored),
        missing_keys=list(result.missing_keys),
        unexpected_keys=list(result.unexpected_keys),
        ignored_key_examples=ignored[:20],
    )
