from pathlib import Path
from typing import Mapping

import torch

from ..checkpoint import LoadReport, load_local_checkpoint


def _unwrap_state_dict(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"]
    return checkpoint


def filter_and_remap_video_state_dict(
    checkpoint: Mapping,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    state = _unwrap_state_dict(checkpoint)
    remapped: dict[str, torch.Tensor] = {}
    ignored: list[str] = []

    prefix_map = {
        "tracker.model.": "",
        "detector.backbone.vision_backbone.": "backbone.vision_backbone.",
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


def load_video_weights(
    model: torch.nn.Module,
    path: str | Path,
    strict: bool = False,
) -> LoadReport:
    checkpoint_path = Path(path)
    checkpoint = load_local_checkpoint(checkpoint_path)
    remapped, ignored = filter_and_remap_video_state_dict(checkpoint)
    result = model.load_state_dict(remapped, strict=strict)
    return LoadReport(
        checkpoint_path=checkpoint_path,
        loaded_keys=len(remapped),
        ignored_keys=len(ignored),
        missing_keys=list(result.missing_keys),
        unexpected_keys=list(result.unexpected_keys),
        ignored_key_examples=ignored[:20],
    )
