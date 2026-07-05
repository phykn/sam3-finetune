from collections.abc import Mapping
from pathlib import Path

import torch


def load_pth(path: str | Path) -> Mapping:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    return torch.load(path, map_location="cpu", weights_only=True)


def unwrap_state(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"]
    return checkpoint


def remap_model(checkpoint: Mapping) -> tuple[dict[str, torch.Tensor], list[str]]:
    state = unwrap_state(checkpoint)
    remapped: dict[str, torch.Tensor] = {}
    ignored: list[str] = []

    prefix_map = {
        "detector.backbone.vision_backbone.": "video.backbone.vision_backbone.",
        "tracker.model.": "video.",
    }

    for key, value in state.items():
        if key.startswith("detector.backbone.language_backbone."):
            ignored.append(key)
            continue

        target_key = None
        for source_prefix, local_prefix in prefix_map.items():
            if key.startswith(source_prefix):
                target_key = local_prefix + key[len(source_prefix) :]
                break
        if target_key is None and key.startswith("detector."):
            target_key = "grounding." + key[len("detector.") :]
        if target_key is None:
            ignored.append(key)
            continue
        remapped[target_key] = value

    return remapped, ignored
