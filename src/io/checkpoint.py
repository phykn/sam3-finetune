from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

LANG = "detector.backbone.language_backbone."

REMAP = (
    ("detector.backbone.vision_backbone.", "video.backbone.vision_backbone."),
    ("tracker.model.", "video."),
    ("detector.", "grounding."),
)

ALIAS = (
    ("video.interactive_sam_mask_decoder.conv_s0.", ("image.features.proj_s0.",)),
    ("video.interactive_sam_mask_decoder.conv_s1.", ("image.features.proj_s1.",)),
    (
        "video.backbone.vision_backbone.",
        ("image.vision.vision_backbone.",),
    ),
    ("video.interactive_sam_prompt_encoder.", ("image.prompt.prompt_encoder.",)),
    ("video.interactive_sam_mask_decoder.", ("image.masks.mask_decoder.",)),
    ("video.interactivity_no_mem_embed", ("image.features.no_mem",)),
    ("grounding.geometry_encoder.", ("grounding.prompt.encoder.",)),
    ("grounding.transformer.", ("grounding.decoder.transformer.",)),
    ("grounding.dot_prod_scoring.", ("grounding.decoder.scorer.",)),
    ("grounding.segmentation_head.", ("grounding.decoder.seg_head.",)),
)


def load_pth(path: str | Path) -> Mapping:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def load_visual(path: str | Path) -> Mapping:
    return load_pth(path)


def unwrap_state(ckpt: Mapping) -> Mapping:
    model = ckpt.get("model")
    return model if isinstance(model, Mapping) else ckpt


@dataclass(frozen=True)
class Checkpoint:
    state: dict[str, torch.Tensor]
    ignored: list[str]

    @classmethod
    def load(cls, path: str | Path) -> "Checkpoint":
        checkpoint = cls.from_state(load_pth(path))
        unsupported = [key for key in checkpoint.ignored if not key.startswith(LANG)]
        if unsupported:
            raise RuntimeError(f"unsupported checkpoint key: {unsupported[0]}")
        return checkpoint

    @classmethod
    def from_state(cls, ckpt: Mapping) -> "Checkpoint":
        state, ignored = remap_model(ckpt)
        return cls(state, ignored)

    def block_state(self, prefix: str) -> dict[str, torch.Tensor]:
        prefix = prefix.rstrip(".") + "."
        return {
            key.removeprefix(prefix): val
            for key, val in self.state.items()
            if key.startswith(prefix)
        }

    def load_block(self, name: str, module: torch.nn.Module) -> None:
        state = self.block_state(name)
        if not state:
            raise RuntimeError(f"SAM3.1 checkpoint block is empty: {name}")
        try:
            module.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                f"failed to load SAM3.1 block {name}: {error}"
            ) from error


def remap_model(ckpt: Mapping) -> tuple[dict[str, torch.Tensor], list[str]]:
    state = {}
    ignored = []

    for key, val in unwrap_state(ckpt).items():
        new_key = _remap_key(key)
        if new_key is None:
            ignored.append(key)
            continue

        state[new_key] = val
        _add_aliases(state, new_key, val)

    return state, ignored


def _remap_key(key: str) -> str | None:
    if key.startswith(LANG):
        return None

    for src, dst in REMAP:
        if key.startswith(src):
            return dst + key.removeprefix(src)
    return None


def _add_aliases(state: dict[str, torch.Tensor], key: str, val: torch.Tensor) -> None:
    for src, dsts in ALIAS:
        if not key.startswith(src):
            continue
        tail = key.removeprefix(src)
        for dst in dsts:
            state[dst + tail] = val
