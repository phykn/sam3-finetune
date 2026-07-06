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
    ("video.interactive_sam_mask_decoder.conv_s0.", ("image.sam_image.proj_s0.",)),
    ("video.interactive_sam_mask_decoder.conv_s1.", ("image.sam_image.proj_s1.",)),
    (
        "video.backbone.vision_backbone.",
        ("image.vision.vision_backbone.",),
    ),
    ("video.interactive_sam_prompt_encoder.", ("image.sam_prompt.prompt_encoder.",)),
    ("video.interactive_sam_mask_decoder.", ("image.sam_mask.mask_decoder.",)),
    ("video.interactivity_no_mem_embed", ("image.sam_image.no_mem",)),
    ("grounding.geometry_encoder.", ("ground_prompt.encoder.",)),
    ("grounding.transformer.", ("ground_dec.transformer.",)),
    ("grounding.dot_prod_scoring.", ("ground_dec.scorer.",)),
    ("grounding.segmentation_head.", ("ground_dec.seg_head.",)),
)


def load_pth(path: str | Path) -> Mapping:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def load_visual(path: str | Path) -> Mapping:
    return load_pth(path)


def unwrap_state(ckpt: Mapping):
    model = ckpt.get("model")
    return model if isinstance(model, Mapping) else ckpt


@dataclass(frozen=True)
class Checkpoint:
    state: dict[str, torch.Tensor]
    ignored: list[str]

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_state(load_pth(path))

    @classmethod
    def from_state(cls, ckpt: Mapping):
        state, ignored = remap_model(ckpt)
        return cls(state, ignored)

    def block_state(self, prefix: str):
        prefix = prefix.rstrip(".") + "."
        return {
            key.removeprefix(prefix): val
            for key, val in self.state.items()
            if key.startswith(prefix)
        }


def remap_model(ckpt: Mapping):
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


def _remap_key(key: str):
    if key.startswith(LANG):
        return None

    for src, dst in REMAP:
        if key.startswith(src):
            return dst + key.removeprefix(src)
    return None


def _add_aliases(state, key, val) -> None:
    for src, dsts in ALIAS:
        if not key.startswith(src):
            continue
        tail = key.removeprefix(src)
        for dst in dsts:
            state[dst + tail] = val
