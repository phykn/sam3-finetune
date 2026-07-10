import pytest
import torch
from torch import nn

from src.io.checkpoint import Checkpoint, remap_model


def test_remap_model_maps_official_keys_to_logical_blocks():
    conv = torch.ones(32, 256, 1, 1)
    vision = torch.zeros(1024, 3, 14, 14)
    language = "detector.backbone.language_backbone.encoder.weight"
    source = {
        "tracker.model.interactivity_no_mem_embed": torch.ones(1, 1, 256),
        "tracker.model.interactive_sam_prompt_encoder.no_mask_embed.weight": (
            torch.zeros(1, 256)
        ),
        "tracker.model.interactive_sam_mask_decoder.conv_s0.weight": conv,
        "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight": vision,
        "detector.geometry_encoder.norm.weight": torch.zeros(256),
        language: torch.zeros(1),
    }

    remapped, ignored = remap_model(source)

    assert (
        remapped["image.features.no_mem"]
        is source["tracker.model.interactivity_no_mem_embed"]
    )
    assert (
        remapped["image.prompt.prompt_encoder.no_mask_embed.weight"]
        is source["tracker.model.interactive_sam_prompt_encoder.no_mask_embed.weight"]
    )
    assert remapped["image.features.proj_s0.weight"] is conv
    assert remapped["image.masks.mask_decoder.conv_s0.weight"] is conv
    assert (
        remapped["grounding.prompt.encoder.norm.weight"]
        is source["detector.geometry_encoder.norm.weight"]
    )
    assert ignored == [language]


def test_checkpoint_exposes_logical_block_state():
    value = torch.ones(1, 256)
    checkpoint = Checkpoint.from_state(
        {
            "model": {
                "tracker.model.interactive_sam_mask_decoder.iou_token.weight": value
            }
        }
    )

    assert checkpoint.block_state("image.masks") == {
        "mask_decoder.iou_token.weight": value
    }


def test_checkpoint_load_block_is_strict_and_names_the_block():
    module = nn.Linear(1, 1, bias=False)
    checkpoint = Checkpoint(
        state={"image.features.weight": torch.ones(1, 1)},
        ignored=[],
    )

    checkpoint.load_block("image.features", module)

    assert module.weight.item() == 1
    with pytest.raises(RuntimeError, match="image.features"):
        Checkpoint(state={}, ignored=[]).load_block("image.features", module)


def test_lora_keys_are_not_mapped_into_base_blocks():
    state, ignored = remap_model({"lora.image.q_proj.weight": torch.ones(1)})

    assert state == {}
    assert ignored == ["lora.image.q_proj.weight"]


def test_checkpoint_load_rejects_unknown_checkpoint_keys(tmp_path):
    path = tmp_path / "adapter.pt"
    torch.save({"model": {"lora.image.q_proj.weight": torch.ones(1)}}, path)

    with pytest.raises(RuntimeError, match="unsupported checkpoint key"):
        Checkpoint.load(path)
