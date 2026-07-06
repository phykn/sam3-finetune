import torch
from src.io.checkpoint import Checkpoint, remap_model


def test_remap_model_maps_checkpoint_to_model_paths():
    conv = torch.ones(32, 256, 1, 1)
    vision = torch.zeros(1024, 3, 14, 14)
    language = "detector.backbone.language_backbone.encoder.weight"
    source = {
        "tracker.model.interactivity_no_mem_embed": torch.ones(1, 1, 256),
        "tracker.model.interactive_sam_prompt_encoder.no_mask_embed.weight": torch.zeros(
            1, 256
        ),
        "tracker.model.interactive_sam_mask_decoder.conv_s0.weight": conv,
        "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight": vision,
        "detector.geometry_encoder.norm.weight": torch.zeros(256),
        language: torch.zeros(1),
    }

    remapped, ignored = remap_model(source)

    assert (
        remapped["image.sam_image.no_mem"]
        is source["tracker.model.interactivity_no_mem_embed"]
    )
    assert (
        remapped["image.sam_prompt.prompt_encoder.no_mask_embed.weight"]
        is source["tracker.model.interactive_sam_prompt_encoder.no_mask_embed.weight"]
    )
    assert remapped["image.sam_image.proj_s0.weight"] is conv
    assert remapped["image.sam_mask.mask_decoder.conv_s0.weight"] is conv
    assert "grounding.geometry_encoder.norm.weight" in remapped
    assert (
        remapped["ground_prompt.encoder.norm.weight"]
        is source["detector.geometry_encoder.norm.weight"]
    )
    assert ignored == [language]


def test_checkpoint_bank_exposes_block_state():
    value = torch.ones(1, 256)
    ckpt = Checkpoint.from_state(
        {
            "model": {
                "tracker.model.interactive_sam_mask_decoder.iou_token.weight": value
            }
        }
    )

    assert ckpt.block_state("image.sam_mask") == {
        "mask_decoder.iou_token.weight": value
    }
