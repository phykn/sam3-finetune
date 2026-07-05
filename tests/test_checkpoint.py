import torch
from src.io.checkpoint import remap_model


def test_remap_model_maps_checkpoint_to_shared_model_paths():
    source = {
        "tracker.model.interactivity_no_mem_embed": torch.ones(1, 1, 256),
        "tracker.model.interactive_sam_prompt_encoder.no_mask_embed.weight": torch.zeros(
            1, 256
        ),
        "tracker.model.interactive_sam_mask_decoder.iou_token.weight": torch.zeros(
            1, 256
        ),
        "tracker.model.maskmem_backbone.mask_downsampler.encoder.0.weight": torch.zeros(
            1
        ),
        "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight": torch.zeros(
            1024, 3, 14, 14
        ),
        "detector.transformer.decoder.layers.0.ca_text.in_proj_weight": torch.zeros(
            1, 1
        ),
        "detector.geometry_encoder.norm.weight": torch.zeros(256),
        "detector.segmentation_head.pixel_decoder.conv_layers.0.weight": torch.zeros(1),
        "detector.dot_prod_scoring.prompt_mlp.layers.0.weight": torch.zeros(1),
        "detector.backbone.language_backbone.encoder.token_embedding.weight": torch.zeros(
            1, 1
        ),
    }

    remapped, ignored = remap_model(source)

    assert "video.interactivity_no_mem_embed" in remapped
    assert "video.interactive_sam_prompt_encoder.no_mask_embed.weight" in remapped
    assert "video.interactive_sam_mask_decoder.iou_token.weight" in remapped
    assert "video.maskmem_backbone.mask_downsampler.encoder.0.weight" in remapped
    assert "video.backbone.vision_backbone.trunk.patch_embed.proj.weight" in remapped
    assert "grounding.transformer.decoder.layers.0.ca_text.in_proj_weight" in remapped
    assert "grounding.geometry_encoder.norm.weight" in remapped
    assert "grounding.segmentation_head.pixel_decoder.conv_layers.0.weight" in remapped
    assert "grounding.dot_prod_scoring.prompt_mlp.layers.0.weight" in remapped
    assert (
        "detector.backbone.language_backbone.encoder.token_embedding.weight" in ignored
    )


def test_remap_model_accepts_nested_model_key():
    source = {
        "model": {
            "tracker.model.interactive_sam_prompt_encoder.no_mask_embed.weight": torch.zeros(
                1, 256
            ),
        }
    }

    remapped, ignored = remap_model(source)

    assert ignored == []
    assert list(remapped.keys()) == [
        "video.interactive_sam_prompt_encoder.no_mask_embed.weight"
    ]
