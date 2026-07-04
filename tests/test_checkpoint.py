import torch
from src.checkpoint import filter_and_remap_state_dict


def test_filter_and_remap_state_dict_keeps_interactive_prompt_decoder_and_backbone():
    source = {
        "tracker.model.interactive_sam_prompt_encoder.point_embeddings.0.weight": torch.zeros(
            1, 256
        ),
        "tracker.model.interactive_sam_mask_decoder.iou_token.weight": torch.zeros(
            1, 256
        ),
        "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight": torch.zeros(
            1024, 3, 14, 14
        ),
        "detector.backbone.language_backbone.encoder.token_embedding.weight": torch.zeros(
            1, 1
        ),
        "detector.transformer.decoder.layers.0.ca_text.in_proj_weight": torch.zeros(
            1, 1
        ),
    }

    remapped, ignored = filter_and_remap_state_dict(source)

    assert "prompt_encoder.point_embeddings.0.weight" in remapped
    assert "mask_decoder.iou_token.weight" in remapped
    assert "image_encoder.vision_backbone.trunk.patch_embed.proj.weight" in remapped
    assert (
        "detector.backbone.language_backbone.encoder.token_embedding.weight" in ignored
    )
    assert "detector.transformer.decoder.layers.0.ca_text.in_proj_weight" in ignored


def test_filter_and_remap_state_dict_accepts_nested_model_key():
    source = {
        "model": {
            "tracker.model.interactive_sam_prompt_encoder.no_mask_embed.weight": torch.zeros(
                1, 256
            ),
        }
    }

    remapped, ignored = filter_and_remap_state_dict(source)

    assert ignored == []
    assert list(remapped.keys()) == ["prompt_encoder.no_mask_embed.weight"]


def test_filter_and_remap_state_dict_keeps_interactivity_no_mem_embed():
    source = {
        "tracker.model.interactivity_no_mem_embed": torch.ones(1, 1, 256),
    }

    remapped, ignored = filter_and_remap_state_dict(source)

    assert ignored == []
    assert list(remapped.keys()) == ["interactivity_no_mem_embed"]
