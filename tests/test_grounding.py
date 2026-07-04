from pathlib import Path

import torch

from src.grounding.builder import filter_and_remap_grounding_state_dict
from src.grounding.cache import VisualLanguageCache
from src.grounding.inference import GroundingInference


def test_filter_and_remap_grounding_state_dict_drops_language_backbone_only():
    source = {
        "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight": torch.zeros(
            1
        ),
        "detector.backbone.language_backbone.encoder.token_embedding.weight": torch.zeros(
            1
        ),
        "detector.transformer.decoder.layers.0.ca_text.in_proj_weight": torch.zeros(1),
        "tracker.model.interactive_sam_mask_decoder.iou_token.weight": torch.zeros(1),
    }

    remapped, ignored = filter_and_remap_grounding_state_dict(source)

    assert "backbone.vision_backbone.trunk.patch_embed.proj.weight" in remapped
    assert "transformer.decoder.layers.0.ca_text.in_proj_weight" in remapped
    assert "detector.backbone.language_backbone.encoder.token_embedding.weight" in ignored
    assert "tracker.model.interactive_sam_mask_decoder.iou_token.weight" in ignored


def test_visual_language_cache_loads_and_moves_tensors(tmp_path: Path):
    cache_path = tmp_path / "visual_cache.pt"
    torch.save(
        {
            "language_features": torch.ones(2, 1, 4, dtype=torch.float32),
            "language_mask": torch.zeros(1, 2, dtype=torch.bool),
            "language_embeds": torch.full((2, 1, 8), 2.0, dtype=torch.float32),
        },
        cache_path,
    )

    cache = VisualLanguageCache.from_file(cache_path)
    out = cache.to_backbone_out(device=torch.device("cpu"), dtype=torch.float16)

    assert out["language_features"].dtype == torch.float16
    assert out["language_features"].shape == (2, 1, 4)
    assert out["language_mask"].dtype == torch.bool
    assert out["language_embeds"].dtype == torch.float16


def test_grounding_inference_is_not_a_predictor_entrypoint():
    root = Path(__file__).resolve().parents[1]

    assert GroundingInference.__module__ == "src.grounding.inference"
    assert VisualLanguageCache.__module__ == "src.grounding.cache"
    assert not (root / "src" / "grounding_predictor.py").exists()


def test_grounding_modules_live_under_grounding_package():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "grounding" / "builder.py").is_file()
    assert (root / "src" / "grounding" / "model.py").is_file()
    assert (root / "src" / "grounding" / "geometry.py").is_file()
    assert (root / "src" / "grounding" / "segmentation.py").is_file()
    assert (root / "src" / "grounding" / "cache.py").is_file()
    assert (root / "src" / "grounding" / "types.py").is_file()
    for filename in (
        "grounding_builder.py",
        "grounding_model.py",
        "geometry_encoders.py",
        "maskformer_segmentation.py",
    ):
        assert not (root / "src" / filename).exists()


def test_grounding_package_does_not_reexport_internal_api():
    import src.grounding as grounding

    for name in (
        "GroundingInference",
        "GroundingPrediction",
        "VisualLanguageCache",
        "filter_grounding_prediction",
    ):
        assert not hasattr(grounding, name)
