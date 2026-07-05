from pathlib import Path

import torch
from src.model.grounding.prompt import Prompt
from src.predict.grounding.inference import GroundingInference
from src.types import VisualLanguageCache


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

    assert GroundingInference.__module__ == "src.predict.grounding.inference"
    assert VisualLanguageCache.__module__ == "src.types"
    assert not (root / "src" / "grounding_predictor.py").exists()


def test_prompt_clone_preserves_mask_prompt():
    masks = torch.ones(1, 2, 1, 3, 4)
    labels = torch.tensor([[1, 0]])
    mask = torch.tensor([[False], [True]])

    prompt = Prompt(mask_embeddings=masks, mask_labels=labels, mask_mask=mask)
    clone = prompt.clone()

    assert torch.equal(clone.mask_embeddings, masks)
    assert torch.equal(clone.mask_labels, labels)
    assert torch.equal(clone.mask_mask, mask)
    assert clone.mask_embeddings.data_ptr() != masks.data_ptr()


def test_prompt_append_initializes_default_masks():
    prompt = Prompt()

    prompt.append_boxes(
        torch.ones(1, 1, 4),
        torch.ones(1, 1, dtype=torch.long),
    )
    prompt.append_points(
        torch.ones(1, 1, 2),
        torch.ones(1, 1, dtype=torch.long),
    )

    assert prompt.box_mask.shape == (1, 1)
    assert prompt.point_mask.shape == (1, 1)
    assert not prompt.box_mask.any()
    assert not prompt.point_mask.any()


def test_grounding_modules_live_under_grounding_package():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "model" / "build.py").is_file()
    assert not (root / "src" / "model" / "grounding" / "builder.py").exists()
    assert (root / "src" / "model" / "grounding" / "create.py").is_file()
    assert (root / "src" / "model" / "grounding" / "backbone.py").is_file()
    assert (root / "src" / "model" / "grounding" / "model.py").is_file()
    assert (root / "src" / "model" / "grounding" / "encoder.py").is_file()
    assert (root / "src" / "model" / "grounding" / "prompt.py").is_file()
    assert (root / "src" / "model" / "grounding" / "mask_encoder.py").is_file()
    assert (root / "src" / "model" / "grounding" / "pixel.py").is_file()
    assert (root / "src" / "model" / "grounding" / "segmentation.py").is_file()
    assert (root / "src" / "types.py").is_file()
    assert not (root / "src" / "model" / "grounding" / "geometry.py").exists()
    assert not (root / "src" / "grounding").exists()
    for filename in (
        "grounding_builder.py",
        "grounding_model.py",
        "geometry_encoders.py",
        "maskformer_segmentation.py",
    ):
        assert not (root / "src" / filename).exists()


def test_grounding_package_exports_user_facing_api():
    import src.predict.grounding as grounding
    from src.types import GroundingPrediction

    assert grounding.GroundingInference is GroundingInference
    assert grounding.VisualLanguageCache is VisualLanguageCache
    assert grounding.GroundingPrediction is GroundingPrediction
    assert not hasattr(grounding, "filter_grounding_prediction")
    assert not hasattr(grounding, "__all__")
