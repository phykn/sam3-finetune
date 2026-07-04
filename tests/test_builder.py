import torch

from src.builder import build_model


def test_build_model_has_expected_submodules():
    model = build_model(device=torch.device("cpu"))

    assert hasattr(model, "image_encoder")
    assert hasattr(model, "prompt_encoder")
    assert hasattr(model, "mask_decoder")
    assert model.image_size == 1008
    assert model.backbone_stride == 14
