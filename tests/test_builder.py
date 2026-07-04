import torch

from src.builder import build_model
from src.backbone.image_encoder import InteractiveImageEncoder


class _FakeBackbone(torch.nn.Module):
    def forward(
        self,
        images,
        *,
        need_sam3_out,
        need_interactive_out,
        need_propagation_out,
    ):
        feature_maps = [
            type("Feature", (), {"tensors": torch.zeros(1, 256, 4, 4)})(),
            type("Feature", (), {"tensors": torch.zeros(1, 256, 2, 2)})(),
            type("Feature", (), {"tensors": torch.zeros(1, 256, 1, 1)})(),
        ]
        return None, None, feature_maps, None, None, None


class _FakeMaskDecoder(torch.nn.Module):
    conv_s0 = torch.nn.Identity()
    conv_s1 = torch.nn.Identity()


def test_build_model_has_expected_submodules():
    model = build_model(device=torch.device("cpu"))

    assert hasattr(model, "image_encoder")
    assert hasattr(model, "prompt_encoder")
    assert hasattr(model, "mask_decoder")
    assert hasattr(model, "interactivity_no_mem_embed")
    assert model.interactivity_no_mem_embed.shape == (1, 1, 256)
    assert model.image_size == 1008
    assert model.backbone_stride == 14


def test_interactive_image_encoder_adds_interactivity_no_mem_embed():
    encoder = InteractiveImageEncoder(_FakeBackbone())
    interactivity_no_mem_embed = torch.full((1, 1, 256), 2.0)

    features = encoder(
        torch.zeros(1, 3, 4, 4),
        _FakeMaskDecoder(),
        interactivity_no_mem_embed=interactivity_no_mem_embed,
    )

    assert torch.all(features["image_embed"] == 2.0)
