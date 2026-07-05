import torch
from src.model.build import build_model
from src.model.components.backbone.encoder import ImageEncoder


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


def test_build_model_has_expected_execution_paths():
    model = build_model(device=torch.device("cpu"))

    assert hasattr(model, "image")
    assert hasattr(model, "grounding")
    assert hasattr(model, "video")
    assert hasattr(model.image, "image_encoder")
    assert hasattr(model.image, "prompt_encoder")
    assert hasattr(model.image, "mask_decoder")
    assert hasattr(model.image, "interactivity_no_mem_embed")
    assert model.image.interactivity_no_mem_embed.shape == (1, 1, 256)
    assert model.image.image_size == 1008
    assert model.image.backbone_stride == 14


def test_build_model_shares_backbone_across_execution_paths():
    model = build_model(device=torch.device("cpu"))

    image_backbone = model.image.image_encoder.vision_backbone
    grounding_backbone = model.grounding.backbone.vision_backbone
    video_backbone = model.video.backbone.vision_backbone

    assert image_backbone is grounding_backbone
    assert image_backbone is video_backbone


def test_build_model_shares_interactive_sam_parts():
    model = build_model(device=torch.device("cpu"))

    assert model.image.prompt_encoder is model.video.interactive_sam_prompt_encoder
    assert model.image.mask_decoder is model.video.interactive_sam_mask_decoder
    assert (
        model.image.interactivity_no_mem_embed is model.video.interactivity_no_mem_embed
    )


def test_interactive_image_encoder_adds_interactivity_no_mem_embed():
    encoder = ImageEncoder(_FakeBackbone())
    interactivity_no_mem_embed = torch.full((1, 1, 256), 2.0)

    features = encoder(
        torch.zeros(1, 3, 4, 4),
        _FakeMaskDecoder(),
        interactivity_no_mem_embed=interactivity_no_mem_embed,
    )

    assert torch.all(features["image_embed"] == 2.0)
