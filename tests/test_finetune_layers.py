import torch
from torch import nn

from src.finetune.adapter import FeatureAdapter, LoraLinear
from src.finetune.router import Router
from src.finetune.model import FinetuneModel
from src.ml.components.sam.transformer import Attention


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.keep = nn.Linear(3, 3)
        self.sam_mask = nn.Module()
        self.sam_mask.mask_decoder = nn.Module()
        self.sam_mask.mask_decoder.transformer = nn.Module()
        self.sam_mask.mask_decoder.transformer.q_proj = nn.Linear(3, 3)
        self.sam_mask.mask_decoder.transformer.lin1 = nn.Linear(3, 4)


class FakeImageApiModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.keep = nn.Linear(3, 3)
        self.sam_mask = nn.Module()
        self.sam_mask.mask_decoder = nn.Module()
        self.sam_mask.mask_decoder.transformer = nn.Module()
        self.sam_mask.mask_decoder.transformer.q_proj = nn.Linear(256, 256)
        self.decode_calls = []

    @property
    def mask_input_size(self):
        return (288, 288)

    def get_image_position_encoding(self, device=None):
        pe = torch.zeros(1, 256, 2, 2)
        return pe if device is None else pe.to(device)

    def encode_image(self, image):
        batch = image.shape[0]
        return {
            "image_embed": torch.ones(batch, 256, 2, 2),
            "high_res_features": (
                torch.ones(batch, 32, 8, 8),
                torch.ones(batch, 64, 4, 4),
            ),
        }

    def encode_prompt(self, points=None, boxes=None, masks=None):
        batch = points[0].shape[0] if points is not None else masks.shape[0]
        return torch.zeros(batch, 1, 256), torch.zeros(batch, 256, 2, 2)

    def decode_masks(
        self,
        image_embed,
        high_res_features,
        prompt,
        image_pe,
        multimask=True,
        repeat_image=False,
        mix=None,
    ):
        self.decode_calls.append(
            {
                "image_embed": tuple(image_embed.shape),
                "high_res": [tuple(item.shape) for item in high_res_features],
                "prompt": tuple(prompt[0].shape),
                "image_pe": tuple(image_pe.shape),
                "multimask": multimask,
                "repeat_image": repeat_image,
                "mix": tuple(mix.shape),
            }
        )
        batch = prompt[0].shape[0]
        return (
            torch.ones(batch, 1, 8, 8),
            torch.ones(batch, 1),
            torch.ones(batch, 1, 256),
            torch.ones(batch, 1),
        )


def test_router_uses_image_condition_and_prompt_type():
    router = Router(
        image_dim=4,
        num_conditions=3,
        num_experts=2,
        hidden_dim=8,
        embed_dim=6,
    )
    image = torch.ones(2, 4, 3, 3)
    cond = torch.tensor([0, 2])
    prompts = ["point", "mask"]

    out = router(image, cond, prompts)

    assert out.shape == (2, 2)
    assert torch.allclose(out.sum(dim=1), torch.ones(2))
    assert router.cond.weight.shape[1] == 6
    assert router.prompt.weight.shape[1] == 6


def test_lora_linear_keeps_base_frozen_and_adds_expert_delta():
    base = nn.Linear(3, 2)
    lora = LoraLinear(base, rank=2, num_experts=2, alpha=2.0)
    x = torch.ones(2, 3)
    mix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    out = lora(x, mix)

    assert out.shape == (2, 2)
    assert base.weight.requires_grad is False
    assert base.bias is not None
    assert base.bias.requires_grad is False
    assert any(param.requires_grad for param in lora.adapter_parameters())
    assert not hasattr(lora, "set_mix")


def test_lora_linear_matches_weighted_expert_sum():
    base = nn.Linear(2, 2, bias=False)
    nn.init.ones_(base.weight)
    layer = LoraLinear(base, rank=1, num_experts=2, alpha=1.0)
    for down, up in zip(layer.down, layer.up):
        nn.init.ones_(down.weight)
        nn.init.ones_(up.weight)
    x = torch.ones(2, 2)
    mix = torch.tensor([[1.0, 0.0], [0.25, 0.75]])

    expected = base(x)
    for expert, (down, up) in enumerate(zip(layer.down, layer.up)):
        expected = expected + up(down(x)) * mix[:, expert, None]

    assert torch.allclose(layer(x, mix), expected)


def test_zero_lora_mix_preserves_autocast_dtype_and_output():
    layer = LoraLinear(nn.Linear(8, 8), rank=2, num_experts=2).eval()
    x = torch.randn(1, 3, 8)
    mix = torch.tensor([[0.5, 0.5]])

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        plain = layer(x, None)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        adapted = layer(x, mix)

    assert adapted.dtype == plain.dtype
    assert torch.equal(adapted, plain)


def test_attention_passes_mix_directly_to_lora_layers():
    attention = Attention(embedding_dim=4, num_heads=2)
    attention.q_proj = LoraLinear(attention.q_proj, rank=2, num_experts=2)
    attention.k_proj = LoraLinear(attention.k_proj, rank=2, num_experts=2)
    attention.v_proj = LoraLinear(attention.v_proj, rank=2, num_experts=2)
    attention.out_proj = LoraLinear(attention.out_proj, rank=2, num_experts=2)
    x = torch.ones(1, 3, 4)
    mix = torch.tensor([[0.7, 0.3]])

    out = attention(q=x, k=x, v=x, mix=mix)

    assert out.shape == x.shape


def test_feature_adapter_keeps_feature_shape():
    adapter = FeatureAdapter(channels=4, rank=2, num_experts=3)
    x = torch.ones(2, 4, 5, 6)
    mix = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]])

    out = adapter(x, mix)

    assert out.shape == x.shape
    assert any(param.requires_grad for param in adapter.parameters())


def test_feature_adapter_matches_weighted_expert_sum():
    adapter = FeatureAdapter(channels=2, rank=1, num_experts=2, alpha=1.0)
    for down, up in zip(adapter.down, adapter.up):
        nn.init.ones_(down.weight)
        nn.init.ones_(up.weight)
    x = torch.ones(2, 2, 1, 1)
    mix = torch.tensor([[1.0, 0.0], [0.25, 0.75]])

    expected = x.clone()
    for expert, (down, up) in enumerate(zip(adapter.down, adapter.up)):
        expected = expected + up(down(x)) * mix[:, expert, None, None, None]

    assert torch.allclose(adapter(x, mix), expected)


def test_zero_feature_adapter_preserves_bfloat16_dtype_and_output():
    adapter = FeatureAdapter(channels=2, rank=1, num_experts=2).eval()
    x = torch.randn(1, 2, 2, 2, dtype=torch.bfloat16)
    mix = torch.tensor([[0.5, 0.5]])

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        adapted = adapter(x, mix)

    assert adapted.dtype == x.dtype
    assert torch.equal(adapted, x)


def test_finetune_model_freezes_base_and_wraps_decoder_linear():
    base = FakeModel()
    model = FinetuneModel(
        base,
        num_conditions=2,
        num_experts=2,
        num_classes=3,
        lora_rank=2,
        feature_rank=2,
    )

    assert isinstance(
        base.sam_mask.mask_decoder.transformer.q_proj,
        LoraLinear,
    )
    assert isinstance(
        base.sam_mask.mask_decoder.transformer.lin1,
        LoraLinear,
    )
    assert base.keep.weight.requires_grad is False
    assert sum(isinstance(module, LoraLinear) for module in model.modules()) == 2
    assert all(param.requires_grad for param in model.trainable_parameters())


def test_finetune_model_uses_image_model_api_with_mix():
    base = FakeImageApiModel()
    model = FinetuneModel(
        base,
        num_conditions=3,
        num_experts=2,
        num_classes=4,
        lora_rank=2,
        feature_rank=2,
    )
    image = torch.zeros(1, 3, 16, 16)
    points = (
        torch.zeros(3, 1, 2),
        torch.ones(3, 1, dtype=torch.int),
    )

    encoded = model.encode_image(image)
    prompt = model.encode_prompt(points=points)
    out = model.decode_masks(
        encoded["image_embed"],
        tuple(encoded["high_res_features"]),
        prompt,
        model.get_image_position_encoding(),
        multimask=False,
        repeat_image=True,
        cond=1,
        prompt_type="point",
    )

    assert out[0].shape == (3, 1, 8, 8)
    assert model.mask_input_size == (288, 288)
    assert base.decode_calls == [
        {
            "image_embed": (1, 256, 2, 2),
            "high_res": [(1, 32, 8, 8), (1, 64, 4, 4)],
            "prompt": (3, 1, 256),
            "image_pe": (1, 256, 2, 2),
            "multimask": False,
            "repeat_image": True,
            "mix": (1, 2),
        }
    ]
