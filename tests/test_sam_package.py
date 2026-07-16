from pathlib import Path

import torch
from torch import nn

import src.ml.components.nn.attention as model_misc
import src.ml.components.video.memory as memory
from src.ml.components.grounding.box_out import write_box_outputs
from src.ml.components.grounding.scoring import DotProductScorer
from src.ml.components.nn.activation import resolve_activation
from src.ml.components.nn.layers import (
    clone_modules,
    LayerNorm2d,
    LayerScale,
    MLP,
    MLPBlock,
)
from src.ml.components.sam.mask_decoder import MaskDecoder
from src.ml.components.sam.prompt_encoder import PositionEmbeddingRandom, PromptEncoder
from src.ml.components.sam.rope import (
    apply_rotary_enc,
    compute_axial_cis,
    VisionRotaryEmbeddingVE,
)
from src.ml.components.sam.transformer import (
    Attention,
    RoPEAttention,
    TwoWayTransformer,
)
from src.ml.components.transformer.model import Transformer
from src.ops.tensor import inverse_sigmoid


def test_sam_layers_are_in_nn_layers():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "ml" / "components" / "nn" / "layers.py").is_file()
    assert not (root / "src" / "ml" / "components" / "sam" / "layers.py").exists()
    assert not (root / "src" / "ml" / "nn").exists()
    assert not (root / "src" / "ml" / "sam").exists()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "sam" / "common.py").exists()
    assert not (root / "src" / "common.py").exists()
    assert MLPBlock.__module__ == "src.ml.components.nn.layers"
    assert LayerNorm2d.__module__ == "src.ml.components.nn.layers"


def test_sam_prompt_encoder_is_the_prompt_encoder_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "ml" / "components" / "sam" / "prompt_encoder.py").is_file()
    assert not (root / "src" / "ml" / "sam").exists()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "prompt_encoder.py").exists()
    assert PromptEncoder.__module__ == "src.ml.components.sam.prompt_encoder"
    assert PositionEmbeddingRandom.__module__ == "src.ml.components.sam.prompt_encoder"


def test_sam_mask_decoder_is_the_mask_decoder_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "ml" / "components" / "sam" / "mask_decoder.py").is_file()
    assert not (root / "src" / "ml" / "sam").exists()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "mask_decoder.py").exists()
    assert MaskDecoder.__module__ == "src.ml.components.sam.mask_decoder"


def test_dynamic_mask_selection_returns_the_matching_token():
    decoder = MaskDecoder.__new__(MaskDecoder)
    nn.Module.__init__(decoder)
    decoder.dynamic_multimask_via_stability = True
    decoder.dynamic_multimask_stability_delta = 0.05
    decoder.dynamic_multimask_stability_thresh = 0.98
    decoder.eval()

    masks = torch.zeros(1, 4, 1, 2)
    masks[0, 0, 0] = torch.tensor([0.1, 0.0])
    masks[0, 2] = 2
    ious = torch.tensor([[0.1, 0.2, 0.9, 0.3]])
    tokens = torch.arange(4.0).view(1, 4, 1)
    objects = torch.ones(1, 1)
    decoder.predict_masks = lambda **_kwargs: (masks, ious, tokens, objects)

    out_masks, out_ious, out_tokens, _objects = decoder(
        image_embeddings=torch.empty(0),
        image_pe=torch.empty(0),
        sparse_prompt_embeddings=torch.empty(0),
        dense_prompt_embeddings=torch.empty(0),
        multimask_output=False,
        repeat_image=False,
    )

    assert torch.equal(out_masks, masks[:, 2:3])
    assert torch.equal(out_ious, ious[:, 2:3])
    assert out_tokens.item() == 2


def test_sam_transformer_is_the_transformer_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "ml" / "components" / "sam" / "transformer.py").is_file()
    assert not (root / "src" / "ml" / "sam").exists()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "transformer.py").exists()
    assert TwoWayTransformer.__module__ == "src.ml.components.sam.transformer"
    assert Attention.__module__ == "src.ml.components.sam.transformer"
    assert RoPEAttention.__module__ == "src.ml.components.sam.transformer"


def test_sam_rope_is_the_rope_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "ml" / "components" / "sam" / "rope.py").is_file()
    assert not (root / "src" / "ml" / "sam").exists()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "rope.py").exists()
    assert apply_rotary_enc.__module__ == "src.ml.components.sam.rope"
    assert compute_axial_cis.__module__ == "src.ml.components.sam.rope"
    assert VisionRotaryEmbeddingVE.__module__ == "src.ml.components.sam.rope"


def test_layer_norm_2d_has_single_source():
    assert memory.LayerNorm2d is LayerNorm2d
    assert not hasattr(model_misc, "LayerNorm2d")


def test_mlp_has_single_source():
    root = Path(__file__).resolve().parents[1] / "src" / "ml" / "components"

    assert MLP.__module__ == "src.ml.components.nn.layers"
    assert not (root / "video" / "mlp.py").exists()
    assert not hasattr(
        __import__("src.ml.components.sam.mask_decoder", fromlist=["MLP"]), "MLP"
    )


def test_nn_modules_are_split_by_responsibility():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "ml" / "components" / "nn" / "layers.py").is_file()
    assert (root / "src" / "ml" / "components" / "grounding" / "box_out.py").is_file()
    assert not (root / "src" / "ml" / "components" / "grounding" / "output.py").exists()
    assert (root / "src" / "ml" / "components" / "grounding" / "scoring.py").is_file()
    assert (root / "src" / "ml" / "components" / "transformer" / "decoder.py").is_file()
    assert (root / "src" / "ml" / "components" / "transformer" / "encoder.py").is_file()
    assert (root / "src" / "ml" / "components" / "transformer" / "model.py").is_file()
    assert (root / "src" / "ml" / "components" / "nn" / "activation.py").is_file()
    assert not (root / "src" / "ml" / "transformer").exists()
    assert not (root / "src" / "ml" / "nn").exists()
    assert not (root / "src" / "nn").exists()
    assert LayerScale.__module__ == "src.ml.components.nn.layers"
    assert LayerNorm2d.__module__ == "src.ml.components.nn.layers"
    assert MLPBlock.__module__ == "src.ml.components.nn.layers"
    assert write_box_outputs.__module__ == "src.ml.components.grounding.box_out"
    assert DotProductScorer.__module__ == "src.ml.components.grounding.scoring"
    assert Transformer.__module__ == "src.ml.components.transformer.model"
    assert resolve_activation.__module__ == "src.ml.components.nn.activation"
    assert clone_modules.__module__ == "src.ml.components.nn.layers"
    assert inverse_sigmoid.__module__ == "src.ops.tensor"
    for name in (
        "LayerScale",
        "LayerNorm2d",
        "MLPBlock",
        "Transformer",
        "resolve_activation",
        "clone_modules",
        "inverse_sigmoid",
    ):
        assert not hasattr(model_misc, name)


def test_sam_package_does_not_reexport_internal_layers():
    import src.ml.components.sam as sam

    for name in ("LayerNorm2d", "MLPBlock"):
        assert not hasattr(sam, name)
