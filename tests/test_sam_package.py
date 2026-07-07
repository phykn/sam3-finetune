from pathlib import Path

import src.ml.components.nn.modules as model_misc
import src.ml.components.video.memory as memory
from src.ml.components.grounding.box_out import write_box_outputs
from src.ml.components.grounding.scoring import DotProductScorer
from src.ml.components.nn.activation import resolve_activation
from src.ml.components.nn.layers import clone_modules, LayerNorm2d, LayerScale, MLPBlock
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
from src.ml.components.transformer.wrapper import TransformerWrapper
from src.ops.tensor import invert_sigmoid


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


def test_nn_modules_are_split_by_responsibility():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "ml" / "components" / "nn" / "layers.py").is_file()
    assert (root / "src" / "ml" / "components" / "grounding" / "box_out.py").is_file()
    assert not (root / "src" / "ml" / "components" / "grounding" / "output.py").exists()
    assert (root / "src" / "ml" / "components" / "grounding" / "scoring.py").is_file()
    assert (root / "src" / "ml" / "components" / "transformer" / "decoder.py").is_file()
    assert (root / "src" / "ml" / "components" / "transformer" / "encoder.py").is_file()
    assert (root / "src" / "ml" / "components" / "transformer" / "wrapper.py").is_file()
    assert (root / "src" / "ml" / "components" / "nn" / "activation.py").is_file()
    assert not (root / "src" / "ml" / "transformer").exists()
    assert not (root / "src" / "ml" / "nn").exists()
    assert not (root / "src" / "nn").exists()
    assert LayerScale.__module__ == "src.ml.components.nn.layers"
    assert LayerNorm2d.__module__ == "src.ml.components.nn.layers"
    assert MLPBlock.__module__ == "src.ml.components.nn.layers"
    assert write_box_outputs.__module__ == "src.ml.components.grounding.box_out"
    assert DotProductScorer.__module__ == "src.ml.components.grounding.scoring"
    assert TransformerWrapper.__module__ == "src.ml.components.transformer.wrapper"
    assert resolve_activation.__module__ == "src.ml.components.nn.activation"
    assert clone_modules.__module__ == "src.ml.components.nn.layers"
    assert invert_sigmoid.__module__ == "src.ops.tensor"
    for name in (
        "LayerScale",
        "LayerNorm2d",
        "MLPBlock",
        "TransformerWrapper",
        "resolve_activation",
        "clone_modules",
        "invert_sigmoid",
    ):
        assert not hasattr(model_misc, name)


def test_sam_package_does_not_reexport_internal_layers():
    import src.ml.components.sam as sam

    for name in ("LayerNorm2d", "MLPBlock"):
        assert not hasattr(sam, name)
