from pathlib import Path

import src.model.nn.modules as model_misc
import src.model.video._tracker.memory as memory
from src.model.nn.layers import LayerScale
from src.model.nn.output import SAM3Output
from src.model.nn.scoring import DotProductScoring
from src.model.nn.transformer import TransformerWrapper
from src.model.nn.utils import get_activation_fn, get_clones, inverse_sigmoid
from src.model.sam.layers import LayerNorm2d, MLPBlock
from src.model.sam.mask_decoder import MaskDecoder
from src.model.sam.prompt_encoder import PositionEmbeddingRandom, PromptEncoder
from src.model.sam.rope import (
    apply_rotary_enc,
    compute_axial_cis,
    VisionRotaryEmbeddingVE,
)
from src.model.sam.transformer import Attention, RoPEAttention, TwoWayTransformer


def test_sam_layers_is_the_layers_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "model" / "sam" / "layers.py").is_file()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "sam" / "common.py").exists()
    assert not (root / "src" / "common.py").exists()
    assert MLPBlock.__module__ == "src.model.sam.layers"
    assert LayerNorm2d.__module__ == "src.model.sam.layers"


def test_sam_prompt_encoder_is_the_prompt_encoder_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "model" / "sam" / "prompt_encoder.py").is_file()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "prompt_encoder.py").exists()
    assert PromptEncoder.__module__ == "src.model.sam.prompt_encoder"
    assert PositionEmbeddingRandom.__module__ == "src.model.sam.prompt_encoder"


def test_sam_mask_decoder_is_the_mask_decoder_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "model" / "sam" / "mask_decoder.py").is_file()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "mask_decoder.py").exists()
    assert MaskDecoder.__module__ == "src.model.sam.mask_decoder"


def test_sam_transformer_is_the_transformer_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "model" / "sam" / "transformer.py").is_file()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "transformer.py").exists()
    assert TwoWayTransformer.__module__ == "src.model.sam.transformer"
    assert Attention.__module__ == "src.model.sam.transformer"
    assert RoPEAttention.__module__ == "src.model.sam.transformer"


def test_sam_rope_is_the_rope_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "model" / "sam" / "rope.py").is_file()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "rope.py").exists()
    assert apply_rotary_enc.__module__ == "src.model.sam.rope"
    assert compute_axial_cis.__module__ == "src.model.sam.rope"
    assert VisionRotaryEmbeddingVE.__module__ == "src.model.sam.rope"


def test_layer_norm_2d_has_single_source():
    assert memory.LayerNorm2d is LayerNorm2d
    assert not hasattr(model_misc, "LayerNorm2d")


def test_nn_modules_are_split_by_responsibility():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "model" / "nn" / "layers.py").is_file()
    assert (root / "src" / "model" / "nn" / "output.py").is_file()
    assert (root / "src" / "model" / "nn" / "scoring.py").is_file()
    assert (root / "src" / "model" / "nn" / "transformer.py").is_file()
    assert (root / "src" / "model" / "nn" / "utils.py").is_file()
    assert not (root / "src" / "nn").exists()
    assert LayerScale.__module__ == "src.model.nn.layers"
    assert SAM3Output.__module__ == "src.model.nn.output"
    assert DotProductScoring.__module__ == "src.model.nn.scoring"
    assert TransformerWrapper.__module__ == "src.model.nn.transformer"
    assert get_activation_fn.__module__ == "src.model.nn.utils"
    assert get_clones.__module__ == "src.model.nn.utils"
    assert inverse_sigmoid.__module__ == "src.model.nn.utils"
    for name in (
        "LayerScale",
        "SAM3Output",
        "DotProductScoring",
        "TransformerWrapper",
        "get_activation_fn",
        "get_clones",
        "inverse_sigmoid",
    ):
        assert not hasattr(model_misc, name)


def test_sam3_output_flattened_negative_index():
    output = SAM3Output([[{"step": 1}], [{"step": 2}, {"step": 3}]])

    with SAM3Output.iteration_mode(output, SAM3Output.IterMode.FLATTENED):
        assert output[-1] == {"step": 3}


def test_sam_package_does_not_reexport_internal_layers():
    import src.model.sam as sam

    for name in ("LayerNorm2d", "MLPBlock"):
        assert not hasattr(sam, name)
