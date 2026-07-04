from pathlib import Path

import src.nn.modules as model_misc
import src.video._tracker.memory as memory
from src.nn.layers import LayerScale
from src.nn.output import SAM3Output
from src.nn.scoring import DotProductScoring
from src.nn.transformer import TransformerWrapper
from src.nn.utils import get_activation_fn, get_clones, inverse_sigmoid
from src.sam.layers import LayerNorm2d, MLPBlock
from src.sam.mask_decoder import MaskDecoder
from src.sam.prompt_encoder import PositionEmbeddingRandom, PromptEncoder
from src.sam.rope import VisionRotaryEmbeddingVE, apply_rotary_enc, compute_axial_cis
from src.sam.transformer import Attention, RoPEAttention, TwoWayTransformer


def test_sam_layers_is_the_layers_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "sam" / "layers.py").is_file()
    assert not (root / "src" / "sam" / "common.py").exists()
    assert not (root / "src" / "common.py").exists()
    assert MLPBlock.__module__ == "src.sam.layers"
    assert LayerNorm2d.__module__ == "src.sam.layers"


def test_sam_prompt_encoder_is_the_prompt_encoder_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "sam" / "prompt_encoder.py").is_file()
    assert not (root / "src" / "prompt_encoder.py").exists()
    assert PromptEncoder.__module__ == "src.sam.prompt_encoder"
    assert PositionEmbeddingRandom.__module__ == "src.sam.prompt_encoder"


def test_sam_mask_decoder_is_the_mask_decoder_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "sam" / "mask_decoder.py").is_file()
    assert not (root / "src" / "mask_decoder.py").exists()
    assert MaskDecoder.__module__ == "src.sam.mask_decoder"


def test_sam_transformer_is_the_transformer_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "sam" / "transformer.py").is_file()
    assert not (root / "src" / "transformer.py").exists()
    assert TwoWayTransformer.__module__ == "src.sam.transformer"
    assert Attention.__module__ == "src.sam.transformer"
    assert RoPEAttention.__module__ == "src.sam.transformer"


def test_sam_rope_is_the_rope_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "sam" / "rope.py").is_file()
    assert not (root / "src" / "rope.py").exists()
    assert apply_rotary_enc.__module__ == "src.sam.rope"
    assert compute_axial_cis.__module__ == "src.sam.rope"
    assert VisionRotaryEmbeddingVE.__module__ == "src.sam.rope"


def test_layer_norm_2d_has_single_source():
    assert memory.LayerNorm2d is LayerNorm2d
    assert not hasattr(model_misc, "LayerNorm2d")


def test_nn_modules_are_split_by_responsibility():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "nn" / "layers.py").is_file()
    assert (root / "src" / "nn" / "output.py").is_file()
    assert (root / "src" / "nn" / "scoring.py").is_file()
    assert (root / "src" / "nn" / "transformer.py").is_file()
    assert (root / "src" / "nn" / "utils.py").is_file()
    assert LayerScale.__module__ == "src.nn.layers"
    assert SAM3Output.__module__ == "src.nn.output"
    assert DotProductScoring.__module__ == "src.nn.scoring"
    assert TransformerWrapper.__module__ == "src.nn.transformer"
    assert get_activation_fn.__module__ == "src.nn.utils"
    assert get_clones.__module__ == "src.nn.utils"
    assert inverse_sigmoid.__module__ == "src.nn.utils"
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
    import src.sam as sam

    for name in ("LayerNorm2d", "MLPBlock"):
        assert not hasattr(sam, name)
