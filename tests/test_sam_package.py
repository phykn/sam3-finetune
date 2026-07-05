from pathlib import Path

import src.model.components.nn.modules as model_misc
import src.model.video.tracker.memory.encoder as memory
from src.model.components.nn.activation import resolve_activation
from src.model.components.nn.layers import (
    clone_modules,
    LayerNorm2d,
    LayerScale,
    MLPBlock,
)
from src.model.components.sam.mask_decoder import MaskDecoder
from src.model.components.sam.prompt_encoder import (
    PositionEmbeddingRandom,
    PromptEncoder,
)
from src.model.components.sam.rope import (
    apply_rotary_enc,
    compute_axial_cis,
    VisionRotaryEmbeddingVE,
)
from src.model.components.sam.transformer import (
    Attention,
    RoPEAttention,
    TwoWayTransformer,
)
from src.model.components.transformer.wrapper import TransformerWrapper
from src.model.grounding.output import SAM3Output
from src.model.grounding.scoring import DotProductScorer
from src.ops.tensor import invert_sigmoid


def test_sam_layers_are_in_nn_layers():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "model" / "components" / "nn" / "layers.py").is_file()
    assert not (root / "src" / "model" / "components" / "sam" / "layers.py").exists()
    assert not (root / "src" / "model" / "nn").exists()
    assert not (root / "src" / "model" / "sam").exists()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "sam" / "common.py").exists()
    assert not (root / "src" / "common.py").exists()
    assert MLPBlock.__module__ == "src.model.components.nn.layers"
    assert LayerNorm2d.__module__ == "src.model.components.nn.layers"


def test_sam_prompt_encoder_is_the_prompt_encoder_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (
        root / "src" / "model" / "components" / "sam" / "prompt_encoder.py"
    ).is_file()
    assert not (root / "src" / "model" / "sam").exists()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "prompt_encoder.py").exists()
    assert PromptEncoder.__module__ == "src.model.components.sam.prompt_encoder"
    assert (
        PositionEmbeddingRandom.__module__ == "src.model.components.sam.prompt_encoder"
    )


def test_sam_mask_decoder_is_the_mask_decoder_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "model" / "components" / "sam" / "mask_decoder.py").is_file()
    assert not (root / "src" / "model" / "sam").exists()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "mask_decoder.py").exists()
    assert MaskDecoder.__module__ == "src.model.components.sam.mask_decoder"


def test_sam_transformer_is_the_transformer_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "model" / "components" / "sam" / "transformer.py").is_file()
    assert not (root / "src" / "model" / "sam").exists()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "transformer.py").exists()
    assert TwoWayTransformer.__module__ == "src.model.components.sam.transformer"
    assert Attention.__module__ == "src.model.components.sam.transformer"
    assert RoPEAttention.__module__ == "src.model.components.sam.transformer"


def test_sam_rope_is_the_rope_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "model" / "components" / "sam" / "rope.py").is_file()
    assert not (root / "src" / "model" / "sam").exists()
    assert not (root / "src" / "sam").exists()
    assert not (root / "src" / "rope.py").exists()
    assert apply_rotary_enc.__module__ == "src.model.components.sam.rope"
    assert compute_axial_cis.__module__ == "src.model.components.sam.rope"
    assert VisionRotaryEmbeddingVE.__module__ == "src.model.components.sam.rope"


def test_layer_norm_2d_has_single_source():
    assert memory.LayerNorm2d is LayerNorm2d
    assert not hasattr(model_misc, "LayerNorm2d")


def test_nn_modules_are_split_by_responsibility():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "model" / "components" / "nn" / "layers.py").is_file()
    assert (root / "src" / "model" / "grounding" / "output.py").is_file()
    assert (root / "src" / "model" / "grounding" / "scoring.py").is_file()
    assert (
        root / "src" / "model" / "components" / "transformer" / "decoder.py"
    ).is_file()
    assert (
        root / "src" / "model" / "components" / "transformer" / "encoder.py"
    ).is_file()
    assert (
        root / "src" / "model" / "components" / "transformer" / "wrapper.py"
    ).is_file()
    assert (root / "src" / "model" / "components" / "nn" / "activation.py").is_file()
    assert not (root / "src" / "model" / "transformer").exists()
    assert not (root / "src" / "model" / "nn").exists()
    assert not (root / "src" / "nn").exists()
    assert LayerScale.__module__ == "src.model.components.nn.layers"
    assert LayerNorm2d.__module__ == "src.model.components.nn.layers"
    assert MLPBlock.__module__ == "src.model.components.nn.layers"
    assert SAM3Output.__module__ == "src.model.grounding.output"
    assert DotProductScorer.__module__ == "src.model.grounding.scoring"
    assert TransformerWrapper.__module__ == "src.model.components.transformer.wrapper"
    assert resolve_activation.__module__ == "src.model.components.nn.activation"
    assert clone_modules.__module__ == "src.model.components.nn.layers"
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


def test_sam3_output_flattened_negative_index():
    output = SAM3Output([[{"step": 1}], [{"step": 2}, {"step": 3}]])

    with SAM3Output.iteration_mode(output, SAM3Output.IterMode.FLATTENED):
        assert output[-1] == {"step": 3}


def test_sam_package_does_not_reexport_internal_layers():
    import src.model.components.sam as sam

    for name in ("LayerNorm2d", "MLPBlock"):
        assert not hasattr(sam, name)
