from pathlib import Path

from src.ml.blocks.grounding.decoder import GroundingDecoder
from src.ml.blocks.grounding.image import GroundingImage
from src.ml.blocks.grounding.prompt import GroundingPromptEncoder
from src.ml.blocks.grounding.tokens import VisualTokens
from src.ml.blocks.image.features import ImageFeatures
from src.ml.blocks.image.masks import ImageMaskDecoder
from src.ml.blocks.image.prompt import ImagePromptEncoder
from src.ml.blocks.vision import make_vision_backbone, VisionEncoder


def test_core_blocks_are_grouped_by_workflow():
    assert make_vision_backbone.__module__ == "src.ml.blocks.vision"
    assert VisionEncoder.__module__ == "src.ml.blocks.vision"
    assert ImageFeatures.__module__ == "src.ml.blocks.image.features"
    assert ImagePromptEncoder.__module__ == "src.ml.blocks.image.prompt"
    assert ImageMaskDecoder.__module__ == "src.ml.blocks.image.masks"
    assert VisualTokens.__module__ == "src.ml.blocks.grounding.tokens"
    assert GroundingImage.__module__ == "src.ml.blocks.grounding.image"
    assert GroundingPromptEncoder.__module__ == "src.ml.blocks.grounding.prompt"
    assert GroundingDecoder.__module__ == "src.ml.blocks.grounding.decoder"


def test_replaced_flat_block_files_are_removed():
    root = Path(__file__).resolve().parents[1] / "src" / "ml" / "blocks"
    for name in (
        "cond.py",
        "ground_dec.py",
        "ground_image.py",
        "ground_prompt.py",
        "sam_image.py",
        "sam_mask.py",
        "sam_prompt.py",
    ):
        assert not (root / name).exists()


def test_component_factories_move_to_owning_blocks():
    root = Path(__file__).resolve().parents[1] / "src" / "ml" / "components"

    assert not (root / "backbone" / "create.py").exists()
    assert not (root / "grounding" / "create.py").exists()
