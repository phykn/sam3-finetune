from pathlib import Path

from src.backbone.image_encoder import InteractiveImageEncoder
from src.backbone.neck import Sam3DualViTDetNeck, Sam3TriViTDetNeck
from src.backbone.vit import PatchEmbed, ViT


def test_vit_is_the_backbone_vit_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "backbone" / "vit.py").is_file()
    assert not (root / "src" / "vit.py").exists()
    assert ViT.__module__ == "src.backbone.vit"
    assert PatchEmbed.__module__ == "src.backbone.vit"


def test_neck_is_the_backbone_neck_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "backbone" / "neck.py").is_file()
    assert not (root / "src" / "neck.py").exists()
    assert Sam3DualViTDetNeck.__module__ == "src.backbone.neck"
    assert Sam3TriViTDetNeck.__module__ == "src.backbone.neck"


def test_image_encoder_is_the_backbone_image_encoder_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "backbone" / "image_encoder.py").is_file()
    assert not (root / "src" / "image_encoder.py").exists()
    assert InteractiveImageEncoder.__module__ == "src.backbone.image_encoder"


def test_backbone_package_does_not_reexport_internal_modules():
    import src.backbone as backbone

    for name in (
        "InteractiveImageEncoder",
        "Sam3DualViTDetNeck",
        "Sam3TriViTDetNeck",
        "ViT",
    ):
        assert not hasattr(backbone, name)
