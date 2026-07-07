from pathlib import Path

from src.ml.components.backbone.neck import Sam3DualViTDetNeck, Sam3TriViTDetNeck
from src.ml.components.backbone.vit import PatchEmbed, ViT


def test_vit_is_the_backbone_vit_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "ml" / "components" / "backbone" / "vit.py").is_file()
    assert not (root / "src" / "ml" / "backbone").exists()
    assert not (root / "src" / "backbone").exists()
    assert not (root / "src" / "vit.py").exists()
    assert ViT.__module__ == "src.ml.components.backbone.vit"
    assert PatchEmbed.__module__ == "src.ml.components.backbone.vit"


def test_neck_is_the_backbone_neck_module_location():
    root = Path(__file__).resolve().parents[1]

    assert (root / "src" / "ml" / "components" / "backbone" / "neck.py").is_file()
    assert not (root / "src" / "ml" / "backbone").exists()
    assert not (root / "src" / "backbone").exists()
    assert not (root / "src" / "neck.py").exists()
    assert Sam3DualViTDetNeck.__module__ == "src.ml.components.backbone.neck"
    assert Sam3TriViTDetNeck.__module__ == "src.ml.components.backbone.neck"


def test_backbone_package_does_not_reexport_internal_modules():
    import src.ml.components.backbone as backbone

    for name in (
        "Sam3DualViTDetNeck",
        "Sam3TriViTDetNeck",
        "ViT",
    ):
        assert not hasattr(backbone, name)
