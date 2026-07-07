import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"


def import_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            prefix = "." * node.level
            names.append(f"{prefix}{node.module}")
    return names


def test_model_assembles_blocks_without_video_runtime() -> None:
    bad = [
        name
        for name in import_names(ROOT / "ml" / "model.py")
        if name.startswith(".video")
        or name.startswith("src.ml.video")
        or name == ".blocks.video_tracker"
    ]
    assert bad == []


def test_video_runtime_does_not_import_blocks() -> None:
    bad = []
    for path in (ROOT / "ml" / "video").rglob("*.py"):
        for name in import_names(path):
            if name.startswith("....blocks") or name.startswith("...blocks"):
                bad.append(f"{path.relative_to(ROOT)}: {name}")
            if name.startswith("src.ml.blocks"):
                bad.append(f"{path.relative_to(ROOT)}: {name}")
    assert bad == []


def test_predict_does_not_import_model_internals() -> None:
    bad = []
    for path in (ROOT / "predict").rglob("*.py"):
        for name in import_names(path):
            if name.endswith("ml.structures"):
                bad.append(f"{path.relative_to(ROOT)}: {name}")
            if ".ml.blocks" in name or ".ml.components" in name:
                bad.append(f"{path.relative_to(ROOT)}: {name}")
    assert bad == []


def test_video_tracker_is_model_not_block() -> None:
    assert not (ROOT / "ml" / "blocks" / "video_tracker.py").exists()
