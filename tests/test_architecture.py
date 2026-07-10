import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
WORKSPACE = ROOT.parent

LAYERS = {
    "src.data": 0,
    "src.io": 0,
    "src.ops": 0,
    "src.ml.runtime": 0,
    "src.ml.structures": 0,
    "src.ml.components": 1,
    "src.ml.blocks": 2,
    "src.ml.model": 3,
    "src.build": 4,
    "src.predict": 4,
    "scripts": 5,
}


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


def module_name(path: Path) -> str:
    return ".".join(path.relative_to(WORKSPACE).with_suffix("").parts)


def resolve_imports(path: Path) -> list[str]:
    module = module_name(path)
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = module.split(".")[: -node.level]
                if node.module:
                    parts.extend(node.module.split("."))
                names.append(".".join(parts))
            elif node.module:
                names.append(node.module)
    return names


def layer(name: str) -> int | None:
    matches = [
        (prefix, rank)
        for prefix, rank in LAYERS.items()
        if name == prefix or name.startswith(prefix + ".")
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: len(item[0]))[1]


def test_imports_follow_one_way_layers() -> None:
    bad = []
    paths = list(ROOT.rglob("*.py")) + list((WORKSPACE / "scripts").glob("*.py"))
    for path in paths:
        source_rank = layer(module_name(path))
        if source_rank is None:
            continue
        for name in resolve_imports(path):
            target_rank = layer(name)
            if target_rank is not None and target_rank > source_rank:
                bad.append(f"{path.relative_to(WORKSPACE)}: {name}")

    assert bad == []


def test_model_assembles_blocks_without_video_runtime() -> None:
    bad = []
    for filename in ("image.py", "grounding.py"):
        path = ROOT / "ml" / "model" / filename
        for name in import_names(path):
            if ".video" in name:
                bad.append(f"{filename}: {name}")
    assert bad == []


def test_video_runtime_does_not_import_blocks() -> None:
    bad = []
    runtime = ROOT / "ml" / "components" / "video" / "tracker"
    for path in runtime.rglob("*.py"):
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


def test_video_blocks_do_not_create_runtime() -> None:
    bad = []
    for path in (ROOT / "ml" / "blocks").glob("video_*.py"):
        for name in resolve_imports(path):
            if name == "src.ml.components.video.tracking_model":
                bad.append(f"{path.relative_to(WORKSPACE)}: {name}")
    assert bad == []
