import ast
from pathlib import Path


def test_script_files_do_not_import_other_script_files() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    offenders: list[str] = []
    for path in scripts_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "scripts":
                offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "scripts."
            ):
                offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "scripts" or alias.name.startswith("scripts."):
                        offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == []


def test_runtime_scripts_use_predict_api_only() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    offenders: list[str] = []
    runtime_scripts = ("single.py", "grid.py", "ground.py", "video.py")

    for name in runtime_scripts:
        path = scripts_dir / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module.startswith("src.ml"):
                offenders.append(f"{name}:{node.lineno}:{node.module}")
            if node.module.startswith("src.predict.") and "_ops" in node.module:
                offenders.append(f"{name}:{node.lineno}:{node.module}")

    assert offenders == []
