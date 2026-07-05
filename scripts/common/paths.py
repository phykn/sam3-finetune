import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ensure_workspace_on_path(root: Path = ROOT) -> None:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def resolve_workspace_path(
    path: str | Path,
    *,
    root: Path = ROOT,
) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path
