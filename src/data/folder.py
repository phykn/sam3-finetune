from pathlib import Path


def expand(items: list[dict]) -> tuple[list[str], list[int], list[dict]]:
    paths = []
    conds = []
    labels = []
    for item in items:
        folder = Path(item["path"])
        if not folder.is_dir():
            raise ValueError(f"finetune folder is not a directory: {folder}")
        files = sorted(folder.glob("*.json"))
        if not files:
            raise ValueError(f"finetune folder contains no JSON files: {folder}")
        label = {"target": item["target"], "weight": item["weight"]}
        paths.extend(str(path) for path in files)
        conds.extend([item["cond"]] * len(files))
        labels.extend([label] * len(files))
    return paths, conds, labels
