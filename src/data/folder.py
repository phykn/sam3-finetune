from pathlib import Path


def expand(items: list[dict]) -> tuple[list[str], list[int], list[dict]]:
    paths = []
    conds = []
    labels = []
    for item in items:
        files = sorted(Path(item["path"]).glob("*.json"))
        label = {"target": item["target"], "weight": item["weight"]}
        paths.extend(str(path) for path in files)
        conds.extend([int(item.get("cond", 0))] * len(files))
        labels.extend([label] * len(files))
    return paths, conds, labels
