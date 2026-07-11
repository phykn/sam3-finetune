from pathlib import Path

import yaml

from src.data.sample import load

ROOT = Path(__file__).resolve().parents[1]


def test_finetune_dataset_matches_folder_classes():
    config = yaml.safe_load(
        (ROOT / "config" / "finetune_test.yaml").read_text(encoding="utf-8")
    )

    for split in ("train", "valid"):
        folders = config["data"][split]["folders"]
        assert len(folders) == 4
        for item in folders:
            folder = ROOT / item["path"]
            class_id = int(folder.name.split("_", 1)[0])
            paths = sorted(folder.glob("*.json"))
            assert paths
            for path in paths:
                sample = load(path)
                assert sample.objects
                assert all(obj.class_id == class_id for obj in sample.objects)
                assert all(
                    obj.mask(sample.image.shape).sum() > 0 for obj in sample.objects
                )


def test_finetune_dataset_uses_object_background_class_targets():
    config = yaml.safe_load(
        (ROOT / "config" / "finetune_test.yaml").read_text(encoding="utf-8")
    )
    folders = config["data"]["train"]["folders"]

    assert [item["target"] for item in folders] == [
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
    ]
    assert [item["weight"] for item in folders] == [
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1],
        [1, 1, 1],
    ]
    assert [item["cond"] for item in folders] == [0, 1, 0, 1]
