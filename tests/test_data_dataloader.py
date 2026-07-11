import numpy as np

from torch.utils.data.distributed import DistributedSampler

from src.data.dataloader import (
    collate,
    make_finetune_loader,
    make_infinite_train_loader,
)
from src.data.dataset import ValidDataset
from src.data.sample import Image, Object, Sample, save


def write_sample(path):
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    save(Sample(image=Image(array=image, id="img-1"), objects=[obj]), path)
    return path


def test_collate_stacks_tensors_and_keeps_prompt_list():
    batch = [
        {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "target": np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32),
            "mask_valid": True,
            "is_auto_bg": False,
            "cond": 2,
            "label_target": np.array([1.0, 0.0], dtype=np.float32),
            "label_weight": np.array([1.0, 0.0], dtype=np.float32),
            "prompt": {
                "type": "point",
                "points": np.array([[1.0, 1.0]], dtype=np.float32),
                "point_labels": np.array([1]),
                "box": None,
                "mask": None,
            },
        },
        {
            "image": np.full((4, 4, 3), 255, dtype=np.uint8),
            "target": np.zeros((2, 2), dtype=np.uint8),
            "mask_valid": False,
            "is_auto_bg": True,
            "cond": 5,
            "label_target": np.array([0.0, 1.0], dtype=np.float32),
            "label_weight": np.array([1.0, 1.0], dtype=np.float32),
            "prompt": {
                "type": "box",
                "points": None,
                "point_labels": None,
                "box": np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32),
                "mask": None,
            },
        },
    ]

    out = collate(batch)

    assert out["image"].shape == (2, 3, 4, 4)
    assert out["target"].shape == (2, 1, 2, 2)
    assert out["target"][0, 0].tolist() == [[0.0, 0.25], [0.5, 1.0]]
    assert out["mask_valid"].tolist() == [1.0, 0.0]
    assert out["is_auto_bg"].tolist() == [0.0, 1.0]
    assert out["prompt"][0]["type"] == "point"
    assert out["prompt"][1]["box"].tolist() == [0.0, 0.0, 2.0, 2.0]
    assert float(out["image"][0].min()) == -1.0
    assert float(out["image"][1].max()) == 1.0
    assert out["cond"].tolist() == [2, 5]
    assert out["label_target"].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert out["label_weight"].tolist() == [[1.0, 0.0], [1.0, 1.0]]


def test_infinite_train_loader_returns_next_batch(tmp_path):
    path = write_sample(tmp_path / "sample.json")
    loader = make_infinite_train_loader(
        [str(path)],
        batch_size=1,
        num_workers=0,
    )

    first = next(loader)
    second = next(loader)

    assert first["image"].shape == (1, 3, 1008, 1008)
    assert first["target"].shape == (1, 1, 288, 288)
    assert len(first["prompt"]) == 1
    assert second["image"].shape == (1, 3, 1008, 1008)


def test_make_finetune_loader_builds_distributed_validation_loader(tmp_path):
    path = write_sample(tmp_path / "sample.json")

    loader = make_finetune_loader(
        {
            "paths": [str(path)],
            "batch_size": 1,
            "num_workers": 0,
        },
        train=False,
        rank=0,
        world_size=2,
    )

    assert isinstance(loader.loader.dataset, ValidDataset)
    assert isinstance(loader.sampler, DistributedSampler)
    assert loader.sampler.shuffle is False
    assert loader.loader.drop_last is False


def test_make_finetune_loader_expands_folder_labels(tmp_path):
    folder = tmp_path / "particle"
    folder.mkdir()
    second = write_sample(folder / "b.json")
    first = write_sample(folder / "a.json")

    loader = make_finetune_loader(
        {
            "folders": [
                {
                    "path": str(folder),
                    "cond": 2,
                    "target": [1, 0, 1],
                    "weight": [1, 1, 0],
                }
            ],
            "batch_size": 1,
            "num_workers": 0,
        },
        train=False,
    )
    dataset = loader.loader.dataset

    assert dataset.paths == [str(first), str(second)]
    assert dataset.conds == (2, 2)
    assert dataset.labels == (
        {"target": [1, 0, 1], "weight": [1, 1, 0]},
        {"target": [1, 0, 1], "weight": [1, 1, 0]},
    )
