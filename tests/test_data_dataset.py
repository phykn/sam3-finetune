import numpy as np

from src.data.dataset import BaseDataset
from src.data.sample import Image, Object, Sample, save


def write_sample(path, objects):
    image = Image(array=np.zeros((6, 8, 3), dtype=np.uint8), id="img-1")
    sample = Sample(image=image, objects=objects)
    save(sample, path)
    return path


def test_base_dataset_loads_sample_json(tmp_path):
    obj = Object(
        object_id=1,
        class_id=2,
        box=(1, 1, 3, 3),
        roi=np.ones((2, 2), dtype=np.uint8),
    )
    path = write_sample(tmp_path / "sample.json", [obj])

    dataset = BaseDataset([str(path)])
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample.image.shape == (6, 8, 3)
    assert sample.image.id == "img-1"
    assert sample.objects[0].object_id == 1
