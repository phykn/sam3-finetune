import numpy as np
import torch
from src.data import prompt


def test_build_points_and_box_make_sam_prompt_labels():
    points, point_labels = prompt.build_points(
        [[10, 5]],
        None,
        (10, 20),
        100,
        torch.device("cpu"),
    )
    box, box_labels = prompt.build_box(
        [2, 1, 18, 9],
        (10, 20),
        100,
        torch.device("cpu"),
    )

    assert points.tolist() == [[[50.0, 50.0]]]
    assert point_labels.tolist() == [[1]]
    assert box.tolist() == [[[10.0, 10.0], [90.0, 90.0]]]
    assert box_labels.tolist() == [[2, 3]]


def test_mask_resizes_to_prompt_size():
    out = prompt.build_mask(
        np.ones((2, 2), dtype=np.float32),
        (4, 6),
        torch.device("cpu"),
    )

    assert out.shape == (1, 1, 4, 6)
    assert out.dtype == torch.float32


def test_prompt_module_exposes_only_input_builders():
    assert not hasattr(prompt, "make")
    assert not hasattr(prompt, "scale")
    assert not hasattr(prompt, "merge")
    assert not hasattr(prompt, "make_dummy")
