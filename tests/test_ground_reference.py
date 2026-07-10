import numpy as np
import pytest
import torch

from src.predict.ground_ops import reference, sim


def test_validate_clips_boxes_and_preserves_classes():
    boxes, classes = reference.validate(
        [[-2, 1, 5, 7], [4, 2, 12, 9]],
        [2, 1],
        (8, 10),
    )

    np.testing.assert_array_equal(boxes, [[0, 1, 5, 7], [4, 2, 10, 8]])
    assert boxes.dtype == np.float32
    assert classes.dtype == np.int64


@pytest.mark.parametrize(
    "boxes,classes,message",
    [
        ([], [], "empty"),
        ([[0, 0, 2, 2]], [], "length"),
        ([[0, 0, 2]], [0], "Nx4"),
        ([[2, 0, 1, 2]], [0], "positive area"),
        ([[20, 20, 30, 30]], [0], "outside"),
        ([[0, 0, float("nan"), 2]], [0], "finite"),
        ([[0, 0, 2, 2]], [0.5], "integers"),
        ([[0, 0, 2, 2]], [-1], "non-negative"),
    ],
)
def test_validate_rejects_invalid_reference(boxes, classes, message):
    with pytest.raises(ValueError, match=message):
        reference.validate(boxes, classes, (8, 10))


def test_groups_boxes_by_sorted_class():
    boxes = np.array(
        [[0, 0, 2, 2], [2, 2, 4, 4], [4, 4, 6, 6]],
        dtype=np.float32,
    )

    classes, grouped = reference.groups(boxes, np.array([2, 1, 2]))

    assert classes.tolist() == [1, 2]
    np.testing.assert_array_equal(grouped[0], boxes[[1]])
    np.testing.assert_array_equal(grouped[1], boxes[[0, 2]])


def test_box_vectors_match_explicit_feature_grid_means():
    features = torch.tensor(
        [
            [
                [[1.0, 1.0, 0.0, 0.0]] * 4,
                [[0.0, 0.0, 1.0, 1.0]] * 4,
            ]
        ]
    )
    image = {"backbone_fpn": (features,)}
    boxes = np.array([[0, 0, 2, 4], [2, 0, 4, 4]], dtype=np.float32)

    out = sim.box_vectors(image, boxes, (4, 4))

    torch.testing.assert_close(out, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_max_scores_uses_best_exemplar():
    refs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[0.8, 0.2], [0.1, 0.9]])

    scores = sim.max_scores(refs, target)

    torch.testing.assert_close(scores, torch.tensor([0.8, 0.9]))


def test_feature_bank_concatenates_classes_across_references():
    references = [
        {
            "features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "feature_classes": np.array([1, 2]),
        },
        {
            "features": torch.tensor([[0.5, 0.5]]),
            "feature_classes": np.array([1]),
        },
    ]

    bank = reference.feature_bank(references)

    assert set(bank) == {1, 2}
    torch.testing.assert_close(
        bank[1],
        torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
    )


def test_prompt_groups_preserve_reference_order():
    references = [
        {
            "prompt": {
                "features": torch.ones(3, 1, 2),
                "mask": torch.zeros(1, 3, dtype=torch.bool),
            },
            "prompt_classes": np.array([2]),
        },
        {
            "prompt": {
                "features": torch.zeros(3, 2, 2),
                "mask": torch.zeros(2, 3, dtype=torch.bool),
            },
            "prompt_classes": np.array([1, 3]),
        },
    ]

    prompt, classes = reference.prompt_groups(references)

    assert prompt["features"].shape == (3, 3, 2)
    assert prompt["mask"].shape == (3, 3)
    assert classes.tolist() == [2, 1, 3]
