import torch
from src.data import ground


def test_build_points_and_boxes_for_grounding_prompt():
    device = torch.device("cpu")

    points, point_labels = ground.build_points(
        [[10.0, 5.0]],
        [1],
        (10, 20),
        device,
    )
    boxes, box_labels = ground.build_boxes(
        [2.0, 1.0, 18.0, 9.0],
        (10, 20),
        device,
    )

    assert points.shape == (1, 1, 2)
    assert point_labels.shape == (1, 1)
    assert boxes.shape == (1, 1, 4)
    assert box_labels.shape == (1, 1)
    torch.testing.assert_close(points[:, 0], torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(boxes[:, 0], torch.tensor([[0.5, 0.5, 0.8, 0.8]]))
