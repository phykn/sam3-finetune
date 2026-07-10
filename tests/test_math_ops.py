import numpy as np
import torch

from src.ops.box import cxcywh_to_xyxy, nms_indices
from src.ops.tensor import inverse_sigmoid


def test_inverse_sigmoid_round_trip():
    values = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])

    torch.testing.assert_close(torch.sigmoid(inverse_sigmoid(values)), values)


def test_cxcywh_to_xyxy_uses_center_and_size():
    boxes = torch.tensor([[0.5, 0.25, 0.4, 0.2]])
    expected = torch.tensor([[0.3, 0.15, 0.7, 0.35]])

    torch.testing.assert_close(cxcywh_to_xyxy(boxes), expected)


def test_nms_indices_accepts_numpy_and_keeps_score_order():
    boxes = np.array([[0, 0, 2, 2], [0, 0, 2, 2], [4, 4, 5, 5]])
    scores = np.array([0.8, 0.9, 0.7])

    assert nms_indices(boxes, scores, 0.5) == [1, 2]
