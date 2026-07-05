import numpy as np


def test_calc_iou_measures_binary_overlap() -> None:
    from src.ops.mask import calc_iou

    first = np.zeros((4, 4), dtype=bool)
    second = np.zeros((4, 4), dtype=bool)
    first[0:2, 0:2] = True
    second[1:3, 1:3] = True

    assert calc_iou(first, second) == 1.0 / 7.0


def test_calc_iou_returns_zero_for_empty_union() -> None:
    from src.ops.mask import calc_iou

    assert calc_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0


def test_grounding_postprocess_uses_shared_mask_metric() -> None:
    import src.predict.grounding.postprocess as postprocess

    assert not hasattr(postprocess, "_mask_iou")


def test_ops_package_does_not_reexport_mask_functions() -> None:
    import src.ops as ops

    assert not hasattr(ops, "calc_iou")
