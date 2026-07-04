import numpy as np


def test_mask_iou_measures_binary_overlap() -> None:
    from src.metrics.mask import mask_iou

    first = np.zeros((4, 4), dtype=bool)
    second = np.zeros((4, 4), dtype=bool)
    first[0:2, 0:2] = True
    second[1:3, 1:3] = True

    assert mask_iou(first, second) == 1.0 / 7.0


def test_mask_iou_returns_zero_for_empty_union() -> None:
    from src.metrics.mask import mask_iou

    assert mask_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0


def test_grounding_postprocess_uses_shared_mask_metric() -> None:
    import src.grounding.postprocess as postprocess

    assert not hasattr(postprocess, "_mask_iou")


def test_metrics_package_does_not_reexport_metric_functions() -> None:
    import src.metrics as metrics

    assert not hasattr(metrics, "mask_iou")
