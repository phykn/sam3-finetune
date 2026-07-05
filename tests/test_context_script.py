import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from src.types import ContextPrediction

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "context.py"


def load_context_script():
    spec = importlib.util.spec_from_file_location("context_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_simple_reference_context_defaults():
    script = load_context_script()

    args = script.parse_args([])

    assert args.reference_image == "asset/frog_reference.jpg"
    assert args.target_image == "asset/frog_target.jpg"
    assert args.checkpoint == "weight/sam3.1_multiplex.pt"
    assert args.output_dir == "outputs/context"
    assert args.device == "cuda"
    for name in (
        "reference_box",
        "reference_mask_source",
        "reference_mask",
        "feature_layer",
        "candidate_count",
        "decode_batch_size",
        "max_masks",
        "target_point",
        "output",
        "reference_overlay",
    ):
        assert not hasattr(args, name)


def test_parse_args_rejects_removed_context_tuning_flags():
    script = load_context_script()

    for flag in (
        "--reference-box",
        "--reference-mask-source",
        "--candidate-count",
        "--max-masks",
        "--target-point",
        "--output",
    ):
        with pytest.raises(SystemExit):
            script.parse_args([flag, "1"])


def test_resolve_paths_uses_single_context_output(tmp_path):
    script = load_context_script()
    args = script.parse_args(["--output-dir", "outputs/context_check"])

    paths = script.resolve_paths(args, root=tmp_path)

    assert paths.reference_image == tmp_path / "asset" / "frog_reference.jpg"
    assert paths.target_image == tmp_path / "asset" / "frog_target.jpg"
    assert paths.checkpoint == tmp_path / "weight" / "sam3.1_multiplex.pt"
    assert paths.output_dir == tmp_path / "outputs" / "context_check"
    assert paths.output == tmp_path / "outputs" / "context_check" / "context.png"


def test_reference_context_description_shows_sam_mask_as_context_input():
    script = load_context_script()

    context = script.describe_reference_context()

    assert context == {
        "type": "sam_mask",
        "mask_source": "sam_box_prompt",
        "source_prompt": {
            "type": "box",
            "box": {
                "x0": 270.0,
                "y0": 450.0,
                "x1": 610.0,
                "y1": 900.0,
            },
        },
    }


def test_summarize_reference_mask_reports_mask_stats_only():
    script = load_context_script()
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:9] = True
    result = script.ReferenceMaskResult(
        mask=mask,
        score=0.9,
        selected_index=1,
        refined_score=0.8,
    )

    summary = script.summarize_reference_mask(result)

    assert summary == {
        "area": 24,
        "score": 0.9,
        "selected_index": 1,
        "refined_score": 0.8,
    }


def test_summarize_predictions_returns_every_context_prediction():
    script = load_context_script()
    predictions = [
        _prediction((1, 1, 3, 3), score=0.7),
        _prediction((4, 4, 7, 7), score=0.8),
    ]

    summary = script.summarize_predictions(predictions)

    assert len(summary) == 2
    assert summary[0]["bbox"] == [1, 1, 3, 3]
    assert summary[1]["bbox"] == [4, 4, 7, 7]


def test_apply_visual_nms_keeps_best_non_overlapping_predictions():
    script = load_context_script()
    predictions = [
        _prediction((1, 1, 6, 6), score=0.9),
        _prediction((2, 2, 7, 7), score=0.8),
        _prediction((7, 1, 9, 3), score=0.7),
    ]

    kept = script.apply_visual_nms(predictions)

    assert [prediction.bbox for prediction in kept] == [
        (1, 1, 6, 6),
        (7, 1, 9, 3),
    ]


def test_predict_reference_mask_uses_stateless_embedding_api():
    script = load_context_script()
    predictor = FakeStatelessPredictor()

    result = script.predict_reference_mask(predictor, Image.new("RGB", (10, 8)))

    assert result.selected_index == 1
    assert result.score == pytest.approx(0.9)
    assert result.refined_score == pytest.approx(0.8)
    assert int(result.mask.sum()) == 9
    assert predictor.encode_image_calls == 1
    assert len(predictor.predict_from_embedding_calls) == 2
    assert predictor.predict_from_embedding_calls[0].get("mask_input") is None
    assert predictor.predict_from_embedding_calls[1]["mask_input"].shape == (4, 4)
    assert not hasattr(predictor, "set_image")
    assert not hasattr(predictor, "predict")


def test_save_context_visualization_writes_reference_target_and_nms_sheet(tmp_path):
    script = load_context_script()
    reference = Image.new("RGB", (10, 8), (30, 40, 50))
    target = Image.new("RGB", (10, 8), (60, 70, 80))
    reference_mask = np.zeros((8, 10), dtype=bool)
    reference_mask[2:6, 3:9] = True
    predictions = [_prediction((2, 2, 6, 6), score=0.9)]
    nms_predictions = [_prediction((3, 2, 7, 6), score=0.8)]
    output_path = tmp_path / "context.png"

    script.save_context_visualization(
        reference,
        reference_mask,
        target,
        predictions,
        nms_predictions,
        output_path,
    )

    saved = Image.open(output_path)
    assert saved.size == (1380, 416)


class FakeStatelessPredictor:
    def __init__(self) -> None:
        self.encode_image_calls = 0
        self.predict_from_embedding_calls = []

    def encode_image(self, image):
        self.encode_image_calls += 1
        return object()

    def predict_from_embedding(self, embedding, **kwargs):
        self.predict_from_embedding_calls.append(kwargs)
        if len(self.predict_from_embedding_calls) == 1:
            masks = np.zeros((2, 8, 10), dtype=bool)
            masks[0, 1:3, 1:3] = True
            masks[1, 2:6, 3:9] = True
            scores = np.array([0.1, 0.9], dtype=np.float32)
            low_res = np.zeros((2, 4, 4), dtype=np.float32)
            low_res[1, 1:3, 1:3] = 2.0
            return masks, scores, low_res

        masks = np.zeros((1, 8, 10), dtype=bool)
        masks[0, 2:5, 3:6] = True
        scores = np.array([0.8], dtype=np.float32)
        low_res = np.zeros((1, 4, 4), dtype=np.float32)
        return masks, scores, low_res


def _prediction(
    bbox: tuple[int, int, int, int],
    *,
    score: float,
) -> ContextPrediction:
    x0, y0, x1, y1 = bbox
    return ContextPrediction(
        segmentation=np.ones((y1 - y0, x1 - x0), dtype=bool),
        bbox=bbox,
        area=(y1 - y0) * (x1 - x0),
        point_coords=(float(x0 + 1), float(y0 + 1)),
        context_score=score,
        predicted_iou=0.8,
        stability_score=0.7,
        score=score,
        image_size=(10, 8),
    )
