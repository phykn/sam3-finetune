import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from src.types import ContextPrediction, MaskInstance

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "refine.py"


def load_refine_script():
    spec = importlib.util.spec_from_file_location("refine_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_simple_grid_context_refine_defaults():
    script = load_refine_script()

    args = script.parse_args([])

    assert args.image == "asset/frog_target.jpg"
    assert args.checkpoint == "weight/sam3.1_multiplex.pt"
    assert args.output_dir == "outputs/refine"
    assert args.device == "cuda"
    assert args.tiles == [1, 2]
    assert args.points_per_side == [32, 16]
    for name in (
        "points_per_batch",
        "grid_max_masks",
        "max_masks",
        "max_masks_per_crop",
        "pred_iou_thresh",
        "stability_score_thresh",
        "box_nms_thresh",
        "crop_nms_thresh",
        "keep_crop_edge_masks",
        "image_batch_size",
        "prompt_batch_size",
        "allow_cross_crop_prompt_decode",
        "feature_layer",
        "candidate_count",
        "decode_batch_size",
        "min_cell_distance",
        "mask_nms_thresh",
        "candidate_score_mode",
        "context_score_weight",
        "predicted_iou_weight",
        "stability_score_weight",
        "area_score_weight",
        "negative_context_mode",
        "negative_context_weight",
        "negative_context_scale",
        "min_context_score",
        "min_mask_area",
        "show_masks",
        "top_k",
    ):
        assert not hasattr(args, name)


def test_parse_args_rejects_removed_context_tuning_flags():
    script = load_refine_script()

    removed_flags = [
        "--candidate-count",
        "--max-masks",
        "--show-masks",
        "--top-k",
        "--grid-max-masks",
    ]
    for flag in removed_flags:
        with pytest.raises(SystemExit):
            script.parse_args([flag, "4"])


def test_resolve_paths_uses_single_refine_output(tmp_path):
    script = load_refine_script()
    args = script.parse_args(["--output-dir", "outputs/refine_check"])

    paths = script.resolve_paths(args, root=tmp_path)

    assert paths.image == tmp_path / "asset" / "frog_target.jpg"
    assert paths.checkpoint == tmp_path / "weight" / "sam3.1_multiplex.pt"
    assert paths.output_dir == tmp_path / "outputs" / "refine_check"
    assert paths.output == tmp_path / "outputs" / "refine_check" / "refine.png"


def test_build_grid_kwargs_combines_tile_candidates():
    script = load_refine_script()
    args = script.parse_args(["--overlap", "0.3"])
    configs = [
        script.TileConfig(tile=1, points_per_side=32),
        script.TileConfig(tile=2, points_per_side=16),
    ]

    kwargs = script.build_grid_kwargs(args, configs)

    assert kwargs == {
        "points_per_side": 32,
        "crop_grids": [1, 2],
        "crop_points_per_side": [32, 16],
        "crop_overlap_ratio": 0.3,
    }


def test_summarize_instances_includes_context_and_base_scores():
    script = load_refine_script()
    instance = MaskInstance(
        segmentation=np.ones((2, 3), dtype=bool),
        bbox=(1, 2, 4, 4),
        area=6,
        score=1.2,
        source="reference_guided",
        concept_id=1,
        context_score=1.1,
        base_score=0.9,
        predicted_iou=0.8,
        stability_score=0.7,
        point_coords=(3.0, 4.0),
        crop_box=(0, 0, 8, 8),
        crop_grid=2,
        crop_index=1,
        image_size=(8, 8),
    )

    summary = script.summarize_instances([instance])

    assert summary == [
        {
            "bbox": [1, 2, 4, 4],
            "area": 6,
            "score": 1.2,
            "context_score": 1.1,
            "base_score": 0.9,
            "predicted_iou": 0.8,
            "stability_score": 0.7,
            "point_coords": [3.0, 4.0],
            "crop_grid": 2,
            "crop_index": 1,
        }
    ]


def test_summarize_predictions_includes_context_scores():
    script = load_refine_script()
    prediction = ContextPrediction(
        segmentation=np.ones((2, 3), dtype=bool),
        bbox=(1, 2, 4, 4),
        area=6,
        point_coords=(3.0, 4.0),
        context_score=1.1,
        predicted_iou=0.8,
        stability_score=0.7,
        score=1.2,
        image_size=(8, 8),
        area_score=0.6,
    )

    summary = script.summarize_predictions([prediction])

    assert summary == [
        {
            "bbox": [1, 2, 4, 4],
            "area": 6,
            "score": 1.2,
            "context_score": 1.1,
            "predicted_iou": 0.8,
            "stability_score": 0.7,
            "area_score": 0.6,
            "point_coords": [3.0, 4.0],
        }
    ]


def test_summaries_return_every_result_without_top_k_limit():
    script = load_refine_script()
    instance_a = MaskInstance(
        segmentation=np.ones((1, 1), dtype=bool),
        bbox=(1, 1, 2, 2),
        area=1,
        score=0.5,
        image_size=(8, 8),
    )
    instance_b = MaskInstance(
        segmentation=np.ones((1, 1), dtype=bool),
        bbox=(3, 3, 4, 4),
        area=1,
        score=0.6,
        image_size=(8, 8),
    )
    prediction_a = ContextPrediction(
        segmentation=np.ones((1, 1), dtype=bool),
        bbox=(1, 1, 2, 2),
        area=1,
        point_coords=(1.5, 1.5),
        context_score=0.5,
        predicted_iou=0.8,
        stability_score=0.7,
        score=0.9,
        image_size=(8, 8),
    )
    prediction_b = ContextPrediction(
        segmentation=np.ones((1, 1), dtype=bool),
        bbox=(3, 3, 4, 4),
        area=1,
        point_coords=(3.5, 3.5),
        context_score=0.6,
        predicted_iou=0.8,
        stability_score=0.7,
        score=1.0,
        image_size=(8, 8),
    )

    assert len(script.summarize_instances([instance_a, instance_b])) == 2
    assert len(script.summarize_predictions([prediction_a, prediction_b])) == 2


def test_save_refine_visualization_writes_single_sheet(tmp_path):
    script = load_refine_script()
    image = Image.new("RGB", (8, 8), (40, 50, 60))
    base = MaskInstance(
        segmentation=np.ones((2, 2), dtype=bool),
        bbox=(1, 1, 3, 3),
        area=4,
        score=0.5,
        image_size=(8, 8),
    )
    refined = ContextPrediction(
        segmentation=np.ones((3, 3), dtype=bool),
        bbox=(2, 2, 5, 5),
        area=9,
        point_coords=(3.0, 3.0),
        context_score=0.9,
        predicted_iou=0.8,
        stability_score=0.7,
        score=1.0,
        image_size=(8, 8),
    )
    output_path = tmp_path / "refine.png"

    script.save_refine_visualization(
        image,
        [base],
        [refined],
        output_path,
    )

    saved = Image.open(output_path)
    assert saved.size[0] > image.width
    assert saved.size[1] > image.height
