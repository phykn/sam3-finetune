import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
from src.types import MaskInstance

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "refine.py"


def load_refine_script():
    spec = importlib.util.spec_from_file_location("refine_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_grid_mask_refine_defaults():
    script = load_refine_script()

    args = script.parse_args([])

    assert args.image == "asset/frog_target.jpg"
    assert args.checkpoint == "weight/sam3.1_multiplex.pt"
    assert args.output_dir == "outputs/refine"
    assert args.device == "cuda"
    assert args.tiles == [1, 2]
    assert args.points_per_side == [32, 16]
    assert args.grid_max_masks == 100
    assert args.max_masks == 8
    assert args.show_masks == 1
    assert args.refine_batch_size == 8
    assert args.refine_multimask is False
    assert args.mask_foreground == 4.0
    assert args.mask_background == -4.0


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
    args = script.parse_args(["--overlap", "0.3", "--grid-max-masks", "25"])
    configs = [
        script.TileConfig(tile=1, points_per_side=32),
        script.TileConfig(tile=2, points_per_side=16),
    ]

    kwargs = script.build_grid_kwargs(args, configs)

    assert kwargs["points_per_side"] == 32
    assert kwargs["crop_grids"] == [1, 2]
    assert kwargs["crop_points_per_side"] == [32, 16]
    assert kwargs["crop_overlap_ratio"] == 0.3
    assert kwargs["max_masks"] == 25


def test_build_refiner_kwargs_exposes_mask_refine_settings():
    script = load_refine_script()
    args = script.parse_args(
        [
            "--refine-batch-size",
            "4",
            "--refine-multimask",
            "--mask-foreground",
            "5",
            "--mask-background",
            "-5",
        ]
    )

    kwargs = script.build_refiner_kwargs(args)

    assert kwargs["batch_size"] == 4
    assert kwargs["multimask_output"] is True
    assert kwargs["mask_foreground"] == 5.0
    assert kwargs["mask_background"] == -5.0


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

    summary = script.summarize_instances([instance], top_k=1)

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
    refined = MaskInstance(
        segmentation=np.ones((3, 3), dtype=bool),
        bbox=(2, 2, 5, 5),
        area=9,
        score=1.0,
        source="grid_refined",
        base_score=0.5,
        predicted_iou=1.0,
        stability_score=0.7,
        image_size=(8, 8),
    )
    output_path = tmp_path / "refine.png"

    script.save_refine_visualization(
        image,
        [base],
        [refined],
        output_path,
        max_masks=1,
    )

    saved = Image.open(output_path)
    assert saved.size[0] > image.width
    assert saved.size[1] > image.height
