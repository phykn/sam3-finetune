import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from src.types import MaskProposal

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "grid.py"


def load_grid_script():
    spec = importlib.util.spec_from_file_location("grid_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_simple_frog_tile_defaults():
    script = load_grid_script()

    args = script.parse_args([])

    assert args.image == "asset/frog_target.jpg"
    assert args.checkpoint == "weight/sam3.1_multiplex.pt"
    assert args.output_dir == "outputs/grid"
    assert args.device == "cuda"
    assert args.tiles == [1, 2]
    assert args.points_per_side == [32, 16]
    assert args.overlap == 0.25
    for name in (
        "points_per_batch",
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
        "show_masks",
        "save_extra",
        "overlay_max_masks",
        "grid_max_masks",
        "grid_columns",
        "top_k",
    ):
        assert not hasattr(args, name)


def test_parse_args_rejects_removed_grid_tuning_flags():
    script = load_grid_script()

    for flag in ("--max-masks", "--show-masks", "--top-k", "--save-extra"):
        with pytest.raises(SystemExit):
            script.parse_args([flag, "1"])


def test_resolve_tile_configs_pairs_tiles_and_points():
    script = load_grid_script()
    args = script.parse_args(
        ["--tiles", "1", "2", "3", "--points-per-side", "20", "10", "8"]
    )

    configs = script.resolve_tile_configs(args)

    assert [(config.tile, config.points_per_side) for config in configs] == [
        (1, 20),
        (2, 10),
        (3, 8),
    ]


def test_resolve_tile_configs_reuses_single_points_value():
    script = load_grid_script()
    args = script.parse_args(["--tiles", "1", "2", "--points-per-side", "12"])

    configs = script.resolve_tile_configs(args)

    assert [(config.tile, config.points_per_side) for config in configs] == [
        (1, 12),
        (2, 12),
    ]


def test_resolve_tile_configs_rejects_mismatched_points_values():
    script = load_grid_script()
    args = script.parse_args(
        ["--tiles", "1", "2", "--points-per-side", "32", "16", "8"]
    )

    try:
        script.resolve_tile_configs(args)
    except ValueError as exc:
        assert "points-per-side" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_resolve_paths_uses_single_grid_output(tmp_path):
    script = load_grid_script()
    args = script.parse_args(["--output-dir", "outputs/grid_check"])

    paths = script.resolve_paths(args, root=tmp_path)

    assert paths.image == tmp_path / "asset" / "frog_target.jpg"
    assert paths.checkpoint == tmp_path / "weight" / "sam3.1_multiplex.pt"
    assert paths.output_dir == tmp_path / "outputs" / "grid_check"
    assert paths.output == tmp_path / "outputs" / "grid_check" / "grid.png"


def test_build_generator_kwargs_passes_only_tile_settings():
    script = load_grid_script()
    args = script.parse_args(["--overlap", "0.3"])
    configs = [
        script.TileConfig(tile=1, points_per_side=32),
        script.TileConfig(tile=2, points_per_side=16),
    ]

    kwargs = script.build_generator_kwargs(args, configs)

    assert kwargs == {
        "points_per_side": 32,
        "crop_grids": [1, 2],
        "crop_points_per_side": [32, 16],
        "crop_overlap_ratio": 0.3,
    }


def test_summarize_proposals_returns_every_result():
    script = load_grid_script()
    proposals = [
        _proposal((1, 1, 3, 3), score=0.9),
        _proposal((4, 4, 7, 7), score=0.8),
    ]

    summary = script.summarize_proposals(proposals)

    assert len(summary) == 2
    assert summary[0]["bbox"] == [1, 1, 3, 3]
    assert summary[1]["bbox"] == [4, 4, 7, 7]


def test_build_grid_points_combines_all_tile_prompts():
    script = load_grid_script()
    configs = [
        script.TileConfig(tile=1, points_per_side=2),
        script.TileConfig(tile=2, points_per_side=1),
    ]

    points = script.build_grid_points((8, 8), configs, overlap=0.25)

    assert len(points) == 8
    assert sum(1 for _x, _y, tile in points if tile == 1) == 4
    assert sum(1 for _x, _y, tile in points if tile == 2) == 4


def test_save_grid_visualization_writes_all_results_on_one_sheet(tmp_path):
    script = load_grid_script()
    image = Image.new("RGB", (8, 8), (20, 30, 40))
    proposals = [
        _proposal((2, 2, 6, 6), score=0.9),
        _proposal((0, 0, 3, 3), score=0.8),
    ]
    output_path = tmp_path / "grid.png"

    script.save_grid_visualization(
        image,
        proposals,
        [(4.0, 4.0, 1), (1.0, 1.0, 2)],
        output_path,
    )

    saved = Image.open(output_path)
    assert saved.size[0] > image.width
    assert saved.size[1] > image.height


def _proposal(
    bbox: tuple[int, int, int, int],
    *,
    score: float,
) -> MaskProposal:
    x0, y0, x1, y1 = bbox
    return MaskProposal(
        segmentation=np.ones((y1 - y0, x1 - x0), dtype=bool),
        bbox=bbox,
        area=(y1 - y0) * (x1 - x0),
        predicted_iou=score,
        stability_score=0.8,
        point_coords=(float(x0 + 1), float(y0 + 1)),
        crop_box=(0, 0, 8, 8),
        crop_grid=1,
        crop_index=0,
        image_size=(8, 8),
    )
