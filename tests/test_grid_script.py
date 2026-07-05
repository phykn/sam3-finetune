import importlib.util
from pathlib import Path

import numpy as np
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


def test_parse_args_uses_frog_tile_defaults():
    script = load_grid_script()

    args = script.parse_args([])

    assert args.image == "asset/frog_target.jpg"
    assert args.checkpoint == "weight/sam3.1_multiplex.pt"
    assert args.output_dir == "outputs/grid"
    assert args.device == "cuda"
    assert args.tiles == [1, 2]
    assert args.points_per_side == [32, 16]
    assert args.overlap == 0.25
    assert args.max_masks == 100
    assert args.show_masks == 8
    assert args.save_extra is False


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


def test_resolve_paths_uses_combined_output_names(tmp_path):
    script = load_grid_script()
    args = script.parse_args(["--output-dir", "outputs/grid_check"])

    paths = script.resolve_paths(args, root=tmp_path)

    assert paths.image == tmp_path / "asset" / "frog_target.jpg"
    assert paths.checkpoint == tmp_path / "weight" / "sam3.1_multiplex.pt"
    assert paths.output_dir == tmp_path / "outputs" / "grid_check"
    assert paths.points == tmp_path / "outputs" / "grid_check" / "grid_points.png"
    assert (
        paths.extra_overlay == tmp_path / "outputs" / "grid_check" / "grid_overlay.png"
    )
    assert paths.extra_grid == tmp_path / "outputs" / "grid_check" / "grid_masks.png"


def test_build_generator_kwargs_passes_overlap_and_edge_filter():
    script = load_grid_script()
    args = script.parse_args(["--overlap", "0.3", "--max-masks", "25"])
    configs = [
        script.TileConfig(tile=1, points_per_side=32),
        script.TileConfig(tile=2, points_per_side=16),
    ]

    kwargs = script.build_generator_kwargs(args, configs)

    assert kwargs["points_per_side"] == 32
    assert kwargs["crop_grids"] == [1, 2]
    assert kwargs["crop_points_per_side"] == [32, 16]
    assert kwargs["crop_overlap_ratio"] == 0.3
    assert kwargs["filter_crop_edge_masks"] is True
    assert kwargs["max_masks"] == 25


def test_build_generator_kwargs_can_keep_crop_edge_masks():
    script = load_grid_script()
    args = script.parse_args(["--keep-crop-edge-masks"])
    configs = [script.TileConfig(tile=2, points_per_side=16)]

    kwargs = script.build_generator_kwargs(args, configs)

    assert kwargs["filter_crop_edge_masks"] is False


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


def test_save_grid_point_visualization_writes_combined_sheet(tmp_path):
    script = load_grid_script()
    image = Image.new("RGB", (8, 8), (20, 30, 40))
    proposal = MaskProposal(
        segmentation=np.ones((4, 4), dtype=bool),
        bbox=(2, 2, 6, 6),
        area=16,
        predicted_iou=0.9,
        stability_score=0.8,
        point_coords=(4.0, 4.0),
        crop_box=(0, 0, 8, 8),
        crop_grid=1,
        crop_index=0,
        image_size=(8, 8),
    )
    output_path = tmp_path / "grid_points.png"

    script.save_grid_point_visualization(
        image,
        [proposal],
        [(4.0, 4.0, 1)],
        output_path,
        show_masks=1,
    )

    saved = Image.open(output_path)
    assert saved.size[0] > image.width
    assert saved.size[1] > image.height
