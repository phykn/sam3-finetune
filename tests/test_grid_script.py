import importlib.util
from pathlib import Path

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


def test_resolve_paths_uses_tile_output_names(tmp_path):
    script = load_grid_script()
    args = script.parse_args(["--output-dir", "outputs/grid_check"])

    paths = script.resolve_paths(args, root=tmp_path)

    assert paths.image == tmp_path / "asset" / "frog_target.jpg"
    assert paths.checkpoint == tmp_path / "weight" / "sam3.1_multiplex.pt"
    assert paths.output_dir == tmp_path / "outputs" / "grid_check"
    assert (
        paths.overlay_for(2)
        == tmp_path / "outputs" / "grid_check" / "tile2_overlay.png"
    )
    assert paths.grid_for(2) == tmp_path / "outputs" / "grid_check" / "tile2_grid.png"


def test_build_generator_kwargs_passes_overlap_and_edge_filter():
    script = load_grid_script()
    args = script.parse_args(["--overlap", "0.3", "--max-masks", "25"])
    config = script.TileConfig(tile=2, points_per_side=16)

    kwargs = script.build_generator_kwargs(args, config)

    assert kwargs["crop_grids"] == [2]
    assert kwargs["crop_points_per_side"] == [16]
    assert kwargs["crop_overlap_ratio"] == 0.3
    assert kwargs["filter_crop_edge_masks"] is True
    assert kwargs["max_masks"] == 25


def test_build_generator_kwargs_can_keep_crop_edge_masks():
    script = load_grid_script()
    args = script.parse_args(["--keep-crop-edge-masks"])
    config = script.TileConfig(tile=2, points_per_side=16)

    kwargs = script.build_generator_kwargs(args, config)

    assert kwargs["filter_crop_edge_masks"] is False
