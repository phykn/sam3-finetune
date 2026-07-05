import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prompt.py"


def load_prompt_script():
    spec = importlib.util.spec_from_file_location("prompt_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_frog_batch_defaults():
    script = load_prompt_script()

    args = script.parse_args([])

    assert args.image == "asset/frog_target.jpg"
    assert args.checkpoint == "weight/sam3.1_multiplex.pt"
    assert args.output_dir == "outputs/prompt"
    assert args.device == "cuda"
    assert args.x == 560.0
    assert args.y == 500.0
    assert args.neg_x == 300.0
    assert args.neg_y == 430.0
    assert args.box == [380.0, 270.0, 790.0, 705.0]
    assert script.resolve_cases(args.case) == [
        "point",
        "points",
        "box",
        "point_box",
        "mask",
    ]


def test_build_point_prompt_returns_one_positive_point():
    script = load_prompt_script()

    coords, labels = script.build_point_prompt(12.5, 30.0)

    np.testing.assert_allclose(
        coords,
        np.array([[12.5, 30.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(labels, np.array([1], dtype=np.int64))
    assert coords.dtype == np.float32
    assert labels.dtype == np.int64


def test_describe_prompt_makes_input_point_explicit():
    script = load_prompt_script()
    coords, labels = script.build_point_prompt(12.5, 30.0)

    prompt = script.describe_prompt(point_coords=coords, point_labels=labels)

    assert prompt == {
        "type": "point",
        "points": [{"x": 12.5, "y": 30.0, "label": 1}],
    }


def test_resolve_cases_allows_subset_and_expands_all():
    script = load_prompt_script()

    assert script.resolve_cases(["box", "mask"]) == ["box", "mask"]
    assert script.resolve_cases(["all"]) == [
        "point",
        "points",
        "box",
        "point_box",
        "mask",
    ]


def test_build_filled_box_mask_uses_box_as_binary_mask_prompt():
    script = load_prompt_script()

    mask = script.build_filled_box_mask((10, 8), [2.0, 1.0, 8.0, 7.0])

    assert mask.shape == (8, 10)
    assert mask.dtype == np.float32
    assert int(mask.sum()) == 36
    assert mask[0].sum() == 0
    assert mask[:, 0].sum() == 0
    assert mask[1:7, 2:8].all()


def test_build_prompt_case_can_use_filled_box_mask():
    script = load_prompt_script()
    args = script.parse_args([])

    prompt_case = script.build_prompt_case("mask", args, image_size=(1280, 896))

    assert prompt_case.name == "mask"
    assert prompt_case.point_coords is None
    assert prompt_case.point_labels is None
    assert prompt_case.box is None
    assert prompt_case.mask_input.shape == (896, 1280)
    assert prompt_case.prompt == {
        "type": "mask",
        "mask_source": "filled_box",
        "box": {"x0": 380.0, "y0": 270.0, "x1": 790.0, "y1": 705.0},
        "mask_input_shape": [896, 1280],
        "mask_area": 178350,
    }


def test_resolve_paths_uses_short_prompt_output_names(tmp_path):
    script = load_prompt_script()
    args = script.parse_args(
        [
            "--image",
            "asset/custom.jpg",
            "--checkpoint",
            "weight/custom.pt",
            "--output-dir",
            "outputs/prompt_check",
        ]
    )

    paths = script.resolve_paths(args, root=tmp_path)

    assert paths.image == tmp_path / "asset" / "custom.jpg"
    assert paths.checkpoint == tmp_path / "weight" / "custom.pt"
    assert paths.output_dir == tmp_path / "outputs" / "prompt_check"
    assert paths.mask_for("box") == (
        tmp_path / "outputs" / "prompt_check" / "box_mask.png"
    )
    assert paths.overlay_for("box") == (
        tmp_path / "outputs" / "prompt_check" / "box_overlay.png"
    )
