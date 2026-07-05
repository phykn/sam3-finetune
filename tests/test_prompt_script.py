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


def test_parse_args_uses_single_point_defaults():
    script = load_prompt_script()

    args = script.parse_args([])

    assert args.image == "asset/sample.jpg"
    assert args.checkpoint == "weight/sam3.1_multiplex.pt"
    assert args.output_dir == "outputs"
    assert args.device == "cuda"
    assert args.x == 195.0
    assert args.y == 295.0


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

    prompt = script.describe_prompt(coords, labels)

    assert prompt == {
        "type": "single_positive_point",
        "points": [{"x": 12.5, "y": 30.0, "label": 1}],
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
    assert paths.mask == tmp_path / "outputs" / "prompt_check" / "prompt_mask.png"
    assert paths.overlay == tmp_path / "outputs" / "prompt_check" / "prompt_overlay.png"
