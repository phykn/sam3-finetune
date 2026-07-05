import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prompt.py"


def load_prompt_script():
    spec = importlib.util.spec_from_file_location("prompt_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_simple_frog_defaults():
    script = load_prompt_script()

    args = script.parse_args([])

    assert args.image == "asset/frog_target.jpg"
    assert args.checkpoint == "weight/sam3.1_multiplex.pt"
    assert args.output_dir == "outputs/prompt"
    assert args.device == "cuda"
    for name in ("case", "x", "y", "neg_x", "neg_y", "box"):
        assert not hasattr(args, name)


def test_parse_args_rejects_removed_prompt_tuning_flags():
    script = load_prompt_script()

    for flag in ("--case", "--x", "--y", "--neg-x", "--neg-y", "--box"):
        with pytest.raises(SystemExit):
            script.parse_args([flag, "1"])


def test_build_prompt_cases_always_returns_all_fixed_cases():
    script = load_prompt_script()

    cases = script.build_prompt_cases(image_size=(1280, 896))

    assert [case.name for case in cases] == [
        "point",
        "points",
        "box",
        "point_box",
        "mask",
    ]
    assert cases[0].prompt == {
        "type": "point",
        "points": [{"x": 560.0, "y": 500.0, "label": 1}],
    }
    assert cases[1].prompt["points"] == [
        {"x": 560.0, "y": 500.0, "label": 1},
        {"x": 300.0, "y": 430.0, "label": 0},
    ]
    assert cases[2].prompt["box"] == {
        "x0": 380.0,
        "y0": 270.0,
        "x1": 790.0,
        "y1": 705.0,
    }
    assert cases[4].mask_input.shape == (896, 1280)
    assert cases[4].prompt["mask_source"] == "filled_box"
    assert cases[4].prompt["mask_area"] == 178350


def test_build_filled_box_mask_uses_box_as_binary_mask_prompt():
    script = load_prompt_script()

    mask = script.build_filled_box_mask((10, 8), [2.0, 1.0, 8.0, 7.0])

    assert mask.shape == (8, 10)
    assert mask.dtype == np.float32
    assert int(mask.sum()) == 36
    assert mask[0].sum() == 0
    assert mask[:, 0].sum() == 0
    assert mask[1:7, 2:8].all()


def test_resolve_paths_uses_single_prompt_output(tmp_path):
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
    assert paths.output == tmp_path / "outputs" / "prompt_check" / "prompt.png"


def test_save_prompt_visualization_writes_one_sheet(tmp_path):
    script = load_prompt_script()
    image = Image.new("RGB", (1280, 896), (40, 50, 60))
    cases = script.build_prompt_cases(image_size=image.size)
    results = [
        script.PromptResult(
            case=case,
            mask=np.ones((8, 8), dtype=bool),
            score=0.8,
            mask_shape=(1, 8, 8),
            low_res_shape=(1, 4, 4),
        )
        for case in cases[:2]
    ]
    output_path = tmp_path / "prompt.png"

    script.save_prompt_visualization(image, results, output_path)

    saved = Image.open(output_path)
    assert saved.size[0] > 0
    assert saved.size[1] > 0
