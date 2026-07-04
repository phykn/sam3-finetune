from pathlib import Path


def test_shared_modules_are_grouped_by_responsibility() -> None:
    root = Path(__file__).resolve().parents[1]

    for path in (
        "src/nn/activation_checkpoint.py",
        "src/nn/decoder.py",
        "src/nn/encoder.py",
        "src/nn/fused.py",
        "src/nn/modules.py",
        "src/nn/position.py",
        "src/ops/box.py",
        "src/data/structures.py",
        "src/image/builder.py",
        "src/image/predictor.py",
        "src/image/types.py",
    ):
        assert (root / path).is_file()

    for filename in (
        "act_ckpt_utils.py",
        "decoder.py",
        "encoder.py",
        "fused.py",
        "model_misc.py",
        "position_encoding.py",
        "box_ops.py",
        "data_misc.py",
        "builder.py",
        "predictor.py",
    ):
        assert not (root / "src" / filename).exists()
