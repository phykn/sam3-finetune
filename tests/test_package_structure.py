from pathlib import Path


def test_shared_modules_are_grouped_by_responsibility() -> None:
    root = Path(__file__).resolve().parents[1]

    for path in (
        "src/types.py",
        "src/model/nn/activation_checkpoint.py",
        "src/model/nn/decoder.py",
        "src/model/nn/encoder.py",
        "src/model/nn/fused.py",
        "src/model/nn/modules.py",
        "src/model/nn/position.py",
        "src/model/backbone/vit.py",
        "src/model/sam/prompt_encoder.py",
        "src/model/image/builder.py",
        "src/model/grounding/builder.py",
        "src/model/video/builder.py",
        "src/predict/image.py",
        "src/predict/masks/generator.py",
        "src/predict/reference/matcher.py",
        "src/predict/grounding/inference.py",
        "src/predict/video.py",
        "src/checkpoint/loader.py",
        "src/transforms/image.py",
        "src/io/video.py",
        "src/ops/box.py",
        "src/data/structures.py",
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
        "checkpoint.py",
        "transforms.py",
        "io_utils.py",
    ):
        assert not (root / "src" / filename).exists()
    for dirname in (
        "backbone",
        "sam",
        "nn",
        "image",
        "masks",
        "context",
        "grounding",
        "video",
    ):
        assert not (root / "src" / dirname).exists()
