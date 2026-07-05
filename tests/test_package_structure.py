from pathlib import Path


def test_shared_modules_are_grouped_by_responsibility() -> None:
    root = Path(__file__).resolve().parents[1]

    for path in (
        "src/model/types.py",
        "src/predict/image_types.py",
        "src/predict/video_types.py",
        "src/predict/masks/types.py",
        "src/predict/reference/types.py",
        "src/predict/grounding/types.py",
        "src/model/components/nn/activation.py",
        "src/model/components/nn/modules.py",
        "src/model/components/nn/position.py",
        "src/model/runtime/attention.py",
        "src/model/runtime/checkpointing.py",
        "src/model/runtime/fused.py",
        "src/model/components/transformer/decoder.py",
        "src/model/components/transformer/encoder.py",
        "src/model/components/transformer/wrapper.py",
        "src/model/grounding/output.py",
        "src/model/grounding/scoring.py",
        "src/model/grounding/create.py",
        "src/model/components/backbone/create.py",
        "src/model/components/backbone/vit.py",
        "src/model/components/sam/prompt_encoder.py",
        "src/model/build.py",
        "src/model/model.py",
        "src/model/image/model.py",
        "src/model/video/model.py",
        "src/io/checkpoint.py",
        "src/predict/image.py",
        "src/predict/masks/generator.py",
        "src/predict/reference/matcher.py",
        "src/predict/grounding/inference.py",
        "src/predict/video.py",
        "src/predict/image_transform.py",
        "src/io/load.py",
        "src/ops/box.py",
        "src/ops/mask.py",
        "src/ops/tensor.py",
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
        "structures.py",
        "builder.py",
        "predictor.py",
        "checkpoint.py",
        "transforms.py",
        "io_utils.py",
        "types.py",
        "utils.py",
    ):
        assert not (root / "src" / filename).exists()
    for dirname in (
        "backbone",
        "sam",
        "nn",
        "image",
        "masks",
        "metrics",
        "context",
        "checkpoint",
        "grounding",
        "video",
        "transforms",
        "runtime",
    ):
        assert not (root / "src" / dirname).exists()
    for dirname in ("backbone", "sam", "nn", "transformer"):
        assert not (root / "src" / "model" / dirname).exists()
    assert not (root / "src" / "model" / "image" / "builder.py").exists()
    assert not (root / "src" / "model" / "grounding" / "builder.py").exists()
    assert not (root / "src" / "model" / "video" / "builder.py").exists()
    assert not (root / "src" / "model" / "image" / "checkpoint.py").exists()
    assert not (root / "src" / "model" / "video" / "checkpoint.py").exists()
