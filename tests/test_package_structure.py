from pathlib import Path


def test_shared_modules_are_grouped_by_responsibility() -> None:
    root = Path(__file__).resolve().parents[1]

    for path in (
        "src/types.py",
        "src/model/structures.py",
        "src/model/components/nn/activation.py",
        "src/model/components/nn/modules.py",
        "src/model/components/nn/position.py",
        "src/model/runtime/attention.py",
        "src/model/runtime/checkpointing.py",
        "src/model/runtime/fused.py",
        "src/model/components/transformer/decoder.py",
        "src/model/components/transformer/encoder.py",
        "src/model/components/transformer/wrapper.py",
        "src/model/components/grounding/box_out.py",
        "src/model/components/grounding/scoring.py",
        "src/model/components/grounding/create.py",
        "src/model/components/grounding/geometry.py",
        "src/model/components/video/create.py",
        "src/model/components/video/mask_selection.py",
        "src/model/components/video/memory.py",
        "src/model/components/video/mlp.py",
        "src/model/components/video/multiplex.py",
        "src/model/components/video/multiplex_ops.py",
        "src/model/blocks/ground_image.py",
        "src/model/blocks/ground_prompt.py",
        "src/model/blocks/ground_dec.py",
        "src/model/components/backbone/create.py",
        "src/model/components/backbone/vit.py",
        "src/model/components/sam/prompt_encoder.py",
        "src/build.py",
        "config/model.yaml",
        "src/model/model.py",
        "src/model/video/model.py",
        "src/io/checkpoint.py",
        "src/predict/single.py",
        "src/predict/transform.py",
        "src/predict/prompt.py",
        "src/predict/result.py",
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
    assert not (root / "src" / "model" / "build.py").exists()
    assert not (root / "src" / "model" / "image" / "builder.py").exists()
    assert not (root / "src" / "model" / "image").exists()
    assert not (root / "src" / "model" / "sam3.py").exists()
    assert not (root / "src" / "model" / "grounding").exists()
    assert not (root / "src" / "model" / "language").exists()
    assert not (
        root / "src" / "model" / "components" / "grounding" / "output.py"
    ).exists()
    assert not (
        root / "src" / "model" / "components" / "grounding" / "encoder.py"
    ).exists()
    assert not (
        root / "src" / "model" / "components" / "grounding" / "mask_encoder.py"
    ).exists()
    assert not (root / "src" / "model" / "video" / "builder.py").exists()
    assert not (root / "src" / "model" / "image" / "checkpoint.py").exists()
    assert not (root / "src" / "model" / "video" / "checkpoint.py").exists()
    assert not (root / "src" / "model" / "types.py").exists()
    assert not (root / "src" / "predict" / "image_types.py").exists()
    assert not (root / "src" / "predict" / "video_types.py").exists()
    assert not (root / "src" / "predict" / "masks" / "types.py").exists()
    assert not (root / "src" / "predict" / "reference" / "types.py").exists()
    assert not (root / "src" / "predict" / "grounding" / "types.py").exists()
    assert not (root / "src" / "predict" / "image.py").exists()
    assert not (root / "src" / "predict" / "image_transform.py").exists()
    assert not (root / "src" / "predict" / "video.py").exists()
    assert not (root / "src" / "predict" / "prompted").exists()
    assert not (root / "src" / "predict" / "grid").exists()
    assert not (root / "src" / "predict" / "context").exists()
    assert not (root / "src" / "predict" / "refine").exists()
    assert not (root / "src" / "predict" / "next_frame").exists()
    assert not (root / "src" / "predict" / "grounding").exists()
    assert not (root / "src" / "predict" / "visual_prompt").exists()
    assert not (root / "src" / "predict" / "masks" / "__init__.py").exists()
    assert not (root / "src" / "predict" / "masks" / "generator.py").exists()
    assert not (root / "src" / "predict" / "reference" / "__init__.py").exists()
    assert not (root / "src" / "predict" / "reference" / "matcher.py").exists()
