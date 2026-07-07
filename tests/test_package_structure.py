from pathlib import Path

from src.predict import GridPredictor, GroundPredictor, SinglePredictor, VideoPredictor


def test_predict_package_exports_workflows() -> None:
    assert SinglePredictor.__name__ == "SinglePredictor"
    assert GridPredictor.__name__ == "GridPredictor"
    assert GroundPredictor.__name__ == "GroundPredictor"
    assert VideoPredictor.__name__ == "VideoPredictor"


def test_shared_modules_are_grouped_by_responsibility() -> None:
    root = Path(__file__).resolve().parents[1]

    for path in (
        "src/ml/structures.py",
        "src/ml/components/nn/activation.py",
        "src/ml/components/nn/modules.py",
        "src/ml/components/nn/position.py",
        "src/ml/runtime/attention.py",
        "src/ml/runtime/checkpointing.py",
        "src/ml/runtime/fused.py",
        "src/ml/components/transformer/decoder.py",
        "src/ml/components/transformer/encoder.py",
        "src/ml/components/transformer/wrapper.py",
        "src/ml/components/grounding/box_out.py",
        "src/ml/components/grounding/scoring.py",
        "src/ml/components/grounding/create.py",
        "src/ml/components/grounding/geometry.py",
        "src/ml/components/video/create.py",
        "src/ml/components/video/frame.py",
        "src/ml/components/video/mask_selection.py",
        "src/ml/components/video/memory.py",
        "src/ml/components/video/mlp.py",
        "src/ml/components/video/tracking_model.py",
        "src/ml/components/video/multiplex.py",
        "src/ml/components/video/multiplex_ops.py",
        "src/ml/components/video/sam_heads.py",
        "src/ml/components/video/init_parts.py",
        "src/ml/blocks/ground_image.py",
        "src/ml/blocks/ground_prompt.py",
        "src/ml/blocks/ground_dec.py",
        "src/ml/blocks/video_feat.py",
        "src/ml/blocks/video_mem.py",
        "src/ml/blocks/video_track.py",
        "src/ml/components/backbone/create.py",
        "src/ml/components/backbone/vit.py",
        "src/ml/components/sam/prompt_encoder.py",
        "src/build.py",
        "config/model.yaml",
        "src/ml/model.py",
        "src/ml/components/video/tracker/model.py",
        "src/io/checkpoint.py",
        "src/data/__init__.py",
        "src/data/ground.py",
        "src/data/image.py",
        "src/data/prompt.py",
        "src/predict/ground.py",
        "src/predict/single.py",
        "src/predict/video.py",
        "src/predict/video_ops/session.py",
        "src/predict/grid.py",
        "src/predict/grid_ops/__init__.py",
        "src/predict/grid_ops/boxes.py",
        "src/predict/grid_ops/candidates.py",
        "src/predict/grid_ops/points.py",
        "src/predict/grid_ops/tiles.py",
        "src/predict/ground_ops/__init__.py",
        "src/predict/ground_ops/sim.py",
        "src/predict/mask/__init__.py",
        "src/predict/mask/format.py",
        "src/io/load.py",
        "src/ops/box.py",
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
        assert not (root / "src" / "ml" / dirname).exists()
    assert not (root / "src" / "ml" / "build.py").exists()
    assert not (root / "src" / "ml" / "image" / "builder.py").exists()
    assert not (root / "src" / "ml" / "image").exists()
    assert not (root / "src" / "ml" / "sam3.py").exists()
    assert not (root / "src" / "ml" / "grounding").exists()
    assert not (root / "src" / "ml" / "language").exists()
    assert not (root / "src" / "ml" / "video").exists()
    assert not (root / "src" / "ml" / "components" / "video" / "model.py").exists()
    assert not (
        root / "src" / "ml" / "components" / "video" / "memory_model.py"
    ).exists()
    assert not (root / "src" / "ml" / "components" / "video" / "sam.py").exists()
    assert not (root / "src" / "ml" / "components" / "video" / "setup.py").exists()
    assert not (root / "src" / "ml" / "blocks" / "video.py").exists()
    assert not (root / "src" / "ml" / "blocks" / "video_tracker.py").exists()
    assert not (root / "src" / "ml" / "components" / "grounding" / "output.py").exists()
    assert not (
        root / "src" / "ml" / "components" / "grounding" / "encoder.py"
    ).exists()
    assert not (
        root / "src" / "ml" / "components" / "grounding" / "mask_encoder.py"
    ).exists()
    assert not (root / "src" / "ml" / "video" / "builder.py").exists()
    assert not (root / "src" / "ml" / "image" / "checkpoint.py").exists()
    assert not (root / "src" / "ml" / "video" / "checkpoint.py").exists()
    assert not (root / "src" / "ml" / "types.py").exists()
    assert not (root / "src" / "predict" / "image_types.py").exists()
    assert not (root / "src" / "predict" / "video_types.py").exists()
    assert not (root / "src" / "predict" / "masks" / "types.py").exists()
    assert not (root / "src" / "predict" / "reference" / "types.py").exists()
    assert not (root / "src" / "predict" / "grounding" / "types.py").exists()
    assert not (root / "src" / "predict" / "image.py").exists()
    assert not (root / "src" / "predict" / "image_transform.py").exists()
    assert not (root / "src" / "predict" / "prompt.py").exists()
    assert not (root / "src" / "predict" / "result.py").exists()
    assert not (root / "src" / "predict" / "transform.py").exists()
    assert not (root / "src" / "predict" / "output.py").exists()
    assert not (root / "src" / "predict" / "grid_parts").exists()
    assert not (root / "src" / "predict" / "ground_parts").exists()
    assert not (root / "src" / "predict" / "video_parts").exists()
    assert not (root / "src" / "predict" / "video_parts" / "memory.py").exists()
    assert not (root / "src" / "data" / "transform.py").exists()
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
