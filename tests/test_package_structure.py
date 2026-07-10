from pathlib import Path

from src.predict import GridPredictor, GroundPredictor, SinglePredictor, VideoPredictor


def test_predict_package_exports_workflows():
    assert SinglePredictor.__name__ == "SinglePredictor"
    assert GridPredictor.__name__ == "GridPredictor"
    assert GroundPredictor.__name__ == "GroundPredictor"
    assert VideoPredictor.__name__ == "VideoPredictor"


def test_core_packages_are_grouped_by_responsibility():
    root = Path(__file__).resolve().parents[1]
    paths = (
        "src/ops/box.py",
        "src/ops/tensor.py",
        "src/ml/runtime/attention.py",
        "src/ml/runtime/checkpointing.py",
        "src/ml/runtime/fused.py",
        "src/ml/components/nn/attention.py",
        "src/ml/components/nn/layers.py",
        "src/ml/components/nn/position.py",
        "src/ml/components/backbone/vit.py",
        "src/ml/components/backbone/neck.py",
        "src/ml/components/transformer/encoder.py",
        "src/ml/components/transformer/decoder.py",
        "src/ml/components/transformer/video.py",
        "src/ml/components/transformer/model.py",
        "src/ml/blocks/vision.py",
        "src/ml/blocks/image/features.py",
        "src/ml/blocks/image/prompt.py",
        "src/ml/blocks/image/masks.py",
        "src/ml/blocks/grounding/tokens.py",
        "src/ml/blocks/grounding/image.py",
        "src/ml/blocks/grounding/prompt.py",
        "src/ml/blocks/grounding/decoder.py",
        "src/ml/model/image.py",
        "src/ml/model/grounding.py",
        "src/ml/model/video/model.py",
        "src/ml/model/video/runtime.py",
        "src/ml/model/video/state.py",
        "src/build.py",
    )

    assert all((root / path).is_file() for path in paths)
