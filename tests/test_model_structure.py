from src.ml.model import Sam3GroundingModel, Sam3ImageModel, Sam3VideoModel


def test_models_are_split_by_workflow():
    assert Sam3ImageModel.__module__ == "src.ml.model.image"
    assert Sam3GroundingModel.__module__ == "src.ml.model.grounding"
    assert Sam3VideoModel.__module__ == "src.ml.model.video.model"
