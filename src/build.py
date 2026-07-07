from .ml.model import Sam3GroundingModel, Sam3ImageModel, Sam3VideoModel


def build_image_model(config: dict) -> Sam3ImageModel:
    path = config.get("path")
    device = config.get("device", "cuda")
    model = Sam3ImageModel(path=path).to(device)
    return model


def build_grounding_model(config: dict) -> Sam3GroundingModel:
    path = config.get("path")
    visual_path = config.get("visual_path")
    device = config.get("device", "cuda")
    model = Sam3GroundingModel(
        path=path,
        visual_path=visual_path,
    ).to(device)
    return model


def build_video_model(config: dict) -> Sam3VideoModel:
    path = config.get("path")
    device = config.get("device", "cuda")
    model = Sam3VideoModel(path=path).to(device)
    return model
