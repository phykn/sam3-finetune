from .data.dataloader import InfiniteLoader, make_infinite_train_loader
from .finetune.model import FinetuneModel
from .ml.model import Sam3GroundingModel, Sam3ImageModel, Sam3VideoModel


def build_image_model(config: dict) -> Sam3ImageModel:
    path = config.get("path")
    device = config.get("device", "cuda")
    model = Sam3ImageModel(path=path).to(device)
    return model


def build_finetune_model(config: dict) -> FinetuneModel:
    path = config.get("path")
    device = config.get("device", "cuda")
    model = FinetuneModel(
        Sam3ImageModel(path=path),
        num_conditions=config.get("num_conditions", 1),
        num_experts=config.get("num_experts", 4),
        num_labels=config.get("num_labels", 1),
        lora_rank=config.get("lora_rank", 8),
        feature_rank=config.get("feature_rank", 16),
    ).to(device)
    return model


def build_finetune_loader(config: dict) -> InfiniteLoader:
    return make_infinite_train_loader(
        paths=config["paths"],
        batch_size=config["batch_size"],
        conds=config.get("conds"),
        labels=config.get("labels"),
        num_workers=config.get("num_workers", 4),
    )


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
