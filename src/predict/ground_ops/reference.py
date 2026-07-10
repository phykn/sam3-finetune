import numpy as np
import torch


def validate(boxes, class_ids, image_shape):
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        raise ValueError("reference boxes are empty")
    if boxes.ndim == 1 and boxes.size == 4:
        boxes = boxes[None]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("reference boxes must have shape Nx4")
    boxes = boxes.astype(np.float32, copy=False)
    if not np.isfinite(boxes).all():
        raise ValueError("reference boxes must be finite")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("reference boxes must have positive area")

    classes = np.asarray(class_ids)
    if classes.ndim != 1 or len(classes) != len(boxes):
        raise ValueError("class_ids length must match boxes")
    if not np.issubdtype(classes.dtype, np.integer):
        raise ValueError("class_ids must be integers")
    classes = classes.astype(np.int64, copy=False)
    if np.any(classes < 0):
        raise ValueError("class_ids must be non-negative")

    height, width = image_shape
    boxes = boxes.copy()
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("reference box is outside the image")
    return boxes, classes


def groups(boxes, class_ids):
    classes = np.unique(class_ids)
    return classes, [boxes[class_ids == value] for value in classes]


def feature_bank(references):
    classes = np.concatenate([item["feature_classes"] for item in references])
    features = torch.cat([item["features"] for item in references])
    return {
        int(value): features[torch.as_tensor(classes == value, device=features.device)]
        for value in np.unique(classes)
    }


def prompt_groups(references):
    features = torch.cat(
        [item["prompt"]["features"] for item in references],
        dim=1,
    )
    mask = torch.cat([item["prompt"]["mask"] for item in references], dim=0)
    classes = np.concatenate([item["prompt_classes"] for item in references])
    return {"features": features, "mask": mask}, classes
