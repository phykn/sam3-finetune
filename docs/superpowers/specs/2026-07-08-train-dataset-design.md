# Train Dataset Design

## Goal

Build an image prompt-segmentation training dataset from `sam3.sample.v1` JSON files.
The dataset is for the image/single path only. Grounding and video stay on the
base model and are out of scope.

## Scope

- Read existing sample JSON files through `src.data.sample.load`.
- Use one training item per object.
- Return image, prompt, target mask, and object-existence target.
- Generate point, box, or mask prompts from the ground-truth ROI/mask.
- Keep image sizes in original coordinates at this dataset layer.
- Put prompt augmentation code under `src/data/augment/prompt`.

Out of scope:

- Grounding dataset.
- Video/tracking dataset.
- Training loop and loss implementation.
- Multi-point correction training.
- Image augmentation implementation. The namespace should leave room for it
  under `src/data/augment`.

## Components

### BaseDataset

`BaseDataset(paths)` stores a list of JSON paths and loads a `Sample` by index.
It does not scan directories. Train/valid split code owns the path list.

### TrainDataset

`TrainDataset(paths, config)` extends `BaseDataset` and exposes object-level
items. Each item selects one object from one sample and builds the configured
prompt type.

`config` starts with these fields:

```python
{
    "prompt": "point" | "box" | "mask",
    "bg_prob": 0.2,
    "box_jitter": 0.1,
    "mask_ops": ("none", "shift", "erode", "dilate", "blur", "resize"),
}
```

The returned item has this shape:

```python
{
    "image": image_array,
    "prompt": {
        "type": "point" | "box" | "mask",
        "points": point_array_or_none,
        "point_labels": label_array_or_none,
        "box": box_array_or_none,
        "mask": mask_array_or_none,
    },
    "target": target_mask,
    "has_object": True | False,
}
```

Only the active prompt field is populated. Inactive prompt fields are `None`.

### Prompt Augmentation

Prompt generation and prompt-specific randomization live under
`src/data/augment/prompt`. This keeps `dataset.py` focused on indexing,
loading, and item assembly. Image augmentation can later live under
`src/data/augment` without mixing image and prompt transforms in the dataset
class.

`target` is always the training target mask. It is exact for object clicks and
empty for background point clicks.

## Prompt Generation

### Point Prompt

Point training uses one positive point only.

- `point_labels` is always `[1]`.
- Object click: sample one random foreground pixel from the object's target
  mask. `target` is the object mask and `has_object` is `True`.
- Background click: sample one random point outside the union of all object
  masks in the image. `target` is an empty mask and `has_object` is `False`.
- Background points may be adjacent to object boundaries.
- Background click probability is configurable, default `0.2`.
- Negative points and multiple points are out of scope for v1.

### Box Prompt

Box training uses a box made from the target mask.

- Compute a tight box from the target mask.
- Apply random jitter to each edge using a percentage of the box width/height.
- Default `box_jitter = 0.1`.
- Clip to image bounds.
- If jitter makes the box invalid or too small, fall back to the tight box.
- Background box cases are not generated.

### Mask Prompt

Mask training uses a prompt mask derived from the exact target mask.

- `target` remains the exact binary mask.
- `prompt["mask"]` is a float mask.
- Pick exactly one operation uniformly at random:
  - `none`: exact GT mask
  - `shift`
  - `erode`
  - `dilate`
  - `blur`
  - `resize`: down/up coarse mask
- Operations are not stacked.
- The implementation should stay simple and fast because this runs in dataset
  loading.

## Data Flow

1. Load a sample JSON.
2. Pick the object assigned to the dataset item.
3. Restore the object's full mask from `roi` and `box`.
4. Build the selected prompt type.
5. Return original image array, prompt, exact target mask, and `has_object`.

The dataset does not normalize or resize images. Tensor conversion and model
input scaling remain a later train-loop or collate responsibility.

## Error Handling

- If a sample has no objects, it contributes no object-level training items.
- If an object mask is empty, skip it when building the object index.
- If background point sampling has no available background pixels, fall back to
  an object click.
- If a generated box is invalid, fall back to the tight box.

## Testing

Use narrow checks first:

- `BaseDataset` loads a JSON path and returns a `Sample`.
- `TrainDataset` length equals the number of non-empty objects.
- Point prompt returns one point with label `1`.
- Background point items return an empty target and `has_object == False`.
- Box prompt is clipped and valid.
- Mask prompt applies only one operation and keeps target exact.

After implementation, run the focused dataset tests, then `python -m pytest tests`
if the focused tests pass.
