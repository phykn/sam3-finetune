# Grounding Reference Refactor Design

**Status:** Implemented
**Date:** 2026-07-10
**Branch:** `codex/grounding-refactor`

## Goal

Refactor grounding into a box-reference pipeline that extracts class-specific
features from one or more reference images and predicts matching masks in a new
image. Improve latency and peak VRAM while preserving the original SAM3.1 box
prompt and decoder math.

The common case is one reference image. Multiple reference images use the same
API.

## Confirmed Model Boundary

The SAM3.1 grounding geometry encoder has trained point and box prompt paths. Its
standard builder does not create a mask encoder, and the official checkpoint has
no geometry mask-encoder weights.

Reference masks therefore do not belong in the public grounding API. If a caller
has masks, a separate helper may convert them to boxes later. That conversion is
not part of this refactor.

The reference and target flow is:

```text
reference image + reference boxes + integer class IDs
  -> reference image features
  -> one feature vector per box
  -> one geometry prompt per reference-image/class group

target image
  -> target image features
  -> decode each reference-class prompt while reusing target features
  -> predicted boxes and masks
  -> class-specific feature similarity filtering
  -> object dictionaries
```

Reference box coordinates are never copied to the target image. They only select
objects in the reference image when the reusable prompt features are created.

## Public API

Reference encoding accepts pixel-space `xyxy` boxes using the same half-open box
meaning as `src.data.sample.Object.box`:

```python
reference = predictor.encode_reference(
    image,
    boxes,       # NumPy-compatible [N, 4], pixel xyxy
    class_ids,   # NumPy-compatible [N], non-negative integers
)
```

Prediction always consumes a list. One reference image is a one-element list:

```python
objects = predictor.predict(target_image, [reference])
```

Multiple reference images use the same call:

```python
objects = predictor.predict(target_image, [reference_a, reference_b])
```

The old `encode_ref()` API, point/mask reference options, reference names, and
reference-name result dictionary are removed without compatibility aliases.

## Reference Classes and Prompts

Every reference box keeps its own feature vector and integer class ID. Boxes with
the same class ID in one reference image form one box prompt. The prompt contains
all boxes in that class; boxes are not unioned or averaged.

For multiple reference images, one prompt group exists for each
`(reference image, class_id)` pair. Feature vectors with the same class ID are
combined into one class feature bank across all reference images.

The encoded reference artifact is a private plain dictionary containing:

- encoded prompt features and prompt masks;
- one class ID per prompt group;
- one feature vector per original box; and
- one class ID per feature vector.

It does not retain the full reference image embedding.

## Feature and Similarity Math

Let `F` be the final reference FPN feature map. A reference box is scaled to that
feature grid and represented as a rectangular indicator `b_i`. Its normalized
masked mean is:

```math
v_i = \operatorname{normalize}
\left(
\frac{\sum_{x,y} F(x,y)b_i(x,y)}
{\max(\sum_{x,y}b_i(x,y), \epsilon)}
\right)
```

The target candidate feature `u_j` uses the predicted target mask rather than the
predicted box. Similarity between candidate `j` and class `c` is the maximum cosine
similarity over every reference box with that class ID:

```math
s_{c,j} = \max_{i:\ class_i=c} v_i^\top u_j
```

Reference features are never averaged into a prototype.

The SAM3.1 grounding score and feature similarity have separate roles:

1. reject model scores below `score_thr`;
2. reject class similarities below `sim_thr`;
3. merge prompt-group results for the same class;
4. apply box NMS within that class using model confidence; and
5. apply `top_k` per class when configured.

Classes are independent. No NMS runs across different classes, so the same target
object may appear with several class IDs.

## Execution and Memory

Reference vision encoding runs once per reference image. Target vision encoding
runs once per `predict()` call.

Reference box prompts are encoded as a padded class batch. Target prompt groups
are decoded one at a time with batch size 1. The target vision backbone is never
rerun per prompt.

For `G` prompt groups, the target decoder call count is `G`.

Target prompt batching was implemented and measured, but rejected for the public
predictor because the BF16 transformer path is shape-dependent. Identical target
inputs produced an encoder maximum difference of `0.25` between batch sizes 1 and
4, which changed the final object count from 12 to 13. Float32 reduced the maximum
difference to `1.57356e-5`, but did not provide exact equality. Sequential target
decoding preserves exact single-prompt behavior and keeps the execution direction
simple.

Score filtering, target mask feature extraction, and similarity filtering stay on
the GPU. Only retained objects are converted to CPU NumPy values.

The predictor never returns the decoder `raw` dictionary or other GPU tensors.
Small encoded reference prompts and box features intentionally stay on the model
device for reuse across target images.

## Prediction Result and Existing JSON

`predict()` returns a list of plain object dictionaries:

```python
{
    "object_id": 1,
    "class_id": 2,
    "box": (x0, y0, x1, y1),
    "mask": np.ndarray,   # bool [H, W]
    "logit": np.ndarray,  # float32 low-resolution refinement input
    "metrics": {
        "score": 0.93,
        "similarity": 0.88,
    },
}
```

The final `box` is the integer bounding box of the returned mask, matching the
existing JSON object's compact `box + roi` representation. The model-predicted box
is used internally for NMS and is not returned separately.

This is an in-memory NumPy form of the existing `sam3.sample.v1` model, not a new
persisted schema:

- `mask` becomes `Object.box` plus `Object.roi` through `pack.box_roi()`;
- `class_id` becomes `Object.class_id`;
- scores stay in `Object.metrics`;
- the target image stays in `sample.Image`; and
- low-resolution `logit` is available for refinement but is not written to JSON.

## Code Boundaries

```text
src/data/ground.py
  pixel xyxy boxes -> padded normalized prompt tensors

src/predict/ground_ops/reference.py
  validate boxes/classes, group boxes, merge prompts and feature banks

src/predict/ground_ops/sim.py
  reference box features, target mask features, maximum cosine similarity

src/predict/ground_ops/output.py
  score/similarity filtering, within-class NMS, object dictionaries

src/predict/ground.py
  encode_reference() and predict() orchestration

src/ml/model/grounding.py
  public encode/decode boundary and reference prompt batch alignment

src/ml/blocks/grounding/*
  checkpoint-compatible box prompt and decoder math
```

No dataclass, result wrapper, compatibility adapter, lazy import, or new option
layer is added.

`scripts/ground.py` reads reference `Sample/Object` data directly:

1. load one or more reference samples;
2. read each object's `box` and integer `class_id`;
3. encode references;
4. predict a target image;
5. optionally refine returned mask logits; and
6. save through the existing `Sample/Object` JSON path.

## Validation and Errors

Reference input fails with `ValueError` when:

- boxes are not `[N, 4]` after accepting one `[4]` box as `N=1`;
- there are no boxes;
- a box contains non-finite coordinates;
- a box is empty or lies outside the reference image after clipping;
- `class_ids` is not a length-`N` integer sequence; or
- any class ID is negative.

Boxes are clipped to reference image bounds. Reversed coordinates are invalid and
are not silently reordered.

Constructor thresholds are validated once. `predict()` requires a non-empty list
of encoded references.

## Compatibility Boundary

The compatibility target is:

- SAM3.1 checkpoint keys and strict load;
- cached visual-token load;
- box prompt encoder math; and
- raw logits, boxes, and masks for a single reference box.

The final public object list is intentionally not compatible with the old
reference-name dictionary because class grouping, within-class NMS, object output,
and `raw` removal are approved behavior changes.

## Verification

### Tests

- Box and class validation.
- Padded class box prompt construction.
- Box-region masked mean and L2 normalization.
- Maximum exemplar similarity per class.
- Repeated classes across reference images.
- One vision pass per reference image and one per target image.
- One target decoder call per prompt group while reusing one target image encoding.
- Same-class NMS and cross-class overlap preservation.
- CPU NumPy object output without `raw` or GPU tensors.
- Object dictionary round-trip through existing `Sample/Object` JSON.

### Real checkpoint

- Strictly load `weight/sam3.1_multiplex.pt` and cached visual tokens.
- Compare the pre-refactor and refactored single-box prompt logits, boxes, and masks
  within the existing parity tolerance.
- Run multi-box, multi-class reference prediction on CUDA.

### Performance

Measure repeated sequential prompt-group prediction on the same loaded model and
encoded references:

- reference encoding time;
- target encoding and decode time;
- end-to-end latency;
- peak allocated CUDA memory; and
- repeated output equality.

Performance is reported as measured evidence rather than a hardware-independent
percentage promise. Structural tests guarantee one backbone pass per image,
batch-1 target decoding, and no retained raw result tensors.

## Completion Criteria

- Public grounding reference input is `boxes + class_ids` only.
- One and multiple reference images use the same list API.
- Each reference box retains an individual feature vector.
- Same-image/class boxes form one trained box prompt.
- Target image encoding occurs once and prompt groups decode sequentially.
- Same-class NMS removes duplicates; cross-class overlaps remain.
- Results are CPU NumPy object dictionaries compatible with existing JSON.
- Prediction results retain no `raw` GPU tensors.
- SAM3.1 strict load and single-box raw decoder parity pass.
- CUDA latency and peak VRAM measurements are recorded.
- Full pytest, changed-file Black, repository Ruff, diff, and protected-path checks pass.
