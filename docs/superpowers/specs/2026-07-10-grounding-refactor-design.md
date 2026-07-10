# Grounding Reference Refactor Design

**Status:** Approved for implementation planning  
**Date:** 2026-07-10  
**Branch:** `codex/grounding-refactor`

## 1. Goal

Refactor reference-based grounding into a direct mask-only pipeline that:

- extracts class-specific features from one or more reference images;
- finds every matching object in a new image for each reference class;
- improves latency and peak VRAM without changing SAM3.1 checkpoint math;
- returns object-oriented NumPy results that map directly to the existing
  `Sample`/`Object` JSON schema; and
- keeps the dependency direction simple:

```text
data -> grounding components -> grounding blocks -> model -> predictor -> scripts
```

The common case is one reference image. Multiple reference images use the same
API and remain supported.

## 2. Current Problems

The current grounding path has several mismatched assumptions:

- `GroundPredictor.encode_ref()` accepts points, boxes, or masks even though the
  target workflow uses reference masks only.
- A reference name acts as an implicit class key. Duplicate names overwrite each
  other in the result dictionary.
- A three-dimensional mask array is interpreted as several masks in one prompt,
  but the real geometry encoder asserts that a prompt contains exactly one mask.
  The existing multiple-mask unit test uses a fake model and does not cover this
  real checkpoint path.
- Every reference runs the full decoder separately.
- The result keeps the decoder `raw` dictionary and therefore retains large GPU
  tensors after NumPy results have already been created.
- Results are stored as arrays grouped by reference name instead of objects with
  an explicit integer `class_id`.
- `scripts/ground.py` owns reference preparation, prediction, refinement, JSON
  conversion, and visualization in one large file.

## 3. Scope

### In scope

- Mask-only public reference input.
- One or more reference images.
- Any practical number of masks per reference image.
- Non-negative integer class IDs.
- Individual feature vectors for every reference mask.
- One union grounding prompt for each `(reference image, class_id)` group.
- Batched and chunked target decoding.
- Independent results per class.
- Object-oriented NumPy output and existing JSON conversion.
- Real SAM3.1 checkpoint parity and CUDA performance measurement.
- Focused grounding file and function naming cleanup.

### Out of scope

- Text/VLM loading or Hugging Face integration.
- Automatic insertion of grounding results into the video tracker.
- Cross-class suppression or forced single-class assignment.
- Learned prototype aggregation.
- Training or changing SAM3.1 weights.
- Historical LoRA checkpoint compatibility.

## 4. Public API

Reference encoding accepts an image, full-size masks, and one class ID per mask:

```python
reference = predictor.encode_reference(
    image,
    masks,       # NumPy-compatible [N, H, W]
    class_ids,   # NumPy-compatible [N]
)
```

Prediction always receives a list. A single reference image is represented by a
one-element list, so there is no separate single-reference branch:

```python
objects = predictor.predict(target_image, [reference])
```

Multiple reference images use the same call:

```python
objects = predictor.predict(target_image, [reference_a, reference_b])
```

The old `encode_ref()` point/box/mask API and reference-name result dictionary are
removed without compatibility aliases.

## 5. Reference Representation

`encode_reference()` returns a plain dictionary containing only reusable encoded
state. It does not keep the full reference image embedding.

Conceptually it contains:

```text
groups:
  one encoded prompt for each class present in this reference image
features:
  one normalized vector for every input mask, grouped by class_id
```

The exact internal keys are private to `GroundPredictor`; scripts and callers only
pass the result back to `predict()`.

For a reference image `r` and class `c`, all class masks form one union prompt:

```math
U_{r,c}(x,y) = \max_{i \in (r,c)} m_i(x,y)
```

Each original mask remains separate for similarity measurement. Union is used only
for the geometry prompt.

## 6. Feature and Similarity Math

Let `F` be the last reference FPN feature map and `m_i` a resized soft mask. The
feature of reference mask `i` is the normalized masked mean:

```math
v_i = \operatorname{normalize}
\left(
\frac{\sum_{x,y} F(x,y)m_i(x,y)}
{\max(\sum_{x,y}m_i(x,y), \epsilon)}
\right)
```

The target candidate feature `u_j` uses the same equation. Similarity between
candidate `j` and class `c` is the maximum cosine similarity over all reference
masks with that class ID, including masks from different reference images:

```math
s_{c,j} = \max_{i:\ class_i=c} v_i^\top u_j
```

All reference vectors are retained. They are not averaged into a prototype.

The SAM3.1 grounding logit remains the model confidence. Similarity is an
additional class-specific gate:

1. reject model scores below `score_thr`;
2. reject class similarities below `sim_thr`;
3. merge prompt-group results for the same class;
4. apply box NMS within that class using model confidence;
5. apply `top_k` per class when configured.

No NMS is applied across different classes. The same physical object may therefore
appear more than once with different class IDs.

## 7. Batched Decode and Memory

Reference image encoding runs once per reference image. Target image encoding runs
once per `predict()` call.

Each `(reference image, class_id)` union produces one prompt group. Prompt groups
are decoded in chunks controlled by a positive `prompt_batch_size`, defaulting to
4. A chunk is stacked along the model batch dimension. Target image features are
expanded or indexed for that batch; the vision backbone is never rerun per prompt.

For `G` prompt groups, decoder call count is:

```text
ceil(G / prompt_batch_size)
```

Peak temporary decode memory is bounded by the chunk size instead of total prompt
count. Chunk tensors are released after their candidates have been converted to
the compact intermediate form.

The predictor never returns the decoder `raw` dictionary or other GPU tensors.
Only the small encoded reference artifact intentionally remains on the model
device for reuse across target images.

## 8. Prediction Result

`predict()` returns one plain dictionary per detected object:

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

All arrays are CPU NumPy arrays. `object_id` is assigned after final filtering and
is sequential within one prediction.

This is an in-memory NumPy form of the existing data model, not a new persisted
schema. Saving uses the existing `sam3.sample.v1` representation:

- `mask` becomes the existing `box` plus compact `roi` through `pack.box_roi()`;
- `class_id` becomes `Object.class_id`;
- `score` and `similarity` remain in `Object.metrics`;
- the target image becomes the existing `sample.Image`; and
- low-resolution `logit` is used for optional refinement but is not written to
  JSON.

## 9. Code Boundaries

The grounding predictor keeps one-way orchestration and small helpers:

```text
src/data/ground.py
  NumPy-to-tensor conversion for grounding prompts

src/predict/ground_ops/reference.py
  reference validation, class grouping, union masks, feature bank assembly

src/predict/ground_ops/sim.py
  masked mean vectors and class-wise maximum cosine similarity

src/predict/ground_ops/output.py
  candidate formatting, filtering, within-class NMS, object dictionaries

src/predict/ground.py
  encode_reference() and predict() orchestration

src/ml/model/grounding.py
  public encode/decode model boundary and batch alignment

src/ml/blocks/grounding/*
  checkpoint-compatible grounding prompt and decoder math
```

No dataclass, result wrapper, compatibility adapter, lazy import, or new option
layer is added.

`scripts/ground.py` becomes a short example that:

1. loads one or more existing `Sample` JSON references;
2. converts each reference object's `roi` to full NumPy masks;
3. uses `Object.class_id` as the reference class ID;
4. predicts a target image;
5. optionally refines the returned low-resolution logits; and
6. saves the result through the existing `Sample`/`Object` JSON path.

## 10. Validation and Errors

Reference input fails with `ValueError` when:

- masks are not `[N, H, W]` after accepting a single `[H, W]` mask as `N=1`;
- `class_ids` is not a length-`N` integer sequence;
- any class ID is negative;
- any mask is empty;
- a mask spatial shape differs from its reference image; or
- there are no masks.

Constructor thresholds and `prompt_batch_size` are validated once. The predictor
does not silently clamp invalid reference data or invent class IDs.

The compatibility target is the SAM3.1 decoder math, checkpoint keys, and raw
logits/boxes/masks for a single mask prompt. The final public object list is
intentionally not byte-for-byte compatible: the approved class grouping,
within-class NMS, object-oriented return type, and removal of `raw` change
post-processing behavior.

## 11. Verification

### Unit and integration tests

- Masked mean and L2 normalization match explicit tensor math.
- Class similarity equals the maximum exemplar cosine similarity.
- Union masks are correct for repeated classes.
- Repeated class IDs across reference images share one feature bank.
- Many masks in one class produce one prompt group for that image.
- A single reference and multiple references use the same list API.
- The reference and target backbones run exactly once per image.
- Decoder call count is `ceil(groups / prompt_batch_size)`.
- Same-class duplicates are removed; cross-class overlaps remain.
- Returned objects contain CPU NumPy values and no `raw` key or GPU tensor.
- Object dictionaries round-trip through the existing `Sample`/`Object` JSON.

### Real checkpoint checks

- Strictly load `weight/sam3.1_multiplex.pt` and the cached visual tokens.
- For one reference image, one class, and one mask, compare pre-refactor and
  refactored grounding logits, boxes, and masks within the existing parity
  tolerance.
- Run the upstream grounding parity script after adapting its public call site.
- Run an actual multi-mask, multi-class reference on CUDA.

### Performance evidence

Measure on the local CUDA device:

- reference encoding time;
- target encoding time;
- prompt decode time;
- end-to-end latency; and
- peak allocated CUDA memory.

Compare chunked decoding with an equivalent sequential prompt-group execution.
Performance is reported as measured evidence rather than a hardware-independent
percentage promise. Structural assertions guarantee one backbone pass per image,
bounded chunk memory, and no retained raw result tensors.

### Repository checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe -m black --check <changed-python-files>
.\.venv\Scripts\python.exe -m ruff check src tests scripts
git diff --check
```

The protected `sam3-main/`, `weight/`, and `asset/` paths are read-only verification
inputs and are never staged or committed.

## 12. Completion Criteria

- Mask-only integer-class reference API is implemented without compatibility
  aliases.
- One and multiple reference images work through the same list API.
- Many masks remain individual class features while prompt cost scales with
  reference-image/class groups.
- Target image encoding occurs once and prompt groups decode in chunks.
- Results are object dictionaries compatible with the existing JSON data model.
- Class results are independent and overlap across classes is preserved.
- No prediction result retains `raw` GPU tensors.
- Single-mask SAM3.1 grounding math passes real checkpoint parity.
- Full tests, changed-file Black, repository Ruff, and diff checks pass.
- CUDA latency and peak VRAM measurements are recorded.
