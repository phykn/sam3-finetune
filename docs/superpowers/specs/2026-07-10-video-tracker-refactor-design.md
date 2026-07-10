# SAM3.1 Video Tracker Refactor Design

## Goal

Refactor the SAM3.1 video tracker into a direct forward-inference pipeline that
is easier to follow, while preserving official checkpoint loading, multi-object
tracking, mid-stream mask additions, explicit object removal, and exact output
parity with the current implementation.

Automatic object discovery remains a separate future feature. It belongs to a
detector-tracker integration layer and is not part of this tracker refactor.

## Current Baseline

The repository starts this phase with:

- 155 passing tests
- exact image, grounding, and video parity against `sam3-main`
- strict local loading of official SAM3.1 weights
- a one-way core model structure
- a video tracker implementation spread across 57 files and about 9,000 lines
- 131 top-level functions that receive `self` and are attached to classes as
  methods
- a tracker constructor with training, evaluation, compilation, correction,
  and inference options mixed into one interface

The current public streaming path is:

1. cache a frame
2. create inference state
3. add one or more mask prompts
4. consolidate prompt outputs
5. propagate one frame forward
6. return masks, logits, object ids, and presence scores

The refactor must preserve this behavior before removing any inactive branch.

## Scope

### Preserved

- `Sam3VideoModel` construction and local official SAM3.1 checkpoint loading
- current checkpoint parameter names and strict video block loading
- `VideoPredictor.start()` and `VideoPredictor.predict()` behavior
- one or more initial mask prompts
- adding new object ids with masks after tracking has started
- replacing the mask for an existing object id on a cached frame
- removing one or more object ids from an active state
- multiplexed multi-object inference
- forward temporal memory and object-presence scoring
- output mask resizing to the original frame resolution
- CPU construction and CUDA inference

### Added Public Convenience Methods

`VideoPredictor` gains two thin methods over the preserved model operations:

```python
predictor.add_masks(state, masks, obj_ids, frame_idx=None)
predictor.remove_objects(state, obj_ids, strict=True)
```

`frame_idx=None` means the latest cached frame, which is
`state["next_frame"] - 1`. An explicit frame must already exist in
`state["state"]["cached_features"]`.

Both methods mutate the supplied session state and return the current ordered
object-id list. Prompt masks are already known to the caller, so the predictor
does not invent incomplete prediction outputs without tracker logits or
presence scores.

`Sam3VideoModel` exposes the corresponding model-level operations:

```python
model.add_masks(state, frame_idx, obj_ids, masks)
model.remove_objects(state, obj_ids, strict=True)
```

The existing `add_new_masks` name may remain as a temporary internal alias only
while call sites move. The final public code uses `add_masks`.

### Removed

- video model training entry points and batched training datapoint handling
- backward propagation and reverse-frame branches
- text-prompt detection and automatic object discovery
- distributed and multi-GPU evaluation paths
- compilation-only branches
- point-prompt training and iterative correction sampling
- public video point refinement
- dataset-specific dynamic VOS training transitions
- unused static-tracker and non-multiplex demo variants
- method injection such as `forward_image = forward_image`
- constructor options that are fixed for the supported SAM3.1 inference model

The internal empty point-input stores may remain temporarily if active
checkpoint-compatible inference helpers require their keys. No point APIs or
point-processing branches remain reachable from the public model.

## Compatibility Boundary

This is a structural refactor, not a new video model.

The following must not change:

- registered module and parameter names used by the official video checkpoint
- tensor shapes entering the vision backbone, memory encoder, tracker
  transformer, and mask decoder
- multiplex slot assignment for the same ordered object ids
- memory frame selection for the same state
- mask interpolation mode and threshold
- presence score meaning
- ordering of returned object ids
- `VideoPredictor` output dictionary keys and NumPy dtypes

Old LoRA checkpoint compatibility is not required. Hugging Face loading and
runtime VLM or text-encoder loading remain prohibited.

## Architecture

The dependency direction stays:

```text
components -> blocks -> model -> predict
```

Weight-bearing tensor operations remain below the model layer. Stateful video
orchestration is owned by the video model instead of pretending to be a neural
component.

The target responsibilities are:

```text
src/ml/components/video/
  tensor modules, multiplex math, memory math, SAM heads

src/ml/blocks/video_*.py
  assembled feature, memory, and tracking blocks

src/ml/model/video/
  model.py       model assembly and checkpoint boundary
  runtime.py     fixed SAM3.1 inference runtime
  state.py       state creation and object registry
  masks.py       add or replace mask prompts
  objects.py     remove and reindex objects
  propagate.py   forward-only frame propagation

src/predict/video.py
  public predictor API

src/predict/video_ops/session.py
  image conversion and predictor session adaptation
```

The first implementation may leave low-level tensor helpers in their current
`components/video/tracker/` subpackages when moving them would only create path
churn. Stateful public orchestration and method injection must not remain there.

## Runtime Construction

The supported runtime has one explicit constructor. It receives only the
objects that vary at construction time:

```python
VideoRuntime(
    backbone,
    transformer,
    maskmem_backbone,
    multiplex_controller,
)
```

SAM3.1 inference constants such as image size, memory count, multiplex
capacity, pointer behavior, and mask-decoder settings are defined once near
the runtime construction. Training probabilities and unused evaluation knobs
are removed rather than retained at fixed zero values in the public
constructor.

Registered modules keep the same attribute names. This preserves strict
checkpoint loading even though the Python class and control-flow layout become
simpler.

## Data Flow

### Start

```text
image
  -> preprocess and cache image features
  -> create state
  -> register ordered object ids
  -> resize masks to model and video resolutions
  -> create multiplex state
  -> run the initial mask frame
  -> consolidate conditioning memory
```

### Forward Frame

```text
cached frame features
  -> select prior conditioning and non-conditioning memory
  -> run tracker transformer
  -> decode object masks and presence logits
  -> encode new memory
  -> write frame output once
  -> resize masks to video resolution
  -> return ordered ids, masks, logits, and scores
```

Only increasing frame indices are supported by the public predictor.

### Add Objects

```text
cached frame + ordered new ids + masks
  -> reject duplicate ids in the same request
  -> append truly new ids to the registry
  -> retain indices for existing ids being reconditioned
  -> allocate multiplex slots
  -> suppress overlap against other supplied masks
  -> rebuild that frame's consolidated output
  -> use the result as future conditioning memory
```

Adding an object is explicit. The tracker never interprets a low presence
score as permission to create a new object id.

### Remove Objects

```text
requested ids
  -> resolve ids and strictness
  -> remove multiplex assignments
  -> slice stored mask, score, pointer, and memory tensors
  -> rebuild contiguous public object indices
  -> preserve the relative order of remaining ids
  -> return affected frame outputs
```

An object disappearing from the image does not automatically delete its state.
Presence logits describe visibility; `remove_objects` is an explicit request to
stop tracking selected ids.

## Mathematical Invariants

### Multiplex Mapping

- Every active object maps to exactly one valid multiplex slot.
- Public object ids remain unique and ordered.
- Demultiplexing a multiplexed tensor recovers the original active object rows.
- Removing an object does not change the values or relative order of remaining
  rows.

### Mask Coordinates

- Public masks use `H x W` or `N x H x W` layout.
- Model input masks use `N x 1 x input_mask_size x input_mask_size`.
- Video masks use `N x 1 x video_height x video_width`.
- Bilinear interpolation uses `align_corners=False`.
- Prompt masks are converted to boolean with the existing `> 0.5` threshold.
- Predicted video masks use the existing logit threshold `> 0.0`.

### Temporal Memory

- Forward propagation never reads a frame later than the frame being decoded.
- Conditioning-frame selection remains deterministic for equal input state.
- The first conditioning frame remains available when the configured memory
  limit requires it.
- A removed object contributes no masks, pointers, scores, or memory rows.

### Presence

- Presence values remain logits inside the model.
- A non-positive presence logit may suppress a mask but does not remove the
  object id.
- Output ordering matches the state object-id ordering.

## State Rules

The active state remains a plain dictionary to avoid a compatibility wrapper,
but all mutation is owned by `state.py`, `masks.py`, `objects.py`, or
`propagate.py`.

Required validation:

- frame count and spatial sizes are positive
- cached frame indices are in range
- mask count equals object-id count
- mask rank is exactly three after public normalization
- ids in one add request are unique
- explicit add frames are cached
- strict removal rejects unknown ids
- removing all objects leaves a valid empty state that cannot propagate until
  a new mask is added
- reverse propagation requests fail at the public boundary

Assertions are reserved for internal tensor invariants. Invalid user input
raises `ValueError`, `KeyError`, or `RuntimeError` with a direct message.

## Testing Strategy

### Contract Tests

- existing `VideoPredictor.start()` and `predict()` tests remain green
- predictor add and remove methods call the real state operations
- unknown removal and uncached add frames raise the documented errors
- the public predictor does not expose reverse propagation

### State and Math Tests

- ordered id registration and duplicate rejection
- multiplex then demultiplex identity
- removal preserves remaining rows and id order
- mask layout, resize, and threshold invariants
- forward-only processing order
- conditioning-memory selection never reads future frames
- presence logits do not implicitly delete ids

### Architecture Tests

- model code does not import video training datapoints
- video runtime has no top-level `self` functions bound onto classes
- removed training, reverse, correction-sampling, and compile modules are not
  imported by the supported runtime
- predictor depends on the model boundary, not video components

### Checkpoint Tests

- official video keys map to the same registered parameter names
- strict video block loading has no missing or unexpected keys
- CPU construction does not allocate CUDA tensors

### Runtime Parity

Run the existing three-frame CUDA parity script and compare:

- object ids
- low-resolution logits
- original-resolution mask logits
- presence scores
- thresholded-mask XOR

Add a dynamic parity script that drives both implementations through:

1. multiple initial object masks
2. one forward frame
3. one new object mask on a cached frame
4. another forward frame
5. removal of one object
6. comparison of remaining ids and tensors

The upstream implementation writes empty per-object rows for objects that did
not exist on a historical frame, which makes its removal path fail in this
sequence. The parity script applies the same local-index correction to the
upstream instance in memory. It does not modify `sam3-main`.

All comparable floating outputs must have maximum and mean absolute error zero,
and thresholded masks must have XOR zero.

## Implementation Order

1. Add public API, architecture, state, multiplex, and parity tests against the
   desired interface.
2. Replace the option-heavy training/runtime class hierarchy with one fixed
   inference runtime while preserving registered attribute names.
3. Move state creation and mutation behind explicit model-owned functions.
4. Replace injected methods with direct methods or explicit function calls.
5. Reduce propagation to the forward-only path.
6. Preserve and expose mask-based object addition and explicit removal.
7. Delete unreachable video training, reverse, point-correction, compile, and
   evaluation-only modules after import tests prove they are disconnected.
8. Run focused tests, the full suite, formatting checks, strict checkpoint
   loading, standard video parity, and dynamic add/remove parity.

## Out of Scope

- automatic object detection
- detector-track association
- text prompts in video
- runtime text encoder or VLM loading
- Hugging Face loading
- old LoRA checkpoint compatibility
- video training
- backward tracking
- point-click correction inside the video tracker

Automatic discovery can later consume `GroundPredictor` results and call the
preserved mask-addition API without changing this runtime boundary.

## Success Criteria

The phase is complete when:

- the supported video flow is visibly forward-only
- the public mask add and object remove APIs work after tracking starts
- object disappearance remains represented by presence scores without deleting
  state
- method injection and the option-heavy tracker constructor are gone
- inactive training, reverse, and correction paths are removed
- official SAM3.1 video weights still load strictly from a local path
- the full test suite passes
- changed Python files pass Black and Ruff
- standard and dynamic CUDA parity are exactly zero
- `sam3-main/`, `weight/`, and `asset/` remain unmodified and unstaged
