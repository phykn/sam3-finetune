# SAM3 Core Refactor Design

## Goal

Refactor the shared math and image/grounding model core into a simple,
one-way architecture with consistent names and explicit mathematical
invariants. The refactor may break the current Python API, module paths, and
locally produced LoRA checkpoint keys, but it must keep official SAM3.1 base
checkpoint loading and exact image/grounding inference parity.

## Current Baseline

- The repository contains about 27,000 lines across 202 Python files.
- `python -m pytest tests -q` passes all 140 tests.
- `scripts/parity_image.py` reports zero max and mean differences and zero mask
  XOR pixels against the upstream implementation on CUDA.
- `scripts/parity_ground.py` reports zero max and mean differences and zero
  mask XOR pixels against the upstream implementation on CUDA.
- `scripts/parity_video.py` reports zero differences and zero mask XOR pixels
  for both checked propagation frames on CUDA.
- Ruff currently reports one unused re-export in `src/finetune/__init__.py`.
- Black currently reports 27 files that would be reformatted. The core phase
  will format its changed files; unrelated video files remain untouched until
  the video phase.

These results are the before-refactor behavior contract.

## Scope

The first implementation phase covers:

- `src/ops/`
- `src/ml/runtime/`
- shared neural-network, backbone, transformer, SAM, and grounding components
- image and grounding blocks
- image and grounding model assembly
- the thin video model wrapper when required by the `src/ml/model.py` split
- official SAM3.1 checkpoint mapping in `src/io/checkpoint.py`
- imports and tests that must follow the new paths

The following work is outside this phase:

- restructuring the video tracker state machine
- restructuring data loading, augmentation, or finetuning workflows
- restructuring prediction workflows or example scripts beyond import updates
- compatibility with LoRA checkpoints produced by the current rewrite
- loading the text encoder or VLM at runtime
- Hugging Face checkpoint loading

The later video, data/finetuning, and prediction phases will each receive a
separate design and implementation plan.

## Dependency Direction

All dependencies flow in one direction:

```text
data/io/ops/runtime -> components -> blocks -> model -> build/predict -> scripts
```

Branching inside a layer is allowed, but upward imports and callbacks are not.

- `components` do not import `blocks`, `model`, `build`, or `predict`.
- `blocks` do not import `model`, `build`, or `predict`.
- `model` assembles and coordinates blocks; it does not reach into predictor
  code.
- `predict` uses public model methods and does not import model components.
- `scripts` use build and prediction APIs rather than model internals.

An AST-based architecture test enforces these rules. It replaces the current
large blacklist of historical paths with a small set of dependency invariants.

## Target Structure

This is the planned end-state structure. The core phase realizes the shared,
image, grounding, and model paths shown below. The video block branch keeps its
current internal layout until the separate video phase.

```text
src/ml/
|-- runtime/
|   |-- attention.py
|   |-- checkpointing.py
|   `-- fused.py
|-- components/
|   |-- nn/
|   |   |-- activation.py
|   |   |-- attention.py
|   |   |-- layers.py
|   |   `-- position.py
|   |-- backbone/
|   |   |-- vit.py
|   |   `-- neck.py
|   |-- transformer/
|   |   |-- encoder.py
|   |   |-- decoder.py
|   |   |-- video.py
|   |   `-- model.py
|   |-- sam/
|   |   |-- prompt_encoder.py
|   |   |-- mask_decoder.py
|   |   |-- transformer.py
|   |   `-- rope.py
|   `-- grounding/
|       |-- geometry.py
|       |-- prompt.py
|       |-- scoring.py
|       |-- segmentation.py
|       `-- sequence.py
|-- blocks/
|   |-- vision.py
|   |-- image/
|   |   |-- features.py
|   |   |-- prompt.py
|   |   `-- masks.py
|   |-- grounding/
|   |   |-- tokens.py
|   |   |-- image.py
|   |   |-- prompt.py
|   |   `-- decoder.py
|   `-- video/             # planned; current video blocks stay in place for now
`-- model/
    |-- image.py
    |-- grounding.py
    `-- video.py
```

Construction helpers currently stored in `components/*/create.py` move to the
block that owns the assembled graph. For example, `blocks/vision.py` owns the
ViT/neck assembly and `blocks/grounding/decoder.py` owns the grounding
transformer, scorer, and segmentation-head assembly. Components remain usable
mathematical modules rather than application assemblers.

`components/transformer/decoder.py` keeps only the active image/grounding
decoder implementation. Active video rotary attention and the decoupled video
decoder layer move to `components/transformer/video.py`. The unreferenced
`TransformerEncoderCrossAttention`, `TransformerDecoderLayerv1`, and
`TransformerDecoderLayerv2` implementations are removed.

## Naming

Architectural boundaries use full role names. Standard mathematical and model
abbreviations such as FPN, MLP, IoU, QKV, RoI, and RoPE remain where they are
conventional.

| Current | Target |
| --- | --- |
| `components/nn/modules.py` | `components/nn/attention.py` |
| `VisualCond` | `VisualTokens` |
| `VisionCore` | `VisionEncoder` |
| `SamImage` | `ImageFeatures` |
| `SamPrompt` | `ImagePromptEncoder` |
| `SamMask` | `ImageMaskDecoder` |
| `GroundImage` | `GroundingImage` |
| `GroundPrompt` | `GroundingPromptEncoder` |
| `GroundDec` | `GroundingDecoder` |
| `SimpleRoPEAttention` | `RotaryAttention` |
| `DecoupledTransformerDecoderLayerv2` | `VideoDecoderLayer` |
| `invert_sigmoid` | `inverse_sigmoid` |
| `convert_to_xyxy` | `cxcywh_to_xyxy` |
| `filter_boxes` | `nms_indices` |
| `from_ckpt` | `load_weights` |
| `image_pe` | `get_image_position_encoding` |
| `seq` | `flatten_spatial` |
| `tensor` | `unwrap_tensor` |

Naming rules:

- public graph constructors use `build_*`
- private block constructors use `_make_*`
- computation stages use `encode_*`, `decode_*`, or `predict_*`
- loading methods use `load_weights`
- `get_*` is reserved for simple retrieval
- coordinate conversion names state both source and destination formats
- ambiguous boundary names such as `cond`, `dec`, `feat`, `mem`, `out`, and
  generic `run_*` are replaced with the represented value or operation
- short local tensor names remain acceptable when the mathematical convention
  makes their meaning clear

## Model Data Flow

Image segmentation follows one forward path:

```text
image
-> VisionEncoder
-> ImageFeatures
-> ImagePromptEncoder
-> ImageMaskDecoder
-> masks and scores
```

Visual-token grounding follows one forward path with three inputs joining at
the decoder:

```text
image -> VisionEncoder -> GroundingImage ------------------+
geometry prompt -> GroundingPromptEncoder -----------------+-> GroundingDecoder
visual_token.pt -> VisualTokens ----------------------------+
                                                   -> logits, boxes, masks
```

The implementation continues to use simple tensors and named dictionaries at
block boundaries. It does not add wrapper dataclasses solely for the refactor.

## Mathematical Invariants

### Box Coordinates

The grounding decoder uses normalized `cxcywh` reference boxes. Conversion to
`xyxy` occurs only at boundaries that require corner coordinates, including
relative position bias, NMS, and output formatting. No implicit clipping or
coordinate-order change is introduced.

Iterative box refinement remains:

```python
next_box = sigmoid(delta + inverse_sigmoid(reference_box))
```

The reference box passed to the next decoder layer remains detached, preserving
the current gradient path and upstream behavior.

### Scores and Presence

Class and presence evidence remain combined in probability space and converted
back to logit space:

```python
combined_logit = logit(sigmoid(class_logit) * sigmoid(presence_logit))
```

The output remains clamped to the current stable range. The rewrite keeps its
existing correction for the upstream unassigned `clamp()` result.

### Attention

- Attention tensors use `(batch, heads, query, channels)` at the SDPA boundary.
- A boolean attention or padding mask uses `True` for an excluded position.
- SDPA applies `1 / sqrt(head_dim)`; callers do not apply that scale twice.
- The active Vanilla attention calculation and operation order remain intact.
- Unused XFormers, Sparse, and FA3 branches are removed from the local minimal
  runtime rather than silently falling back to another backend.

### RoPE and Position Encoding

- Complex and real RoPE paths must remain numerically equivalent.
- RoPE channels must be even.
- Repeated key frequencies require an integral key/query sequence-length ratio.
- Spatial flattening is always `(B, C, H, W) -> (H * W, B, C)` at sequence-first
  transformer boundaries.
- Height and width retain their order in spatial shapes, valid ratios, grids,
  and positional encodings.

The current positional-encoding cache creates CUDA tensors during module
construction and keys entries only by spatial size. The refactor makes cache
creation input-device aware so CPU construction works and cached tensors are
never returned on the wrong device. The generated values remain unchanged.

### Changes to Formulas

Unused calculations may be removed. Active formulas are not algebraically
rewritten unless a failing invariant test demonstrates a defect and the change
is separately verified against upstream behavior. Structural moves and formula
changes do not share the same review unit.

## SAM3.1 Checkpoint Contract

The supported checkpoint contract is the official SAM3.1 base checkpoint from
an explicit local path.

```text
SAM3.1 checkpoint
-> official source-key parsing
-> canonical logical block states
-> load_weights(strict=True)
-> image and grounding parity checks
```

Requirements:

- Official SAM3.1 image and grounding weights load after internal module and
  parameter names change.
- Key translation has one owner: `src/io/checkpoint.py`.
- Every required block load is strict. Missing required keys and unexpected
  keys fail with the block name and load report.
- Only intentionally excluded categories, such as the language encoder, are
  ignored explicitly.
- `visual_token.pt` continues to provide cached visual tokens for no-text
  grounding.
- The runtime does not instantiate the VLM or text encoder.
- Hugging Face loading is not added.
- Existing LoRA checkpoints produced by this rewrite are unsupported. The
  refactor does not add aliases or migration code for their adapter keys.

## Error Handling

- Tensor rank, coordinate representation, feature-level count, and spatial
  shape are checked at block boundaries rather than repeatedly inside helpers.
- Shape errors report both the expected contract and the received shape.
- Image segmentation without any point, box, or mask prompt fails explicitly.
  Grounding may omit a geometry prompt because cached visual tokens are a valid
  condition. Malformed provided prompts or masks still fail explicitly.
- Missing SAM3.1 weights and unsupported attention backends fail explicitly.
- Device placement is correct at creation; code does not conceal a CPU/CUDA
  mismatch with broad automatic fallback.
- Broad `try/except`, silent fallback, and redundant defensive wrappers are not
  introduced.

## Verification

### Mathematical Unit Tests

- `sigmoid(inverse_sigmoid(x))` recovers values away from the clamp boundary.
- `cxcywh_to_xyxy` matches hand-calculated coordinate examples.
- Complex and real RoPE implementations agree for fixed tensors.
- RoPE preserves paired-vector norms within floating-point tolerance.
- Boolean attention masks exclude exactly the marked positions.
- Iterative box refinement matches the logit-space formula.
- Presence combination matches the probability-product formula and stable
  clamp range.
- Spatial flattening preserves the documented height/width order.

### Architecture Tests

- The import graph obeys the one-way dependency ranks.
- Components do not import blocks or models.
- Blocks do not import models or predictors.
- Models assemble blocks without importing predictors.
- Predictors do not import component or block internals.
- Target files exist and replaced historical module paths are absent, without a
  long blacklist of every previously attempted name.

### Checkpoint Tests

- Representative official SAM3.1 keys map to their canonical block states.
- Removing a required key produces a strict load failure.
- The full local SAM3.1 checkpoint loads into image and grounding models.
- No test promises compatibility with old LoRA adapter keys.

### Behavior and Runtime Tests

- The updated full test suite passes.
- `scripts/parity_image.py` continues to report zero max/mean differences and
  zero mask XOR pixels on the current CUDA machine.
- `scripts/parity_ground.py` continues to report zero max/mean differences and
  zero mask XOR pixels on the current CUDA machine.
- Image and grounding model construction succeeds on CPU.
- Focused forward/backward tests produce finite gradients for affected active
  components.

### Style Tests

- Ruff reports no error in changed files.
- Black with line length 88 reports no formatting change in changed files.
- Unrelated video files are not reformatted in this phase.
- A later repository-wide style phase makes the full `src`, `tests`, and
  `scripts` checks clean.

## Implementation Order

1. Add invariant and architecture tests against the current implementation.
2. Rename stateless math functions and update their call sites.
3. Separate active transformer implementations and remove unreachable decoder
   and attention branches.
4. Move image and grounding blocks into responsibility-based packages.
5. Split `src/ml/model.py` into the model package and update boundary imports.
6. Centralize the official SAM3.1 key adapter and enforce strict block loads.
7. Make positional-encoding caching device aware.
8. Run focused tests, the full suite, formatting checks, and both CUDA parity
   scripts.

Each step leaves a working, independently verifiable tree. Structural movement
is verified before any behavior-affecting correction begins.

## Success Criteria

The core phase is complete when all of the following are true:

- the import graph is one-way and enforced by tests
- in-scope core folders and names match this design
- unreferenced decoder generations and unsupported attention branches are gone
- official SAM3.1 image and grounding weights load strictly from local paths
- no compatibility layer is added for old LoRA adapter keys
- CPU construction succeeds
- mathematical invariant tests pass
- the complete updated test suite passes
- image and grounding CUDA parity remain exactly zero on the current machine
- changed files pass Ruff and Black
- `sam3-main/`, `weight/`, and `asset/` remain unmodified and unstaged
