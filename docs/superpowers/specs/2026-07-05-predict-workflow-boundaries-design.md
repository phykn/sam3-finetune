# Predict Workflow Boundary Redesign

## Goal

Make `src/predict` match the five user-facing prediction workflows directly:

1. Prompted prediction from a point, box, or mask prompt.
2. Grid prediction that sweeps point prompts over an image.
3. Refinement that reuses an existing low-resolution mask/logit as `mask_input`.
4. Context prediction that uses a previous good mask as a clue to find the object again.
5. Next-frame prediction that uses a previous frame mask as video memory for one target frame.

The current code has most of this behavior, but the boundaries are named by
implementation area (`image`, `masks`, `reference`, `video`) rather than by the
workflow the user is trying to run. The redesign makes the workflow boundary
visible in the package layout.

## Non-Goals

- Do not change model weights, checkpoint loading semantics, or inference math.
- Do not modify `sam3-main/` or `weight/`.
- Do not keep compatibility wrappers for the old `src.predict.image`,
  `src.predict.masks`, `src.predict.reference`, or `src.predict.video` paths.
- Do not expand next-frame prediction into a full streaming tracker loop.
- Do not implement pixel-level boundary sharpening beyond the existing
  `mask_input` refinement capability.

## Package Layout

```text
src/predict/
  prompted/
  grid/
  refine/
  context/
  next_frame/
```

### `src.predict.prompted`

Owns basic SAM prompt decoding for one image embedding or one image.

Public API:

```python
from src.predict.prompted import Sam3Predictor
```

Responsibilities:

- Encode image tensors and image batches.
- Decode point, box, and mask prompts.
- Return full-resolution masks, scores, and low-resolution masks.
- Keep `mask_input` support as a primitive prompt type because the lower-level
  predictor must still know how to pass it into the prompt encoder.

### `src.predict.grid`

Owns automatic grid point generation and proposal filtering.

Public API:

```python
from src.predict.grid import AutomaticMaskGenerator
```

Responsibilities:

- Build grid points.
- Run prompted prediction over grid points and optional crops.
- Filter proposals by predicted IoU, stability, area, crop edge contact, and NMS.
- Convert proposals into `MaskInstance` values.

This package does not own refinement. It may expose proposal utilities, but
second-pass `mask_input` refinement belongs to `src.predict.refine`.

### `src.predict.refine`

Owns explicit second-pass refinement.

Public API:

```python
from src.predict.refine import MaskRefiner
```

Responsibilities:

- Accept an image or image embedding plus an existing low-resolution mask/logit.
- Preserve the original point and box prompts when refining.
- Call prompted prediction with `mask_input` and `multimask_output=False`.
- Provide a small helper for selecting the best candidate from a multimask first
  pass when a caller wants the common two-step flow:
  first decode multimask, then refine the selected low-resolution mask.

This package makes the "3. correction/refinement" workflow visible instead of
leaving it as an implicit pattern in scripts.

### `src.predict.context`

Owns finding an object again from previous good mask context.

Public API:

```python
from src.predict.context import ContextMatcher, ReferenceGuidedMaskGenerator
```

Responsibilities:

- Build visual prototypes from reference image+mask pairs.
- Score target image features against positive and optional negative context.
- Select target candidate points or rerank grid proposals.
- Optionally use the reference mask shape as a `mask_input` prior.

This package should be described as "find again from context", not "copy the
old mask".

### `src.predict.next_frame`

Owns one-step video memory prediction.

Public API:

```python
from src.predict.next_frame import MemoryReference, NextFramePredictor
```

Responsibilities:

- Accept one or more reference frame masks.
- Add reference masks to video memory.
- Predict the target frame mask by propagating to that target frame.
- Optionally combine reference memory with target points.

The public class should be named `NextFramePredictor` because the current scope
is a single target frame prediction, not a full streaming video application.

## Import Migration

Old paths are removed and tests/scripts migrate to the new paths.

```python
from src.predict.image import Sam3Predictor
from src.predict.masks import AutomaticMaskGenerator
from src.predict.reference import ContextMatcher
from src.predict.video import VideoMemoryInference
```

becomes:

```python
from src.predict.prompted import Sam3Predictor
from src.predict.grid import AutomaticMaskGenerator
from src.predict.context import ContextMatcher
from src.predict.next_frame import NextFramePredictor
```

The root `src.predict` package should not re-export workflow classes. Callers
must import from the workflow package that matches the operation they intend to
run. The root `src` package should also stop re-exporting `Sam3Predictor`.

## Data Flow

Prompted:

```text
image -> Sam3Predictor.set_image/encode_image -> predict(point/box/mask_input)
```

Grid:

```text
image -> grid points/crops -> prompted prediction -> proposal filters -> proposals
```

Refine:

```text
image embedding + original prompts + selected low_res_mask -> prompted prediction
with mask_input -> refined mask
```

Context:

```text
reference image+mask -> prototype -> target similarity -> candidate points or
candidate rerank -> predicted target masks
```

Next frame:

```text
reference frame image+mask -> video memory -> target frame -> propagated masks
```

## Error Handling

- Keep shape validation close to the package that owns the workflow.
- Reference context errors should mention reference image/mask alignment.
- Next-frame errors should mention reference frame mask alignment and target
  point mode.
- Refinement errors should distinguish missing first-pass low-resolution masks
  from invalid prompt shapes.

## Testing

Update tests to import from the new package layout and keep coverage for:

- Prompted point, box, mask-only, batched embedding, and batched prompt decoding.
- Grid generation, crop grids, proposal filtering, and proposal-to-instance
  conversion.
- Refinement preserving original prompts while passing `mask_input`.
- Context prototype selection, explicit target points, reference shape prior,
  reranking, and NMS.
- Next-frame memory references, target points, memory target point mode, and mask
  resizing.

The focused verification command after implementation should be:

```bash
python -m pytest tests/test_predictor_api.py tests/test_auto_mask_generator.py tests/test_context_predictor.py tests/test_reference_guided_masks.py tests/test_video_memory.py tests/test_video_memory_reference.py
```

If import movement touches broader package structure, also run:

```bash
python -m pytest tests/test_package_structure.py tests/test_mask_instances.py tests/test_transforms.py
```

## Migration Notes

Because this is a complete boundary change, old import paths should fail rather
than silently forwarding. This is intentional: callers should choose a workflow
package and the code should make the 1-5 split obvious.
