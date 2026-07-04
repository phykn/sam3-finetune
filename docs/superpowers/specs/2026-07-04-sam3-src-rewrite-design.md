# SAM3 Src Rewrite Design

## Goal

Create a new minimal `src/` implementation for prompted segmentation using local
SAM 3.1 weights. Keep `sam3-main/` unchanged and use it only as reference source
for copying and adapting the required code.

The first usable result is a Python API plus smoke test that loads
`weight/sam3.1_multiplex.pt`, runs prompted segmentation on `asset/sample.jpg`,
and writes a mask visualization under `outputs/`.

## Scope

Implement only the image interactive segmentation path:

- image preprocessing and postprocessing
- image feature extraction needed by the interactive path
- point prompt support
- box prompt support
- mask prompt support
- prompt encoder
- mask decoder
- local checkpoint loading and key mapping
- smoke test script

Exclude:

- Hugging Face downloads or APIs
- language/text prompts
- open-vocabulary detector use
- video propagation and tracking sessions
- training, eval, datasets, notebooks, SLURM, distributed code
- broad cleanup in `sam3-main/`

## Layout

Use a flat module layout under `src/`; do not create a nested `sam3lite/`
package.

```text
src/
  __init__.py
  predictor.py
  builder.py
  checkpoint.py
  transforms.py
  image_encoder.py
  prompt_encoder.py
  mask_decoder.py
  transformer.py
  position_encoding.py
  common.py
scripts/
  smoke_test.py
outputs/
```

The exact file list may shrink if copied code proves unnecessary, but it should
not grow into unrelated detector, training, eval, or video modules.

## Public API

Expose one primary class from `src/predictor.py`:

```python
predictor = Sam3Predictor.from_checkpoint("weight/sam3.1_multiplex.pt")
predictor.set_image(image)
masks, scores, low_res_logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=None,
    mask_input=None,
    multimask_output=True,
)
```

Coordinates use pixel-space image coordinates by default:

- point coordinates: `N x 2`, `(x, y)`
- point labels: `N`, `1` for foreground and `0` for background
- box: `4`, `(x0, y0, x1, y1)`
- mask input: low-resolution logits from a previous prediction, or a compatible
  single-mask input

## Checkpoint Strategy

Load only local checkpoints. The default smoke test checkpoint is:

```text
weight/sam3.1_multiplex.pt
```

The checkpoint includes both detector and tracker weights. The rewrite should
load only the interactive tracker/image segmentation keys needed for the new
`src/` modules, especially keys under:

- `tracker.model.interactive_sam_prompt_encoder.*`
- `tracker.model.interactive_sam_mask_decoder.*`
- required image backbone and feature projection keys

The loader must report missing and unexpected keys clearly. Do not silently
ignore broad key mismatches while claiming the model is loaded correctly.

## Implementation Approach

Copy the smallest required source from `sam3-main/` into `src/`, then adapt it:

1. Copy prompt encoder, mask decoder, transformer, position encoding, transforms,
   and common layer utilities needed by interactive segmentation.
2. Copy only the image feature path required by the interactive SAM decoder.
3. Replace `sam3.*` imports with local `src` imports.
4. Remove text/language, detector, video propagation, training, and eval code.
5. Add checkpoint key remapping from SAM 3.1 checkpoint names to local module
   names.
6. Add a predictor facade with simple image and prompt methods.
7. Add a smoke test that saves visual outputs.

## Runtime Assumptions

- Use the workspace-local `.venv`.
- Use local weights only.
- CUDA is expected for real inference. The current detected GPU is an NVIDIA
  GeForce RTX 2060, so the first version should avoid FlashAttention 3-specific
  assumptions and use conservative CUDA paths.
- If CUDA is unavailable, the smoke test should fail with a clear environment
  error instead of falling back to an unvalidated CPU path.

## Git And Publishing

Use this GitHub remote for the project:

```text
https://github.com/phykn/sam3-finetune.git
```

Do not upload these local directories:

- `sam3-main/`
- `weight/`

The root `.gitignore` must keep those paths ignored. Before any commit or push,
verify the staged and unstaged status so the reference source and local weights
are not included.

## Smoke Test

Create `scripts/smoke_test.py` that:

1. Adds the workspace root to `sys.path` and imports `Sam3Predictor` from
   `src.predictor`.
2. Loads `asset/sample.jpg`.
3. Loads `weight/sam3.1_multiplex.pt`.
4. Runs at least one deterministic prompt, preferably a box around the center of
   the image plus one foreground point.
5. Writes outputs such as:

```text
outputs/smoke_mask.png
outputs/smoke_overlay.png
```

The script should print:

- loaded checkpoint path
- device name
- missing/unexpected key summary
- output mask shape
- score values
- output file paths

## Verification

Minimum verification for the first implementation:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

Success requires:

- checkpoint loads from `weight/` without Hugging Face access
- language/text code is not required by the new `src/` runtime path
- inference produces at least one mask for `asset/sample.jpg`
- output image files are created under `outputs/`
- missing/unexpected key report is small enough to explain based on excluded
  detector/text/video components

If the full smoke test fails, preserve the failure details and narrow the next
step to the first failing stage: imports, model construction, checkpoint mapping,
image features, prompt encoding, or mask decoding.
