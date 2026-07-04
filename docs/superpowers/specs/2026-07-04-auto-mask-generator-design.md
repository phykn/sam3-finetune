# Automatic Mask Generator Design

## Goal

Add a SAM-style automatic mask generator on top of the existing minimal `src`
runtime. The generator should extract many mask proposals from one image without
language prompts, Hugging Face access, detector text paths, tracking memory, or
new model weights.

The first version targets a balanced full-image pass:

- Reuse `Sam3Predictor.set_image()` and `Sam3Predictor.predict()`.
- Generate a uniform grid of positive point prompts.
- Batch prompts for GPU efficiency.
- Filter proposals by score, stability, area, and duplicate overlap.
- Save smoke-test visualizations for `asset/sample.jpg`.

## Non-Goals

- No text/language prompting.
- No detector branch revival.
- No video propagation, memory encoder, object pointer state, or tracking
  session logic.
- No crop-layer pyramid in the first version.
- No semantic classification of masks into named object classes.
- No checkpoint upload or redistribution.

## Public API

Create `src/auto_mask_generator.py`.

Primary entrypoint:

```python
from src.auto_mask_generator import Sam3AutomaticMaskGenerator

generator = Sam3AutomaticMaskGenerator.from_checkpoint(
    "weight/sam3.1_multiplex.pt",
    device="cuda",
)
proposals = generator.generate(image)
```

Alternative construction from an existing predictor:

```python
generator = Sam3AutomaticMaskGenerator(predictor)
```

Proposal object:

```python
MaskProposal(
    segmentation: np.ndarray,      # bool mask, H x W
    bbox: tuple[int, int, int, int],# xyxy pixel box
    area: int,
    predicted_iou: float,
    stability_score: float,
    point_coords: tuple[float, float],
    crop_box: tuple[int, int, int, int],
)
```

Default balanced settings:

```python
points_per_side=32
points_per_batch=64
pred_iou_thresh=0.0
stability_score_thresh=0.75
stability_score_offset=1.0
min_mask_region_area=0
box_nms_thresh=0.7
max_masks=None
```

The thresholds are intentionally conservative for the first version. They should
be easy to tune from the constructor without touching model code.

## Components

### `MaskProposal`

A dataclass that stores one candidate mask and its metadata. It should stay
simple and serializable enough for later JSON/image export helpers.

### Grid Point Generation

Function: `build_point_grid(points_per_side: int) -> np.ndarray`

Returns normalized point coordinates in `[0, 1]` with shape
`(points_per_side * points_per_side, 2)`. The points should be centered inside
cells rather than placed directly on image borders.

Pixel coordinates are computed per image as:

```python
x = normalized_x * width
y = normalized_y * height
```

### Candidate Generation

`Sam3AutomaticMaskGenerator.generate(image)` should:

1. Call `predictor.set_image(image)` once.
2. Build the grid.
3. Convert grid points to pixel coordinates.
4. Process points in batches of `points_per_batch`.
5. Call `predictor.predict(point_coords=batch[:, None, :], point_labels=ones)`.
6. Flatten the multimask outputs into individual candidates.
7. Compute metadata and filters.
8. Return proposals sorted by quality.

The predictor already supports batched point prompts and uses the corrected
SAM3.1 multiplex path:

- `interactivity_no_mem_embed`
- mask-only dummy point behavior
- `repeat_image=True`
- high-resolution decoder features

### Stability Score

Compute a SAM-style stability score from low-resolution mask logits:

```python
intersection = (logits > threshold + offset).sum()
union = (logits > threshold - offset).sum()
score = intersection / max(union, 1)
```

Use `threshold=0.0` and `offset=stability_score_offset` by default.

### BBox And Area

Area is `mask.sum()`.

The bbox should be derived from the final full-resolution bool mask:

- Empty masks are discarded.
- Non-empty bbox is `(x0, y0, x1, y1)` with inclusive-exclusive coordinates.

### Duplicate Filtering

Use box IoU based NMS for the first version.

1. Filter by predicted IoU.
2. Filter by stability score.
3. Filter by `min_mask_region_area`.
4. Apply box NMS sorted by `(predicted_iou, stability_score, area)`.

Box NMS is cheaper and sufficient for the first balanced version. Mask NMS can be
added later if duplicate quality is poor.

### Visualization Helpers

Create helper functions in `src/auto_mask_generator.py` or a small script-local
helper if they do not need to be public:

- Save one composite overlay with distinct colors.
- Save a mask grid/contact sheet for inspection.

Avoid adding UI or notebooks in this step.

## Smoke Script

Create `scripts/auto_mask_smoke_test.py`.

Behavior:

1. Load `asset/sample.jpg`.
2. Load local checkpoint from `weight/sam3.1_multiplex.pt`.
3. Generate automatic masks with balanced defaults.
4. Print proposal count and top proposal metadata.
5. Write:
   - `outputs/auto_masks_overlay.png`
   - `outputs/auto_masks_grid.png`

The script must not write under `sam3-main/` or `weight/`.

## Tests

Unit tests should avoid loading the real checkpoint.

Create `tests/test_auto_mask_generator.py` covering:

- Grid generation shape and range.
- BBox extraction from bool masks.
- Stability score computation.
- Box IoU/NMS duplicate filtering.
- Generator batching with a fake predictor.
- Proposal sorting and `max_masks` behavior.

Existing tests should continue passing:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Real checkpoint verification:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py
```

## Error Handling

- `points_per_side` and `points_per_batch` must be positive integers.
- Empty masks are skipped.
- If all candidates are filtered out, return an empty list and still allow the
  smoke script to write a clear message instead of failing obscurely.
- If CUDA is unavailable, the smoke script should raise a clear error. Unit tests
  remain CPU-only.

## Performance

Balanced default produces `32 * 32 = 1024` point prompts. With
`points_per_batch=64`, this is 16 predictor calls after one image embedding pass.

The first implementation should favor correctness and clear metadata over heavy
optimization. Later performance options can include:

- Lower `points_per_side`
- Larger `points_per_batch`
- Crop layers
- Mask NMS on GPU

## Git And Artifact Rules

- Commit code, tests, scripts, and docs only.
- Do not commit `sam3-main/`, `weight/`, `.venv/`, or `outputs/`.
- Generated PNGs remain ignored under `outputs/`.

## Open Decisions Resolved

- Default approach: balanced full-image grid.
- Crop layers: not in first version.
- Output format: dataclass list plus smoke PNGs.
- Core model path: existing `Sam3Predictor` only.
