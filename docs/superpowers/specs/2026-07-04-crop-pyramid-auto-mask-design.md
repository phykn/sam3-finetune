# Crop Pyramid Automatic Mask Design

## Goal

Improve small-object recall in the automatic mask generator by running point
prompt grids over overlapping image crops. The user wants explicit crop grid
control:

```python
crop_grids=[1, 2, 4]
crop_points_per_side=[32, 32, 32]
```

This means:

- `1` runs one full-image `1x1` crop.
- `2` runs four `2x2` crops.
- `4` runs sixteen `4x4` crops.
- Each entry in `crop_points_per_side` controls the point grid used inside each
  tile for the matching crop grid level.

The first implementation should favor recall, but it should avoid avoidable
Python and memory overhead.

## Non-Goals

- Do not use Hugging Face.
- Do not modify `sam3-main/`; it remains reference-only.
- Do not commit or redistribute checkpoints under `weight/`.
- Do not add language prompts, detector branches, tracking memory, or video
  propagation.
- Do not implement fully batched multi-image crop encoding in this first pass.
  That is a larger predictor API change and should be handled after measuring
  the crop-pyramid bottleneck.

## Public API

Extend `Sam3AutomaticMaskGenerator`.

```python
generator = Sam3AutomaticMaskGenerator.from_checkpoint(
    "weight/sam3.1_multiplex.pt",
    device="cuda",
    crop_grids=[1, 2, 4],
    crop_points_per_side=[32, 32, 32],
    crop_overlap_ratio=0.25,
    max_masks=200,
)
```

New constructor arguments:

```python
crop_grids: Sequence[int] | None = None
crop_points_per_side: Sequence[int] | None = None
crop_overlap_ratio: float = 0.25
crop_nms_thresh: float | None = None
max_masks_per_crop: int | None = None
filter_crop_edge_masks: bool = True
```

Compatibility:

- If `crop_grids is None`, keep the existing full-image behavior using
  `points_per_side`.
- If `crop_grids` is supplied, `crop_points_per_side` must also be supplied and
  must have the same length.
- `crop_grids=[1]` with `crop_points_per_side=[points_per_side]` is equivalent
  to the existing full-image path, apart from the internal crop scheduler.
- The generator should run exactly the crop grids provided. It should not
  automatically insert `1`; callers include `1` when they want a full-image
  pass.

Validation:

- Every `crop_grids` entry must be a positive integer.
- Every `crop_points_per_side` entry must be a positive integer.
- `crop_overlap_ratio` must be in `[0.0, 0.5)`.
- If `crop_nms_thresh is None`, use `box_nms_thresh` for per-crop NMS.
- `points_per_batch` remains a positive integer and applies within each crop.

## Crop Box Generation

For each grid value `g`, generate `g * g` crop boxes covering the whole image.

For `g == 1`:

```python
(0, 0, width, height)
```

For `g > 1`, generate overlapping tiles independently for x and y:

```python
overlap_w = round((width / g) * crop_overlap_ratio)
crop_w = ceil((width + overlap_w * (g - 1)) / g)
stride_w = crop_w - overlap_w
```

Then for each tile index:

```python
x0 = min(ix * stride_w, width - crop_w)
x1 = min(x0 + crop_w, width)
```

Use the same formula for height. This gives neighboring tiles an approximate
overlap of `crop_overlap_ratio` relative to the nominal tile size, while still
covering the image edges.

Crop boxes use inclusive-exclusive pixel coordinates:

```python
(x0, y0, x1, y1)
```

## Data Flow

`generate(image)` should:

1. Resolve image size.
2. Build crop jobs from `crop_grids` and `crop_points_per_side`.
3. Reuse normalized point grids for equal `points_per_side` values.
4. For each crop job:
   - Crop the PIL image or NumPy image.
   - Call `predictor.set_image(crop_image)`.
   - Convert the cached normalized grid to crop-local pixel coordinates.
   - Run prompt batches through `predictor.predict(...)`.
   - Filter proposals in crop-local coordinates before creating full-size masks.
   - Convert survivors to full-image coordinates.
   - Apply optional per-crop NMS and `max_masks_per_crop`.
5. Concatenate proposals from all crops.
6. Apply global NMS using full-image boxes.
7. Sort by `(predicted_iou, stability_score, area)` descending.
8. Apply global `max_masks`.

Extend `MaskProposal` with defaulted crop metadata for debugging and smoke
summaries:

```python
MaskProposal(
    segmentation: np.ndarray,       # bool mask, full image H x W
    bbox: tuple[int, int, int, int], # full-image xyxy
    area: int,
    predicted_iou: float,
    stability_score: float,
    point_coords: tuple[float, float], # full-image point coords
    crop_box: tuple[int, int, int, int],
    crop_grid: int = 1,
    crop_index: int = 0,
)
```

The new fields must have defaults so existing construction sites keep working.

## Local-To-Global Mapping

Each crop-local proposal should be converted to the original image coordinate
space:

- `global_x = local_x + crop_x0`
- `global_y = local_y + crop_y0`
- `global_bbox = local_bbox + crop offset`
- `global_point_coords = local_point_coords + crop offset`

Only filtered survivor masks should be expanded into a full-image bool mask:

```python
full_mask = np.zeros((image_height, image_width), dtype=bool)
full_mask[y0:y1, x0:x1] = crop_mask
```

This avoids allocating full-image masks for candidates that are filtered out by
score, stability, area, bbox, or crop-local NMS.

## Edge Handling

When `filter_crop_edge_masks=True`, discard masks that touch an internal crop
edge. A crop edge is internal when it is not also an original image edge.

This reduces partial-object masks from tile boundaries. Overlap should allow
the same object to be recovered from a neighboring tile where it is not clipped.

Callers can set `filter_crop_edge_masks=False` for maximum raw recall at the
cost of more duplicate and partial masks.

## Optimization Strategy

The first pass should be optimized within the existing predictor API:

- Precompute crop boxes before inference.
- Precompute and cache normalized point grids by `points_per_side`.
- Use vectorized coordinate scaling for each crop.
- Batch point prompts with the existing `points_per_batch`.
- Filter candidates before full-image mask expansion.
- Run per-crop NMS before global NMS to reduce duplicate pressure.
- Keep all image and mask loops at proposal/crop granularity, not per-pixel
  Python loops.

This still requires one `predictor.set_image(crop)` call per crop because the
current `Sam3Predictor` stores one feature set at a time.

## Future Optimization: Batched Crop Encoding

A later version can add a batched predictor path:

```python
predictor.set_image_batch(crop_images)
predictor.predict_on_image_batch(...)
```

The model's `encode_image()` likely accepts batched tensors, but the current
minimal predictor API and mask postprocessing are single-image oriented. This
should be a separate task because it must verify:

- image encoder batch output structure,
- high-resolution feature batch semantics,
- prompt-to-image matching in `mask_decoder` with `repeat_image`,
- per-image original sizes during postprocessing,
- memory usage on the available GPU.

## Smoke Script

Extend `scripts/auto_mask_smoke_test.py` with CLI options:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py `
  --crop-grids 1 2 `
  --crop-points-per-side 32 32 `
  --crop-overlap-ratio 0.25 `
  --max-masks 200
```

The script should print:

- checkpoint path,
- device name,
- crop grid configuration,
- elapsed time,
- total proposal count,
- proposal count by crop grid,
- top proposal metadata,
- output PNG paths.

Write visual outputs under ignored `outputs/`:

- `outputs/auto_masks_overlay.png`
- `outputs/auto_masks_grid.png`

## Tests

Unit tests should not load the real checkpoint.

Add tests for:

- crop box generation for `1`, `2`, and `4` grids,
- overlap coverage and valid inclusive-exclusive boxes,
- constructor validation for mismatched crop lists,
- crop-local point mapping to full-image coordinates,
- proposal bbox and `crop_box` metadata,
- filtering before full-image mask expansion using a fake predictor,
- per-crop and global NMS behavior with duplicate boxes,
- backward compatibility when `crop_grids is None`.

Real checkpoint verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 16 16
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 32 32 --max-masks 200
```

The smaller smoke command gives a fast correctness check. The `[1, 2]` command
is the routine recall-oriented inspection run. A `[1, 2, 4]` run remains
supported for manual experiments, but is too slow for normal verification on the
current GPU.

## Git And Artifact Rules

- Commit code, tests, scripts, and docs only.
- Do not stage or commit `sam3-main/`, `weight/`, `.venv/`, or `outputs/`.
- Generated PNGs remain ignored under `outputs/`.

## Open Decisions Resolved

- Use explicit crop grid lists instead of `crop_n_layers`.
- Use matching point-grid lists for per-tile point density.
- Prioritize small-object recall.
- Add v1 optimization within the current predictor API.
- Leave batched multi-image crop encoding for a later measured optimization.
