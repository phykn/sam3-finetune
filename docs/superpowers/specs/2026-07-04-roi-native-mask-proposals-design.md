# ROI-Native Mask Proposals Design

## Goal

Change automatic mask proposals to store masks in ROI coordinates instead of
full-image coordinates. The crop pyramid can produce many small masks, and
allocating one full `H x W` bool array per proposal wastes memory and work.

This is an intentional breaking change for the new `src` API.

## Current Problem

`Sam3AutomaticMaskGenerator` currently expands every surviving crop-local mask
into a full-image bool array:

```python
full_mask = np.zeros((full_height, full_width), dtype=bool)
full_mask[crop_y0:crop_y1, crop_x0:crop_x1] = mask
```

That makes `MaskProposal.segmentation.shape == (image_height, image_width)`.
For small objects this stores mostly false pixels. With crop grids such as
`[1, 2, 4]`, proposal count increases and the memory waste grows.

## Decision

Use ROI-native proposal masks.

`MaskProposal.segmentation` becomes the mask cropped to its own `bbox`, not the
full image.

```python
MaskProposal(
    segmentation=roi_mask,       # bool mask, bbox_h x bbox_w
    bbox=(x0, y0, x1, y1),       # full-image xyxy
    image_size=(width, height),  # original full-image size
    crop_box=(cx0, cy0, cx1, cy1),
)
```

The `bbox` remains the source of truth for where the ROI mask belongs in the
original image. `image_size` allows helper functions to reconstruct a full-size
mask when needed.

## Public API Changes

Extend `MaskProposal`:

```python
@dataclass(frozen=True)
class MaskProposal:
    segmentation: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    predicted_iou: float
    stability_score: float
    point_coords: tuple[float, float]
    crop_box: tuple[int, int, int, int]
    crop_grid: int = 1
    crop_index: int = 0
    image_size: tuple[int, int] | None = None
```

Field semantics:

- `segmentation`: bbox-local bool mask with shape `(bbox_y1 - bbox_y0,
  bbox_x1 - bbox_x0)`.
- `bbox`: full-image inclusive-exclusive xyxy coordinates.
- `image_size`: original full-image `(width, height)`.
- `crop_box`: full-image inclusive-exclusive crop coordinates.

`image_size` may default to `None` only to keep manual construction in existing
tests convenient. Generator-created proposals must always set it.

Add helpers:

```python
def proposal_to_full_mask(proposal: MaskProposal) -> np.ndarray
def proposal_mask_image(proposal: MaskProposal, alpha: int = 255) -> Image.Image
```

`proposal_to_full_mask` requires `proposal.image_size` and returns a full-image
bool mask.

## Data Flow

Inside `_proposals_from_batch`:

1. Run model on a crop image.
2. Convert each raw crop-local mask to bool.
3. Compute `local_bbox = mask_to_box(mask)`.
4. Apply score, stability, area, and crop-edge filtering using local data.
5. Convert `local_bbox` to global `bbox`.
6. Slice the raw mask to ROI:

```python
lx0, ly0, lx1, ly1 = local_bbox
roi_mask = mask[ly0:ly1, lx0:lx1].copy()
```

7. Store `segmentation=roi_mask`, `bbox=global_bbox`,
   `image_size=(full_width, full_height)`.

Do not allocate full-image masks during proposal generation.

## Visualization

Update visualization helpers to consume ROI-native masks directly.

For overlay:

1. Create the transparent overlay once at full image size.
2. For each proposal, convert `proposal.segmentation` to an alpha mask.
3. Create an RGBA ROI layer with size `(bbox_w, bbox_h)`.
4. Paste that ROI layer at `(bbox_x0, bbox_y0)` using the ROI alpha mask.

For grid:

1. Composite each ROI mask onto a full-image tile using the same paste logic.
2. Draw `proposal.bbox`.
3. Resize the tile for the contact sheet.

This keeps visualization output identical in image coordinates without storing
full-image masks in every proposal.

## NMS And Filtering

Box NMS remains unchanged because it already uses `proposal.bbox`.

Area remains the number of true pixels in the original crop-local mask. Since
`roi_mask` is the tight bbox slice of that mask, `roi_mask.sum()` must equal the
stored `area`.

Crop-edge filtering still uses `local_bbox` before ROI slicing.

## Tests

Update existing tests:

- Full-image non-crop generation should expect
  `proposal.segmentation.shape == (bbox_h, bbox_w)`, not `(image_h, image_w)`.
- Crop generation should assert proposal masks are ROI-sized and that
  `proposal.image_size == (width, height)`.
- Add `proposal_to_full_mask` tests to verify ROI masks reconstruct full-image
  masks at the correct bbox location.
- Add overlay/grid smoke-level tests with fake proposals if needed to confirm
  helpers accept ROI masks.

Real checkpoint smoke tests remain:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 32 32 --max-masks 200
```

The proposal counts may change only if filtering semantics accidentally change.
The intended change is representation and memory behavior, not segmentation
quality.

## Error Handling

- `proposal_to_full_mask` raises `ValueError` if `image_size is None`.
- `proposal_to_full_mask` raises `ValueError` if `segmentation.shape` does not
  match `bbox` size.
- Visualization helpers should also fail clearly through the same validation
  path if proposal geometry is inconsistent.

## Git And Artifact Rules

- Commit code, tests, scripts, and docs only.
- Do not stage or commit `sam3-main/`, `weight/`, `.venv/`, or `outputs/`.
- Generated PNGs remain ignored under `outputs/`.

## Open Decisions Resolved

- Use breaking change option.
- `segmentation` means ROI mask, not full-image mask.
- Keep `bbox` in full-image coordinates.
- Add `image_size` for full-mask reconstruction.
- Avoid full-image allocation during proposal generation.
