# ROI-Native Mask Proposals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store automatic mask proposal masks in bbox-local ROI coordinates instead of full-image coordinates.

**Architecture:** Keep `MaskProposal.bbox` in full-image coordinates and make `MaskProposal.segmentation` a tight ROI mask whose shape matches the bbox. Add explicit helpers for reconstructing full-image masks only when needed, then update generator output and visualization helpers to consume ROI masks directly.

**Tech Stack:** Python, NumPy, PIL, PyTorch-backed `Sam3Predictor`, pytest, local CUDA smoke tests.

---

## File Structure

- Modify: `src/auto_mask_generator.py`
  Adds `image_size` metadata, ROI validation helpers, full-mask reconstruction,
  ROI alpha image helper, ROI-native proposal generation, and ROI-aware
  visualization.
- Modify: `tests/test_auto_mask_generator.py`
  Updates proposal expectations from full-image masks to ROI masks and adds
  helper validation tests.
- No changes: `sam3-main/`, `weight/`, `.venv/`, `outputs/`.

### Task 1: ROI Helper API

**Files:**
- Modify: `tests/test_auto_mask_generator.py`
- Modify: `src/auto_mask_generator.py`

- [ ] **Step 1: Write failing ROI helper tests**

Update the import block in `tests/test_auto_mask_generator.py`:

```python
from src.auto_mask_generator import (
    MaskProposal,
    Sam3AutomaticMaskGenerator,
    batched,
    box_area,
    box_iou,
    build_point_grid,
    calculate_stability_score,
    count_proposals_by_crop_grid,
    generate_crop_boxes,
    mask_to_box,
    nms_boxes,
    proposal_mask_image,
    proposal_to_full_mask,
)
```

Append these tests after `test_mask_proposal_defaults_crop_metadata`:

```python
def test_proposal_to_full_mask_reconstructs_roi_mask():
    roi_mask = np.array([[True, False], [True, True]], dtype=bool)
    proposal = MaskProposal(
        segmentation=roi_mask,
        bbox=(2, 1, 4, 3),
        area=3,
        predicted_iou=0.9,
        stability_score=1.0,
        point_coords=(2.5, 1.5),
        crop_box=(0, 0, 5, 4),
        image_size=(5, 4),
    )

    full_mask = proposal_to_full_mask(proposal)

    expected = np.zeros((4, 5), dtype=bool)
    expected[1:3, 2:4] = roi_mask
    np.testing.assert_array_equal(full_mask, expected)


def test_proposal_to_full_mask_rejects_missing_image_size():
    proposal = MaskProposal(
        segmentation=np.ones((1, 1), dtype=bool),
        bbox=(0, 0, 1, 1),
        area=1,
        predicted_iou=0.9,
        stability_score=1.0,
        point_coords=(0.5, 0.5),
        crop_box=(0, 0, 1, 1),
    )

    try:
        proposal_to_full_mask(proposal)
    except ValueError as exc:
        assert "image_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_proposal_to_full_mask_rejects_shape_mismatch():
    proposal = MaskProposal(
        segmentation=np.ones((1, 2), dtype=bool),
        bbox=(0, 0, 3, 1),
        area=2,
        predicted_iou=0.9,
        stability_score=1.0,
        point_coords=(0.5, 0.5),
        crop_box=(0, 0, 3, 1),
        image_size=(3, 1),
    )

    try:
        proposal_to_full_mask(proposal)
    except ValueError as exc:
        assert "segmentation shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_proposal_mask_image_returns_roi_alpha_mask():
    proposal = MaskProposal(
        segmentation=np.array([[True, False]], dtype=bool),
        bbox=(1, 2, 3, 3),
        area=1,
        predicted_iou=0.9,
        stability_score=1.0,
        point_coords=(1.5, 2.5),
        crop_box=(0, 0, 4, 4),
        image_size=(4, 4),
    )

    mask_image = proposal_mask_image(proposal, alpha=7)

    assert mask_image.mode == "L"
    assert mask_image.size == (2, 1)
    np.testing.assert_array_equal(np.array(mask_image), np.array([[7, 0]], dtype=np.uint8))
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: FAIL during import because `proposal_to_full_mask` and
`proposal_mask_image` are not defined.

- [ ] **Step 3: Implement helper API**

In `src/auto_mask_generator.py`, extend `MaskProposal`:

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

Add:

```python
def _validate_roi_geometry(proposal: MaskProposal) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = proposal.bbox
    expected_shape = (y1 - y0, x1 - x0)
    if proposal.segmentation.shape != expected_shape:
        raise ValueError(
            "segmentation shape must match bbox size: "
            f"got {proposal.segmentation.shape}, expected {expected_shape}"
        )
    return x0, y0, x1, y1


def proposal_to_full_mask(proposal: MaskProposal) -> np.ndarray:
    if proposal.image_size is None:
        raise ValueError("proposal.image_size is required")
    x0, y0, x1, y1 = _validate_roi_geometry(proposal)
    width, height = proposal.image_size
    full_mask = np.zeros((height, width), dtype=bool)
    full_mask[y0:y1, x0:x1] = proposal.segmentation.astype(bool)
    return full_mask


def proposal_mask_image(proposal: MaskProposal, alpha: int = 255) -> Image.Image:
    _validate_roi_geometry(proposal)
    mask = proposal.segmentation.astype(np.uint8) * int(alpha)
    return Image.fromarray(mask, mode="L")
```

- [ ] **Step 4: Run ROI helper tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: PASS for helper tests and existing tests.

- [ ] **Step 5: Commit helper API**

Run:

```powershell
git add src/auto_mask_generator.py tests/test_auto_mask_generator.py
git commit -m "feat: add roi mask proposal helpers"
```

### Task 2: ROI-Native Generator Output

**Files:**
- Modify: `tests/test_auto_mask_generator.py`
- Modify: `src/auto_mask_generator.py`

- [ ] **Step 1: Update generator tests to expect ROI masks**

In `tests/test_auto_mask_generator.py`, update
`test_generator_batches_grid_points_and_returns_sorted_proposals`:

```python
    first = proposals[0]
    x0, y0, x1, y1 = first.bbox
    assert first.segmentation.shape == (y1 - y0, x1 - x0)
    assert first.segmentation.sum() == first.area
    assert first.image_size == (8, 8)
    assert proposal_to_full_mask(first).shape == (8, 8)
```

Replace the old assertion:

```python
    assert proposals[0].segmentation.shape == (8, 8)
```

In `test_generator_runs_explicit_crop_grids_and_maps_to_full_image`, replace:

```python
    assert all(proposal.segmentation.shape == (8, 8) for proposal in proposals)
```

with:

```python
    for proposal in proposals:
        x0, y0, x1, y1 = proposal.bbox
        assert proposal.segmentation.shape == (y1 - y0, x1 - x0)
        assert proposal.segmentation.sum() == proposal.area
        assert proposal.image_size == (8, 8)
        assert proposal_to_full_mask(proposal).shape == (8, 8)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: FAIL because generator-created proposals still store full-image
masks and do not set `image_size`.

- [ ] **Step 3: Store ROI masks in `_proposals_from_batch`**

In `src/auto_mask_generator.py`, replace:

```python
full_mask = np.zeros((full_height, full_width), dtype=bool)
full_mask[crop_y0:crop_y1, crop_x0:crop_x1] = mask
```

with:

```python
lx0, ly0, lx1, ly1 = local_bbox
roi_mask = mask[ly0:ly1, lx0:lx1].copy()
```

Then construct proposals with:

```python
MaskProposal(
    segmentation=roi_mask,
    bbox=bbox,
    area=area,
    predicted_iou=predicted_iou,
    stability_score=stability,
    point_coords=(
        float(point[0] + crop_x0),
        float(point[1] + crop_y0),
    ),
    crop_box=crop_box,
    crop_grid=crop_grid,
    crop_index=crop_index,
    image_size=(full_width, full_height),
)
```

Remove unused `crop_x1` and `crop_y1` locals if they are no longer needed.

- [ ] **Step 4: Run generator tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: PASS.

- [ ] **Step 5: Run all unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 6: Commit ROI generator output**

Run:

```powershell
git add src/auto_mask_generator.py tests/test_auto_mask_generator.py
git commit -m "feat: store generated masks as roi proposals"
```

### Task 3: ROI-Aware Visualization

**Files:**
- Modify: `tests/test_auto_mask_generator.py`
- Modify: `src/auto_mask_generator.py`

- [ ] **Step 1: Add visualization tests with ROI proposals**

Update imports in `tests/test_auto_mask_generator.py`:

```python
from src.auto_mask_generator import (
    ...
    save_proposal_grid,
    save_proposal_overlay,
)
```

Append:

```python
def test_save_proposal_overlay_accepts_roi_masks(tmp_path):
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    proposal = MaskProposal(
        segmentation=np.ones((2, 2), dtype=bool),
        bbox=(1, 1, 3, 3),
        area=4,
        predicted_iou=1.0,
        stability_score=1.0,
        point_coords=(2.0, 2.0),
        crop_box=(0, 0, 4, 4),
        image_size=(4, 4),
    )
    path = tmp_path / "overlay.png"

    save_proposal_overlay(image, [proposal], path)

    output = Image.open(path).convert("RGBA")
    assert output.size == (4, 4)
    assert output.getpixel((1, 1)) != (0, 0, 0, 255)
    assert output.getpixel((0, 0)) == (0, 0, 0, 255)


def test_save_proposal_grid_accepts_roi_masks(tmp_path):
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    proposal = MaskProposal(
        segmentation=np.ones((2, 2), dtype=bool),
        bbox=(1, 1, 3, 3),
        area=4,
        predicted_iou=1.0,
        stability_score=1.0,
        point_coords=(2.0, 2.0),
        crop_box=(0, 0, 4, 4),
        image_size=(4, 4),
    )
    path = tmp_path / "grid.png"

    save_proposal_grid(image, [proposal], path, max_masks=1, columns=1)

    output = Image.open(path)
    assert output.size[0] == 160
    assert output.size[1] == 160
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: FAIL because visualization helpers still create full-image layers and
try to use ROI alpha masks as full-image alpha masks.

- [ ] **Step 3: Add ROI paste helper and update overlay**

In `src/auto_mask_generator.py`, add:

```python
def _paste_proposal_overlay(
    overlay: Image.Image,
    proposal: MaskProposal,
    color: tuple[int, int, int, int],
    alpha: int,
) -> Image.Image:
    x0, y0, x1, y1 = _validate_roi_geometry(proposal)
    mask = proposal_mask_image(proposal, alpha=alpha)
    layer = Image.new("RGBA", (x1 - x0, y1 - y0), color)
    layer.putalpha(mask)
    overlay.alpha_composite(layer, dest=(x0, y0))
    return overlay
```

Update `save_proposal_overlay` loop:

```python
for index, proposal in enumerate(proposals[:max_masks]):
    overlay = _paste_proposal_overlay(
        overlay,
        proposal,
        _proposal_color(index),
        alpha=110,
    )
```

- [ ] **Step 4: Update grid visualization**

In `save_proposal_grid`, replace the full-image alpha layer block with:

```python
tile = image.convert("RGBA")
color = _proposal_color(index)
tile = _paste_proposal_overlay(tile, proposal, color, alpha=130)
draw = ImageDraw.Draw(tile)
draw.rectangle(proposal.bbox, outline=color[:3], width=3)
```

- [ ] **Step 5: Run visualization tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: PASS.

- [ ] **Step 6: Run all unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 7: Commit visualization update**

Run:

```powershell
git add src/auto_mask_generator.py tests/test_auto_mask_generator.py
git commit -m "feat: render roi mask proposals"
```

### Task 4: Final Verification And Push

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run all unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 2: Run existing prompt smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

Expected: PASS with `missing_keys: 0` and `unexpected_keys: 0`.

- [ ] **Step 3: Run default automatic mask smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py
```

Expected: PASS and write `outputs/auto_masks_overlay.png` and
`outputs/auto_masks_grid.png`.

- [ ] **Step 4: Run crop automatic mask smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 32 32 --max-masks 200
```

Expected: PASS and print proposal counts by crop grid.

- [ ] **Step 5: Confirm ignored artifacts are not staged**

Run:

```powershell
git status --short --ignored
```

Expected: `.venv/`, `outputs/`, `sam3-main/`, and `weight/` appear only as
ignored (`!!`) or are absent from tracked/untracked output.

- [ ] **Step 6: Push**

Run:

```powershell
git status --short
git push
```

Expected: clean tracked working tree and successful push to `origin/main`.

## Self-Review

Spec coverage:

- `MaskProposal.segmentation` as ROI mask: Task 2.
- Full-image `bbox` retained: Task 2.
- `image_size` metadata: Task 1 and Task 2.
- `proposal_to_full_mask`: Task 1.
- `proposal_mask_image`: Task 1.
- Visualization without full-image proposal storage: Task 3.
- Box NMS unchanged: Task 2 leaves `_remove_duplicates` on `bbox`.
- Real checkpoint smoke verification: Task 4.
- Artifact rules: Task 4.

Placeholder scan:

- No placeholder markers are used.
- Every test and command references concrete paths and symbols.

Type consistency:

- `image_size` consistently means `(width, height)`.
- `segmentation` consistently means bbox-local bool mask.
- `proposal_to_full_mask` returns full-image `(height, width)` bool arrays.
- `proposal_mask_image` returns bbox-local PIL `L` images.
