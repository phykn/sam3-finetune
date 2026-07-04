# Crop Pyramid Automatic Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit crop-grid automatic mask generation for higher small-object recall.

**Architecture:** Extend the existing `Sam3AutomaticMaskGenerator` rather than adding a second generator. The generator will build crop jobs from `crop_grids` and `crop_points_per_side`, run the existing single-image predictor per crop, batch point prompts within each crop, filter local candidates before expanding masks to the original image size, then apply per-crop and global NMS.

**Tech Stack:** Python, NumPy, PIL, PyTorch-backed `Sam3Predictor`, pytest, local CUDA smoke scripts.

---

## File Structure

- Modify: `src/auto_mask_generator.py`
  Adds crop job generation, crop image extraction, local-to-global proposal
  conversion, per-crop NMS, proposal crop metadata, and crop summary helpers.
- Modify: `tests/test_auto_mask_generator.py`
  Adds CPU-only tests for crop boxes, constructor validation, crop metadata,
  local-to-global mapping, edge filtering, and backward compatibility.
- Modify: `scripts/auto_mask_smoke_test.py`
  Adds CLI options for crop grids and prints timing plus proposal counts by
  crop grid.
- No changes: `sam3-main/`, `weight/`, `.venv/`, `outputs/`.

### Task 1: Crop Utilities And Proposal Metadata

**Files:**
- Modify: `tests/test_auto_mask_generator.py`
- Modify: `src/auto_mask_generator.py`

- [ ] **Step 1: Write failing crop utility tests**

Append these imports and tests to `tests/test_auto_mask_generator.py`:

```python
from src.auto_mask_generator import generate_crop_boxes


def test_generate_crop_boxes_full_image_grid():
    crops = generate_crop_boxes(100, 80, grid_size=1, overlap_ratio=0.25)

    assert crops == [(0, 0, 100, 80)]


def test_generate_crop_boxes_two_by_two_with_overlap_cover_edges():
    crops = generate_crop_boxes(100, 80, grid_size=2, overlap_ratio=0.25)

    assert len(crops) == 4
    assert crops[0] == (0, 0, 56, 45)
    assert crops[-1] == (44, 35, 100, 80)
    assert min(crop[0] for crop in crops) == 0
    assert min(crop[1] for crop in crops) == 0
    assert max(crop[2] for crop in crops) == 100
    assert max(crop[3] for crop in crops) == 80
    assert all(x0 < x1 and y0 < y1 for x0, y0, x1, y1 in crops)


def test_generate_crop_boxes_rejects_invalid_config():
    for grid_size in (0, -1):
        try:
            generate_crop_boxes(100, 80, grid_size=grid_size, overlap_ratio=0.25)
        except ValueError as exc:
            assert "grid_size" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    for overlap_ratio in (-0.1, 0.5):
        try:
            generate_crop_boxes(100, 80, grid_size=2, overlap_ratio=overlap_ratio)
        except ValueError as exc:
            assert "overlap_ratio" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


def test_mask_proposal_defaults_crop_metadata():
    proposal = MaskProposal(
        segmentation=np.zeros((4, 4), dtype=bool),
        bbox=(0, 0, 1, 1),
        area=1,
        predicted_iou=0.9,
        stability_score=1.0,
        point_coords=(0.5, 0.5),
        crop_box=(0, 0, 4, 4),
    )

    assert proposal.crop_grid == 1
    assert proposal.crop_index == 0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: FAIL with `ImportError` or `AttributeError` for
`generate_crop_boxes`, and then missing `crop_grid`/`crop_index` after the
import is added.

- [ ] **Step 3: Implement crop utilities and metadata**

In `src/auto_mask_generator.py`:

```python
from dataclasses import dataclass
from math import ceil
```

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
```

Add:

```python
def generate_crop_boxes(
    width: int,
    height: int,
    grid_size: int,
    overlap_ratio: float,
) -> list[tuple[int, int, int, int]]:
    if grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    if overlap_ratio < 0.0 or overlap_ratio >= 0.5:
        raise ValueError("overlap_ratio must be in [0.0, 0.5)")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if grid_size == 1:
        return [(0, 0, width, height)]

    overlap_w = int(round((width / grid_size) * overlap_ratio))
    overlap_h = int(round((height / grid_size) * overlap_ratio))
    crop_w = int(ceil((width + overlap_w * (grid_size - 1)) / grid_size))
    crop_h = int(ceil((height + overlap_h * (grid_size - 1)) / grid_size))
    stride_w = crop_w - overlap_w
    stride_h = crop_h - overlap_h

    boxes: list[tuple[int, int, int, int]] = []
    for iy in range(grid_size):
        y0 = min(iy * stride_h, height - crop_h)
        y1 = min(y0 + crop_h, height)
        for ix in range(grid_size):
            x0 = min(ix * stride_w, width - crop_w)
            x1 = min(x0 + crop_w, width)
            boxes.append((int(x0), int(y0), int(x1), int(y1)))
    return boxes
```

- [ ] **Step 4: Run crop utility tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: PASS for the new utility tests and existing tests.

- [ ] **Step 5: Commit crop utilities**

Run:

```powershell
git add src/auto_mask_generator.py tests/test_auto_mask_generator.py
git commit -m "feat: add crop box utilities"
```

### Task 2: Crop Scheduler And Generator Pipeline

**Files:**
- Modify: `tests/test_auto_mask_generator.py`
- Modify: `src/auto_mask_generator.py`

- [ ] **Step 1: Add failing constructor and crop pipeline tests**

Append to `tests/test_auto_mask_generator.py`:

```python
class CropAwareFakePredictor:
    def __init__(self):
        self.images = []
        self.predict_batches = []

    def set_image(self, image):
        self.images.append(image.size)

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        self.predict_batches.append((point_coords.copy(), point_labels.copy()))
        batch = point_coords.shape[0]
        crop_w, crop_h = self.images[-1]
        masks = np.zeros((batch, 1, crop_h, crop_w), dtype=bool)
        low_res = np.zeros((batch, 1, crop_h, crop_w), dtype=np.float32)
        scores = np.ones((batch, 1), dtype=np.float32)
        for i in range(batch):
            x = min(max(int(point_coords[i, 0, 0]), 0), crop_w - 1)
            y = min(max(int(point_coords[i, 0, 1]), 0), crop_h - 1)
            x0 = max(x - 1, 0)
            y0 = max(y - 1, 0)
            x1 = min(x0 + 2, crop_w)
            y1 = min(y0 + 2, crop_h)
            masks[i, 0, y0:y1, x0:x1] = True
            low_res[i, 0] = np.where(masks[i, 0], 2.0, -2.0)
            scores[i, 0] = 1.0 - (i * 0.01)
        return masks, scores, low_res


def test_generator_rejects_mismatched_crop_lists():
    try:
        Sam3AutomaticMaskGenerator(
            CropAwareFakePredictor(),
            crop_grids=[1, 2],
            crop_points_per_side=[4],
        )
    except ValueError as exc:
        assert "crop_grids" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_generator_runs_explicit_crop_grids_and_maps_to_full_image():
    predictor = CropAwareFakePredictor()
    generator = Sam3AutomaticMaskGenerator(
        predictor,
        points_per_batch=8,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        box_nms_thresh=1.0,
        crop_grids=[1, 2],
        crop_points_per_side=[1, 1],
        crop_overlap_ratio=0.0,
        filter_crop_edge_masks=False,
    )

    proposals = generator.generate(Image.new("RGB", (8, 8), color=(0, 0, 0)))

    assert predictor.images == [(8, 8), (4, 4), (4, 4), (4, 4), (4, 4)]
    assert len(proposals) == 5
    assert {proposal.crop_grid for proposal in proposals} == {1, 2}
    assert sorted(proposal.crop_index for proposal in proposals if proposal.crop_grid == 2) == [
        0,
        1,
        2,
        3,
    ]
    assert all(proposal.segmentation.shape == (8, 8) for proposal in proposals)
    assert any(proposal.crop_box == (4, 4, 8, 8) for proposal in proposals)
    assert all(0.0 <= proposal.point_coords[0] <= 8.0 for proposal in proposals)
    assert all(0.0 <= proposal.point_coords[1] <= 8.0 for proposal in proposals)


def test_generator_filters_internal_crop_edge_masks():
    predictor = CropAwareFakePredictor()
    generator = Sam3AutomaticMaskGenerator(
        predictor,
        points_per_batch=8,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        box_nms_thresh=1.0,
        crop_grids=[2],
        crop_points_per_side=[1],
        crop_overlap_ratio=0.0,
        filter_crop_edge_masks=True,
    )

    proposals = generator.generate(Image.new("RGB", (8, 8), color=(0, 0, 0)))

    assert proposals == []
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: FAIL because `Sam3AutomaticMaskGenerator.__init__` does not accept
`crop_grids`, `crop_points_per_side`, `crop_overlap_ratio`, or
`filter_crop_edge_masks`.

- [ ] **Step 3: Implement crop scheduler in generator**

In `src/auto_mask_generator.py`, update `Sam3AutomaticMaskGenerator.__init__`:

```python
def __init__(
    self,
    predictor,
    points_per_side: int = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.0,
    stability_score_thresh: float = 0.75,
    stability_score_offset: float = 1.0,
    min_mask_region_area: int = 0,
    box_nms_thresh: float = 0.7,
    max_masks: int | None = None,
    crop_grids: Sequence[int] | None = None,
    crop_points_per_side: Sequence[int] | None = None,
    crop_overlap_ratio: float = 0.25,
    crop_nms_thresh: float | None = None,
    max_masks_per_crop: int | None = None,
    filter_crop_edge_masks: bool = True,
) -> None:
```

Validation behavior:

```python
if crop_grids is None and crop_points_per_side is not None:
    raise ValueError("crop_points_per_side requires crop_grids")
if crop_grids is not None:
    if crop_points_per_side is None or len(crop_grids) != len(crop_points_per_side):
        raise ValueError("crop_grids and crop_points_per_side must have the same length")
    if any(grid <= 0 for grid in crop_grids):
        raise ValueError("crop_grids entries must be positive integers")
    if any(points <= 0 for points in crop_points_per_side):
        raise ValueError("crop_points_per_side entries must be positive integers")
if crop_overlap_ratio < 0.0 or crop_overlap_ratio >= 0.5:
    raise ValueError("crop_overlap_ratio must be in [0.0, 0.5)")
```

Store:

```python
self.crop_grids = tuple(crop_grids) if crop_grids is not None else None
self.crop_points_per_side = (
    tuple(crop_points_per_side) if crop_points_per_side is not None else None
)
self.crop_overlap_ratio = crop_overlap_ratio
self.crop_nms_thresh = box_nms_thresh if crop_nms_thresh is None else crop_nms_thresh
self.max_masks_per_crop = max_masks_per_crop
self.filter_crop_edge_masks = filter_crop_edge_masks
```

Refactor `generate(image)` so it calls a helper:

```python
for crop_grid, points_per_side in self._crop_grid_config():
    normalized_grid = point_grid_cache.setdefault(
        points_per_side,
        build_point_grid(points_per_side),
    )
    for crop_index, crop_box in enumerate(
        generate_crop_boxes(width, height, crop_grid, self.crop_overlap_ratio)
    ):
        crop_image = _crop_image(image, crop_box)
        proposals.extend(
            self._generate_for_crop(
                crop_image,
                crop_box,
                crop_grid,
                crop_index,
                normalized_grid,
                (width, height),
            )
        )
```

Add helpers:

```python
def _crop_grid_config(self) -> list[tuple[int, int]]:
    if self.crop_grids is None:
        return [(1, self.points_per_side)]
    assert self.crop_points_per_side is not None
    return list(zip(self.crop_grids, self.crop_points_per_side))


def _crop_image(
    image: Image.Image | np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> Image.Image | np.ndarray:
    x0, y0, x1, y1 = crop_box
    if isinstance(image, Image.Image):
        return image.crop(crop_box)
    return image[y0:y1, x0:x1, :]
```

- [ ] **Step 4: Implement crop-local filtering and global conversion**

Add a crop-aware proposal path in `src/auto_mask_generator.py`:

```python
def _generate_for_crop(
    self,
    crop_image: Image.Image | np.ndarray,
    crop_box: tuple[int, int, int, int],
    crop_grid: int,
    crop_index: int,
    normalized_grid: np.ndarray,
    full_size: tuple[int, int],
) -> list[MaskProposal]:
```

Inside it:

```python
crop_width, crop_height = _image_size(crop_image)
self.predictor.set_image(crop_image)
pixel_grid = normalized_grid.copy()
pixel_grid[:, 0] *= float(crop_width)
pixel_grid[:, 1] *= float(crop_height)
proposals = []
for point_batch in batched(pixel_grid, self.points_per_batch):
    point_labels = np.ones((len(point_batch), 1), dtype=np.int64)
    masks, scores, low_res_masks = self.predictor.predict(
        point_coords=point_batch[:, None, :].astype(np.float32),
        point_labels=point_labels,
        multimask_output=True,
    )
    proposals.extend(
        self._proposals_from_batch(
            point_batch,
            masks,
            scores,
            low_res_masks,
            crop_box,
            crop_grid=crop_grid,
            crop_index=crop_index,
            full_size=full_size,
        )
    )
proposals = self._remove_duplicates(proposals, self.crop_nms_thresh)
if self.max_masks_per_crop is not None:
    proposals = proposals[: self.max_masks_per_crop]
return proposals
```

Update `_proposals_from_batch(...)` to accept `crop_grid`, `crop_index`, and
`full_size`. It should:

- compute area, bbox, and edge filtering from the crop-local mask,
- skip internal-edge masks when `filter_crop_edge_masks` is true,
- convert bbox and point to full-image coordinates,
- expand only surviving crop masks into full-image masks.

Use:

```python
def _touches_internal_crop_edge(local_bbox, crop_box, full_size):
    x0, y0, x1, y1 = local_bbox
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
    full_width, full_height = full_size
    touches_left = x0 <= 0 and crop_x0 > 0
    touches_top = y0 <= 0 and crop_y0 > 0
    touches_right = x1 >= crop_x1 - crop_x0 and crop_x1 < full_width
    touches_bottom = y1 >= crop_y1 - crop_y0 and crop_y1 < full_height
    return touches_left or touches_top or touches_right or touches_bottom
```

Update `_remove_duplicates` to accept an optional threshold:

```python
def _remove_duplicates(
    self,
    proposals: list[MaskProposal],
    iou_threshold: float | None = None,
) -> list[MaskProposal]:
    threshold = self.box_nms_thresh if iou_threshold is None else iou_threshold
```

- [ ] **Step 5: Run generator tests**

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

- [ ] **Step 7: Commit crop pipeline**

Run:

```powershell
git add src/auto_mask_generator.py tests/test_auto_mask_generator.py
git commit -m "feat: add crop pyramid mask generation"
```

### Task 3: Smoke CLI And Crop Statistics

**Files:**
- Modify: `scripts/auto_mask_smoke_test.py`
- Modify: `tests/test_auto_mask_generator.py`
- Modify: `src/auto_mask_generator.py`

- [ ] **Step 1: Add summary helper tests**

Append to `tests/test_auto_mask_generator.py`:

```python
from src.auto_mask_generator import count_proposals_by_crop_grid


def test_count_proposals_by_crop_grid():
    proposals = [
        MaskProposal(
            segmentation=np.zeros((2, 2), dtype=bool),
            bbox=(0, 0, 1, 1),
            area=1,
            predicted_iou=1.0,
            stability_score=1.0,
            point_coords=(0.5, 0.5),
            crop_box=(0, 0, 2, 2),
            crop_grid=1,
        ),
        MaskProposal(
            segmentation=np.zeros((2, 2), dtype=bool),
            bbox=(0, 0, 1, 1),
            area=1,
            predicted_iou=1.0,
            stability_score=1.0,
            point_coords=(0.5, 0.5),
            crop_box=(0, 0, 1, 1),
            crop_grid=2,
        ),
        MaskProposal(
            segmentation=np.zeros((2, 2), dtype=bool),
            bbox=(1, 1, 2, 2),
            area=1,
            predicted_iou=1.0,
            stability_score=1.0,
            point_coords=(1.5, 1.5),
            crop_box=(1, 1, 2, 2),
            crop_grid=2,
        ),
    ]

    assert count_proposals_by_crop_grid(proposals) == {1: 1, 2: 2}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: FAIL because `count_proposals_by_crop_grid` is not defined.

- [ ] **Step 3: Implement summary helper**

Add to `src/auto_mask_generator.py`:

```python
def count_proposals_by_crop_grid(
    proposals: Sequence[MaskProposal],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for proposal in proposals:
        counts[proposal.crop_grid] = counts.get(proposal.crop_grid, 0) + 1
    return dict(sorted(counts.items()))
```

- [ ] **Step 4: Update smoke script CLI**

Replace `scripts/auto_mask_smoke_test.py` with argparse-based configuration:

```python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auto_mask_generator import (
    Sam3AutomaticMaskGenerator,
    count_proposals_by_crop_grid,
    save_proposal_grid,
    save_proposal_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-grids", nargs="*", type=int, default=None)
    parser.add_argument("--crop-points-per-side", nargs="*", type=int, default=None)
    parser.add_argument("--crop-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--max-masks", type=int, default=100)
    parser.add_argument("--max-masks-per-crop", type=int, default=None)
    parser.add_argument("--keep-crop-edge-masks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    image_path = ROOT / "asset" / "sample.jpg"
    checkpoint_path = ROOT / "weight" / "sam3.1_multiplex.pt"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    crop_grids = args.crop_grids if args.crop_grids else None
    crop_points_per_side = (
        args.crop_points_per_side if args.crop_points_per_side else None
    )

    image = Image.open(image_path).convert("RGB")
    generator = Sam3AutomaticMaskGenerator.from_checkpoint(
        checkpoint_path,
        device="cuda",
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.75,
        box_nms_thresh=0.7,
        max_masks=args.max_masks,
        crop_grids=crop_grids,
        crop_points_per_side=crop_points_per_side,
        crop_overlap_ratio=args.crop_overlap_ratio,
        max_masks_per_crop=args.max_masks_per_crop,
        filter_crop_edge_masks=not args.keep_crop_edge_masks,
    )
    started_at = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        proposals = generator.generate(image)
    elapsed = time.perf_counter() - started_at

    overlay_path = output_dir / "auto_masks_overlay.png"
    grid_path = output_dir / "auto_masks_grid.png"
    save_proposal_overlay(image, proposals, overlay_path, max_masks=50)
    save_proposal_grid(image, proposals, grid_path, max_masks=24, columns=6)

    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"crop_grids: {crop_grids}")
    print(f"crop_points_per_side: {crop_points_per_side}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"proposal_count: {len(proposals)}")
    print(f"proposal_count_by_crop_grid: {count_proposals_by_crop_grid(proposals)}")
    for idx, proposal in enumerate(proposals[:10]):
        print(
            f"proposal[{idx}]: bbox={proposal.bbox} area={proposal.area} "
            f"iou={proposal.predicted_iou:.4f} "
            f"stability={proposal.stability_score:.4f} "
            f"point={proposal.point_coords} crop_grid={proposal.crop_grid} "
            f"crop_index={proposal.crop_index} crop_box={proposal.crop_box}"
        )
    print(f"overlay_path: {overlay_path}")
    print(f"grid_path: {grid_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run all unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 6: Run quick crop smoke**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 16 16 --max-masks 100
```

Expected: command completes, prints elapsed time, nonzero proposal count, and
counts by crop grid.

- [ ] **Step 7: Run recall-oriented crop smoke**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 32 32 --max-masks 200
```

Expected: command completes, writes `outputs/auto_masks_overlay.png` and
`outputs/auto_masks_grid.png`, and prints proposal metadata with crop grid
fields.

- [ ] **Step 8: Commit smoke CLI**

Run:

```powershell
git add src/auto_mask_generator.py tests/test_auto_mask_generator.py scripts/auto_mask_smoke_test.py
git commit -m "test: add crop pyramid smoke options"
```

### Task 4: Final Verification And Push

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run full unit tests**

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

Expected: PASS and write PNGs under `outputs/`.

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

- Explicit `crop_grids` and `crop_points_per_side`: Task 2.
- Full image, 2x2, and 4x4 crop box generation: Task 1.
- Crop-local filtering before full-size mask expansion: Task 2.
- Per-crop NMS and global NMS: Task 2.
- Crop metadata for smoke summaries: Task 1 and Task 3.
- CLI smoke options and timing: Task 3.
- Real checkpoint verification: Task 4.
- Artifact rules for `sam3-main/`, `weight/`, `.venv/`, and `outputs/`: Task 4.

Placeholder scan:

- No placeholder markers are used.
- Every test and command references concrete files and symbols.

Type consistency:

- `crop_grids`, `crop_points_per_side`, `crop_overlap_ratio`,
  `filter_crop_edge_masks`, `crop_grid`, and `crop_index` match the spec.
- `full_size` consistently means `(width, height)`.
- Bboxes remain inclusive-exclusive `(x0, y0, x1, y1)`.
