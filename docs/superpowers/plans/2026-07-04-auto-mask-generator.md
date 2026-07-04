# Automatic Mask Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a SAM-style automatic mask generator that extracts mask proposals from an image by batching grid point prompts through the existing `Sam3Predictor`.

**Architecture:** Add a focused `src/auto_mask_generator.py` module on top of the verified predictor path. Keep model loading and prompt execution inside `Sam3Predictor`; the generator owns only grid creation, candidate filtering, duplicate removal, proposal metadata, and visualization helpers.

**Tech Stack:** Python, NumPy, PIL, PyTorch-backed `Sam3Predictor`, pytest, local CUDA smoke test.

---

## File Structure

- Create: `src/auto_mask_generator.py`
  Contains `MaskProposal`, `Sam3AutomaticMaskGenerator`, grid utilities, bbox/stability/NMS helpers, and visualization helpers.
- Create: `tests/test_auto_mask_generator.py`
  CPU-only tests for utility functions and fake-predictor batching.
- Create: `scripts/auto_mask_smoke_test.py`
  Real checkpoint smoke test using `asset/sample.jpg`, writing ignored PNGs under `outputs/`.
- Modify: `src/__init__.py`
  Export `Sam3AutomaticMaskGenerator` and `MaskProposal`.

### Task 1: Utility Functions

**Files:**
- Create: `tests/test_auto_mask_generator.py`
- Create: `src/auto_mask_generator.py`

- [ ] **Step 1: Write failing utility tests**

Create `tests/test_auto_mask_generator.py` with:

```python
import numpy as np

from src.auto_mask_generator import (
    batched,
    box_area,
    box_iou,
    build_point_grid,
    calculate_stability_score,
    mask_to_box,
    nms_boxes,
)


def test_build_point_grid_centers_points_inside_unit_cells():
    grid = build_point_grid(2)

    assert grid.shape == (4, 2)
    np.testing.assert_allclose(
        grid,
        np.array(
            [
                [0.25, 0.25],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.75, 0.75],
            ],
            dtype=np.float32,
        ),
    )


def test_build_point_grid_rejects_invalid_size():
    try:
        build_point_grid(0)
    except ValueError as exc:
        assert "points_per_side" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_mask_to_box_returns_inclusive_exclusive_xyxy():
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:5, 3:7] = True

    assert mask_to_box(mask) == (3, 2, 7, 5)
    assert mask_to_box(np.zeros((3, 4), dtype=bool)) is None


def test_calculate_stability_score_uses_offset_thresholds():
    logits = np.array([[-2.0, -0.5, 0.5, 2.0]], dtype=np.float32)

    score = calculate_stability_score(logits, mask_threshold=0.0, offset=1.0)

    assert score == 0.5


def test_box_iou_and_nms_boxes_remove_lower_scoring_duplicate():
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [20, 20, 30, 30],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    assert box_area((0, 0, 10, 10)) == 100
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert nms_boxes(boxes, scores, iou_threshold=0.7) == [0, 2]


def test_batched_splits_sequence_without_dropping_items():
    chunks = list(batched(np.arange(5), 2))

    assert [chunk.tolist() for chunk in chunks] == [[0, 1], [2, 3], [4]]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.auto_mask_generator'`.

- [ ] **Step 3: Implement utility functions**

Create `src/auto_mask_generator.py` with the helper functions imported by the test. Use NumPy only for these utilities. `build_point_grid(2)` must return the exact order from the test: x changes fastest, y changes slowest.

- [ ] **Step 4: Run utility tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit utilities**

Run:

```powershell
git add src/auto_mask_generator.py tests/test_auto_mask_generator.py
git commit -m "feat: add auto mask generator utilities"
```

### Task 2: Generator Batching And Filtering

**Files:**
- Modify: `src/auto_mask_generator.py`
- Modify: `tests/test_auto_mask_generator.py`
- Modify: `src/__init__.py`

- [ ] **Step 1: Add failing generator tests**

Append to `tests/test_auto_mask_generator.py`:

```python
from PIL import Image

from src.auto_mask_generator import Sam3AutomaticMaskGenerator


class FakePredictor:
    def __init__(self):
        self.set_image_calls = 0
        self.predict_batches = []

    def set_image(self, image):
        self.set_image_calls += 1

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
        masks = np.zeros((batch, 1, 8, 8), dtype=bool)
        low_res = np.zeros((batch, 1, 8, 8), dtype=np.float32)
        scores = np.zeros((batch, 1), dtype=np.float32)
        for i in range(batch):
            x = min(int(point_coords[i, 0, 0] // 4), 6)
            y = min(int(point_coords[i, 0, 1] // 4), 6)
            masks[i, 0, y : y + 2, x : x + 2] = True
            low_res[i, 0] = np.where(masks[i, 0], 2.0, -2.0)
            scores[i, 0] = 1.0 - (i * 0.01)
        return masks, scores, low_res


def test_generator_batches_grid_points_and_returns_sorted_proposals():
    predictor = FakePredictor()
    generator = Sam3AutomaticMaskGenerator(
        predictor,
        points_per_side=2,
        points_per_batch=3,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        box_nms_thresh=0.0,
    )

    proposals = generator.generate(Image.new("RGB", (8, 8), color=(0, 0, 0)))

    assert predictor.set_image_calls == 1
    assert [batch[0].shape[0] for batch in predictor.predict_batches] == [3, 1]
    assert len(proposals) == 4
    assert proposals[0].predicted_iou >= proposals[-1].predicted_iou
    assert proposals[0].segmentation.shape == (8, 8)
    assert proposals[0].crop_box == (0, 0, 8, 8)


def test_generator_filters_by_score_stability_area_and_max_masks():
    predictor = FakePredictor()
    generator = Sam3AutomaticMaskGenerator(
        predictor,
        points_per_side=2,
        points_per_batch=4,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.5,
        min_mask_region_area=4,
        box_nms_thresh=1.0,
        max_masks=2,
    )

    proposals = generator.generate(np.zeros((8, 8, 3), dtype=np.uint8))

    assert len(proposals) == 2
    assert all(proposal.area >= 4 for proposal in proposals)
```

- [ ] **Step 2: Run generator tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: FAIL with missing `Sam3AutomaticMaskGenerator`.

- [ ] **Step 3: Implement generator classes**

Implement in `src/auto_mask_generator.py`:

- `MaskProposal` dataclass.
- `Sam3AutomaticMaskGenerator.__init__`.
- `Sam3AutomaticMaskGenerator.from_checkpoint`.
- `Sam3AutomaticMaskGenerator.generate`.

The generator must call `predictor.predict(point_coords=batch[:, None, :], point_labels=np.ones((batch_size, 1), dtype=np.int64), multimask_output=True)` and flatten `(B, M, H, W)` outputs.

- [ ] **Step 4: Export public classes**

Update `src/__init__.py`:

```python
from .auto_mask_generator import MaskProposal, Sam3AutomaticMaskGenerator
from .predictor import Sam3Predictor

__all__ = ["MaskProposal", "Sam3AutomaticMaskGenerator", "Sam3Predictor"]
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

- [ ] **Step 7: Commit generator**

Run:

```powershell
git add src/auto_mask_generator.py src/__init__.py tests/test_auto_mask_generator.py
git commit -m "feat: add automatic mask generator"
```

### Task 3: Smoke Script And Visualization

**Files:**
- Modify: `src/auto_mask_generator.py`
- Create: `scripts/auto_mask_smoke_test.py`

- [ ] **Step 1: Add visualization helpers**

Add functions to `src/auto_mask_generator.py`:

- `save_proposal_overlay(image, proposals, path, max_masks=50)`
- `save_proposal_grid(image, proposals, path, max_masks=24, columns=6)`

Use deterministic colors from proposal index. PNGs must be written through PIL.

- [ ] **Step 2: Create smoke script**

Create `scripts/auto_mask_smoke_test.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auto_mask_generator import (
    Sam3AutomaticMaskGenerator,
    save_proposal_grid,
    save_proposal_overlay,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    image_path = ROOT / "asset" / "sample.jpg"
    checkpoint_path = ROOT / "weight" / "sam3.1_multiplex.pt"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    generator = Sam3AutomaticMaskGenerator.from_checkpoint(
        checkpoint_path,
        device="cuda",
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.75,
        box_nms_thresh=0.7,
        max_masks=100,
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        proposals = generator.generate(image)

    overlay_path = output_dir / "auto_masks_overlay.png"
    grid_path = output_dir / "auto_masks_grid.png"
    save_proposal_overlay(image, proposals, overlay_path, max_masks=50)
    save_proposal_grid(image, proposals, grid_path, max_masks=24, columns=6)

    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"proposal_count: {len(proposals)}")
    for idx, proposal in enumerate(proposals[:10]):
        print(
            f"proposal[{idx}]: bbox={proposal.bbox} area={proposal.area} "
            f"iou={proposal.predicted_iou:.4f} stability={proposal.stability_score:.4f} "
            f"point={proposal.point_coords}"
        )
    print(f"overlay_path: {overlay_path}")
    print(f"grid_path: {grid_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 4: Run real smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py
```

Expected: command completes, prints `proposal_count`, and writes both PNG files under `outputs/`.

- [ ] **Step 5: Verify output files exist**

Run:

```powershell
Test-Path outputs\auto_masks_overlay.png
Test-Path outputs\auto_masks_grid.png
```

Expected: both print `True`.

- [ ] **Step 6: Commit smoke script**

Run:

```powershell
git add src/auto_mask_generator.py scripts/auto_mask_smoke_test.py
git commit -m "test: add automatic mask smoke test"
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

- [ ] **Step 2: Run prompt smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

Expected: PASS with `missing_keys: 0` and `unexpected_keys: 0`.

- [ ] **Step 3: Run automatic mask smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py
```

Expected: PASS with non-zero `proposal_count`.

- [ ] **Step 4: Verify forbidden paths are ignored**

Run:

```powershell
git status --short --ignored
```

Expected: `.venv/`, `outputs/`, `sam3-main/`, and `weight/` appear only as ignored (`!!`) or are absent from tracked/untracked output.

- [ ] **Step 5: Push**

Run:

```powershell
git status --short
git push
```

Expected: clean working tree and successful push to `origin/main`.

## Self-Review

Spec coverage:

- Balanced full-image grid: Task 2.
- Batched point prompts: Task 2.
- Score/stability/area filtering: Task 1 and Task 2.
- Box NMS: Task 1 and Task 2.
- Smoke visualization: Task 3.
- No Hugging Face/detector/tracking: generator wraps `Sam3Predictor` only.
- No committed outputs/weights/reference source: Task 4.

Placeholder scan:

- No placeholder markers are used.
- Code-bearing tasks include exact test code or exact implementation requirements.

Type consistency:

- `MaskProposal`, `Sam3AutomaticMaskGenerator`, `build_point_grid`,
  `calculate_stability_score`, `mask_to_box`, `nms_boxes`, `save_proposal_overlay`,
  and `save_proposal_grid` are consistently named across tests, scripts, and API.
