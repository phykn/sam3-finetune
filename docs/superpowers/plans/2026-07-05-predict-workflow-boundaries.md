# Predict Workflow Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `src/predict` so the five intended workflows live under explicit packages: `prompted`, `grid`, `refine`, `context`, and `next_frame`.

**Architecture:** Move existing behavior into workflow-named packages without changing inference math. Remove the old `src.predict.image`, `src.predict.masks`, `src.predict.reference`, and `src.predict.video` public paths, add a dedicated `MaskRefiner`, and migrate tests/scripts to explicit workflow imports.

**Tech Stack:** Python 3.11, PyTorch, NumPy, PIL, pytest. Use `git mv` for tracked file movement and `apply_patch` for code edits.

---

## Current Dirty Worktree Constraint

Before implementation, run:

```bash
git status --short
```

Expected existing unrelated changes:

```text
 M src/predict/image_transform.py
 M src/types.py
 M tests/test_mask_instances.py
 M tests/test_transforms.py
```

Do not revert these files. If a task must touch one of them, read the current
content first and preserve the existing user changes. Stage only files changed
for the current task.

## Target File Structure

Create or move these files:

```text
src/predict/
  __init__.py
  prompted/
    __init__.py
    predictor.py
    transforms.py
  grid/
    __init__.py
    generator.py
    geometry.py
    instances.py
    proposals.py
  refine/
    __init__.py
    masks.py
  context/
    __init__.py
    guided.py
    matcher.py
    postprocess.py
    prototype.py
    scoring.py
  next_frame/
    __init__.py
    predictor.py
  grounding/
    __init__.py
    inference.py
    postprocess.py
```

Keep `src/predict/grounding/` unchanged. It is an existing orthogonal prediction
surface, not one of the five renamed workflows.

Remove these old workflow paths:

```text
src/predict/image.py
src/predict/image_transform.py
src/predict/masks/
src/predict/reference/
src/predict/video.py
```

`src/predict/__init__.py` and `src/__init__.py` must stop re-exporting workflow
classes. Import workflow classes from the workflow package directly.

## Task 1: Public Surface Tests For New Workflow Boundaries

**Files:**
- Modify: `tests/test_predictor_api.py`
- Modify: `tests/test_package_structure.py`

- [ ] **Step 1: Update `tests/test_predictor_api.py` imports and public-surface test**

Replace the import block:

```python
from src.predict import Sam3ImageEmbedding, Sam3Predictor, Sam3PromptBatch
```

with:

```python
import importlib

import pytest
from src.predict.prompted import Sam3Predictor
from src.types import Sam3ImageEmbedding, Sam3PromptBatch
```

Replace `test_package_public_surface_exposes_only_main_predictor` with:

```python
def test_package_public_surface_requires_workflow_imports():
    import src
    import src.predict as predict_root
    import src.predict.prompted as prompted
    import src.predict.prompted.predictor as predictor_module

    assert prompted.Sam3Predictor is Sam3Predictor
    assert not hasattr(src, "Sam3Predictor")
    assert not hasattr(src, "Sam3ImageEmbedding")
    assert not hasattr(src, "Sam3PromptBatch")
    assert not hasattr(predict_root, "Sam3Predictor")
    assert not hasattr(predict_root, "Sam3ImageEmbedding")
    assert not hasattr(predict_root, "Sam3PromptBatch")
    assert not hasattr(predictor_module, "Sam3ImageEmbedding")
    assert not hasattr(predictor_module, "Sam3PromptBatch")
    assert not hasattr(src, "__all__")
    assert not hasattr(predict_root, "__all__")
    for name in (
        "Sam3PromptBatch",
        "AutomaticMaskGenerator",
        "ContextMatcher",
        "NextFramePredictor",
        "VideoMemoryInference",
        "GroundingInference",
        "VisualLanguageCache",
        "build_model",
        "filter_grounding_prediction",
    ):
        assert not hasattr(src, name)

    for old_module in (
        "src.predict.image",
        "src.predict.masks",
        "src.predict.reference",
        "src.predict.video",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old_module)
```

- [ ] **Step 2: Update `tests/test_package_structure.py` expected paths**

In the existing required-path tuple, replace:

```python
"src/predict/image.py",
"src/predict/masks/instances.py",
"src/predict/masks/generator.py",
"src/predict/reference/matcher.py",
"src/predict/video.py",
"src/predict/image_transform.py",
```

with:

```python
"src/predict/prompted/predictor.py",
"src/predict/prompted/transforms.py",
"src/predict/grid/instances.py",
"src/predict/grid/generator.py",
"src/predict/context/matcher.py",
"src/predict/refine/masks.py",
"src/predict/next_frame/predictor.py",
```

At the end of the old-path assertions, add:

```python
    assert not (root / "src" / "predict" / "image.py").exists()
    assert not (root / "src" / "predict" / "image_transform.py").exists()
    assert not (root / "src" / "predict" / "video.py").exists()
    assert not (root / "src" / "predict" / "masks").exists()
    assert not (root / "src" / "predict" / "reference").exists()
```

Keep the existing assertion that root-level `src/masks`, `src/context`, and
`src/video` directories do not exist.

- [ ] **Step 3: Run the new boundary tests and verify they fail**

Run:

```bash
python -m pytest tests/test_predictor_api.py::test_package_public_surface_requires_workflow_imports tests/test_package_structure.py::test_shared_modules_are_grouped_by_responsibility -q
```

Expected: FAIL because `src.predict.prompted` and the new workflow paths do not
exist yet, while old paths still exist.

- [ ] **Step 4: Commit the failing tests**

Stage only the two test files:

```bash
git add tests/test_predictor_api.py tests/test_package_structure.py
git commit -m "test: define predict workflow package boundaries"
```

## Task 2: Move Prompted Prediction And Shared Transforms

**Files:**
- Move: `src/predict/image.py` to `src/predict/prompted/predictor.py`
- Move: `src/predict/image_transform.py` to `src/predict/prompted/transforms.py`
- Create: `src/predict/prompted/__init__.py`
- Modify: `src/__init__.py`
- Modify: `src/predict/__init__.py`
- Modify: `tests/test_transforms.py`
- Modify: `tests/test_predictor_api.py`
- Modify: `scripts/smoke_test.py`
- Modify: `scripts/video_memory_reference.py`

- [ ] **Step 1: Move files**

Run:

```bash
mkdir src\predict\prompted
git mv src\predict\image.py src\predict\prompted\predictor.py
git mv src\predict\image_transform.py src\predict\prompted\transforms.py
```

- [ ] **Step 2: Fix imports inside `src/predict/prompted/predictor.py`**

Change:

```python
from .. import types as api_types
from ..model.build import build_model
from .image_transform import ImageTransforms
```

to:

```python
from ... import types as api_types
from ...model.build import build_model
from .transforms import ImageTransforms
```

- [ ] **Step 3: Create `src/predict/prompted/__init__.py`**

Use:

```python
from ...types import (
    Sam3ImageEmbedding as Sam3ImageEmbedding,
    Sam3PromptBatch as Sam3PromptBatch,
)
from .predictor import Sam3Predictor as Sam3Predictor
```

- [ ] **Step 4: Remove root re-exports**

Replace `src/__init__.py` content with an empty file.

Replace `src/predict/__init__.py` content with an empty file.

- [ ] **Step 5: Update prompted imports in tests and scripts**

In `tests/test_predictor_api.py`, keep the imports added in Task 1:

```python
from src.predict.prompted import Sam3Predictor
from src.types import Sam3ImageEmbedding, Sam3PromptBatch
```

In `tests/test_transforms.py`, replace:

```python
from src.predict.image_transform import (
```

with:

```python
from src.predict.prompted.transforms import (
```

In `scripts/smoke_test.py`, replace:

```python
from src.predict import Sam3Predictor
```

with:

```python
from src.predict.prompted import Sam3Predictor
```

In `scripts/video_memory_reference.py`, replace both local imports:

```python
from src.predict import Sam3Predictor
```

with:

```python
from src.predict.prompted import Sam3Predictor
```

- [ ] **Step 6: Run prompted tests**

Run:

```bash
python -m pytest tests/test_predictor_api.py tests/test_transforms.py -k "not package_public_surface" -q
```

Expected: PASS for prompted prediction and transforms tests. The public-surface
boundary test remains red until the grid, context, and next-frame legacy paths
are removed in later tasks. If
`tests/test_transforms.py` fails because it contains pre-existing user edits,
read the failure and preserve the user-edited behavior while changing only the
import path or moved transform module.

- [ ] **Step 7: Commit prompted move**

Stage only prompted-related files:

```bash
git add src/__init__.py src/predict/__init__.py src/predict/prompted tests/test_predictor_api.py tests/test_transforms.py scripts/smoke_test.py scripts/video_memory_reference.py
git add -u src/predict/image.py src/predict/image_transform.py
git commit -m "refactor: move prompted prediction package"
```

## Task 3: Move Grid Prediction Package

**Files:**
- Move: `src/predict/masks/` to `src/predict/grid/`
- Modify: `src/predict/grid/generator.py`
- Modify: `tests/test_auto_mask_generator.py`
- Modify: `tests/test_mask_instances.py`
- Modify: `scripts/auto_mask_smoke_test.py`
- Modify: `scripts/profile_auto_mask.py`

- [ ] **Step 1: Move package**

Run:

```bash
git mv src\predict\masks src\predict\grid
```

- [ ] **Step 2: Fix `src/predict/grid/generator.py` prompted import**

Change:

```python
from ..image import Sam3Predictor
```

to:

```python
from ..prompted import Sam3Predictor
```

- [ ] **Step 3: Update grid imports in tests and scripts**

In `tests/test_auto_mask_generator.py`, replace:

```python
from src.predict.masks.generator import AutomaticMaskGenerator
from src.predict.masks.geometry import (
from src.predict.masks.proposals import (
```

with:

```python
from src.predict.grid.generator import AutomaticMaskGenerator
from src.predict.grid.geometry import (
from src.predict.grid.proposals import (
```

Change structural assertions in the same test:

```python
assert (root / "src" / "predict" / "grid" / "generator.py").is_file()
assert not (root / "src" / "predict" / "masks").exists()
assert AutomaticMaskGenerator.__module__ == "src.predict.grid.generator"
```

Change helper module assertions:

```python
assert build_point_grid.__module__ == "src.predict.grid.geometry"
```

Change package import:

```python
import src.predict.grid as grid

assert grid.AutomaticMaskGenerator is AutomaticMaskGenerator
assert grid.MaskInstance is MaskInstance
assert grid.MaskProposal is MaskProposal
assert hasattr(grid, "ReferenceExample")
assert hasattr(grid, "mask_instance_from_proposal")
assert hasattr(grid, "mask_instances_from_proposals")
assert not hasattr(grid, "__all__")
```

In `tests/test_mask_instances.py`, replace:

```python
from src.predict.masks import (
```

with:

```python
from src.predict.grid import (
```

In `scripts/auto_mask_smoke_test.py`, replace `src.predict.masks` imports with
`src.predict.grid`.

In `scripts/profile_auto_mask.py`, replace:

```python
from src.predict.masks.generator import AutomaticMaskGenerator
from src.predict.masks.geometry import (
from src.predict.masks.proposals import count_proposals_by_crop_grid
```

with:

```python
from src.predict.grid.generator import AutomaticMaskGenerator
from src.predict.grid.geometry import (
from src.predict.grid.proposals import count_proposals_by_crop_grid
```

- [ ] **Step 4: Run grid tests**

Run:

```bash
python -m pytest tests/test_auto_mask_generator.py tests/test_mask_instances.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit grid move**

Run:

```bash
git add src/predict/grid tests/test_auto_mask_generator.py tests/test_mask_instances.py scripts/auto_mask_smoke_test.py scripts/profile_auto_mask.py
git add -u src/predict/masks
git commit -m "refactor: move grid prediction package"
```

## Task 4: Move Context Prediction Package

**Files:**
- Move: `src/predict/reference/` to `src/predict/context/`
- Modify: `src/predict/context/guided.py`
- Modify: `src/predict/context/matcher.py`
- Modify: `src/predict/context/prototype.py`
- Modify: `tests/test_context_predictor.py`
- Modify: `tests/test_reference_guided_masks.py`
- Modify: `scripts/context_prompt_smoke_test.py`

- [ ] **Step 1: Move package**

Run:

```bash
git mv src\predict\reference src\predict\context
```

- [ ] **Step 2: Fix imports in context modules**

In `src/predict/context/guided.py`, change:

```python
from ..image import Sam3Predictor
```

to:

```python
from ..prompted import Sam3Predictor
```

In `src/predict/context/matcher.py`, change:

```python
from ..image import Sam3Predictor
from ..masks.geometry import calculate_stability_score, mask_to_box
```

to:

```python
from ..grid.geometry import calculate_stability_score, mask_to_box
from ..prompted import Sam3Predictor
```

In `src/predict/context/prototype.py`, change:

```python
from ..masks.geometry import mask_to_box
```

to:

```python
from ..grid.geometry import mask_to_box
```

- [ ] **Step 3: Update context imports in tests and scripts**

In `tests/test_context_predictor.py`, replace every
`src.predict.reference` import with `src.predict.context`.

Update structural assertions:

```python
assert (root / "src" / "predict" / "context" / "matcher.py").is_file()
assert (root / "src" / "predict" / "context" / "postprocess.py").is_file()
assert (root / "src" / "predict" / "context" / "prototype.py").is_file()
assert (root / "src" / "predict" / "context" / "scoring.py").is_file()
assert not (root / "src" / "predict" / "reference").exists()
assert ContextMatcher.__module__ == "src.predict.context.matcher"
assert (
    context_prediction_to_full_mask.__module__
    == "src.predict.context.postprocess"
)
assert area_ratio_score.__module__ == "src.predict.context.scoring"
```

Change the package import test to:

```python
import src.predict.context as context
from src.predict.context.guided import ReferenceGuidedMaskGenerator
from src.predict.context.matcher import ContextMatcher
```

In `tests/test_reference_guided_masks.py`, replace every
`src.predict.reference.guided` import with `src.predict.context.guided`.

In `scripts/context_prompt_smoke_test.py`, replace:

```python
from src.predict.reference.matcher import ContextMatcher
from src.predict.reference.postprocess import context_prediction_to_full_mask
```

with:

```python
from src.predict.context.matcher import ContextMatcher
from src.predict.context.postprocess import context_prediction_to_full_mask
```

- [ ] **Step 4: Run context tests**

Run:

```bash
python -m pytest tests/test_context_predictor.py tests/test_reference_guided_masks.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit context move**

Run:

```bash
git add src/predict/context tests/test_context_predictor.py tests/test_reference_guided_masks.py scripts/context_prompt_smoke_test.py
git add -u src/predict/reference
git commit -m "refactor: move context prediction package"
```

## Task 5: Add Explicit Refinement Workflow

**Files:**
- Create: `src/predict/refine/__init__.py`
- Create: `src/predict/refine/masks.py`
- Modify: `scripts/video_memory_reference.py`
- Modify: `tests/test_video_memory_reference.py`
- Modify: `tests/test_context_predictor.py`

- [ ] **Step 1: Add failing refinement tests**

In `tests/test_video_memory_reference.py`, add imports:

```python
from src.predict.refine import MaskRefiner, select_best_mask
```

Remove `select_best_mask` from the `scripts.video_memory_reference` import list.

Add this test:

```python
def test_mask_refiner_preserves_prompts_during_second_pass() -> None:
    class FakePredictor:
        def __init__(self) -> None:
            self.calls = []

        def set_image(self, image):
            self.image = image

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            masks = np.zeros((1, 1, 4, 6), dtype=bool)
            masks[0, 0, 1:3, 2:5] = True
            scores = np.array([[0.8]], dtype=np.float32)
            low_res = np.zeros((1, 1, 2, 2), dtype=np.float32)
            return masks, scores, low_res

    fake = FakePredictor()
    refiner = MaskRefiner(fake)
    image = Image.new("RGB", (6, 4), color=(0, 0, 0))
    low_res = np.ones((2, 2), dtype=np.float32)

    result = refiner.refine(
        image=image,
        point_coords=np.array([[3, 2]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int64),
        box=np.array([1, 1, 5, 3], dtype=np.float32),
        mask_input=low_res,
    )

    np.testing.assert_array_equal(
        fake.calls[0]["point_coords"],
        np.array([[3, 2]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        fake.calls[0]["point_labels"],
        np.array([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        fake.calls[0]["box"],
        np.array([1, 1, 5, 3], dtype=np.float32),
    )
    assert fake.calls[0]["mask_input"].shape == (2, 2)
    assert fake.calls[0]["multimask_output"] is False
    assert result.mask.shape == (4, 6)
    assert result.score == np.float32(0.8)
```

- [ ] **Step 2: Run refinement test and verify it fails**

Run:

```bash
python -m pytest tests/test_video_memory_reference.py::test_mask_refiner_preserves_prompts_during_second_pass -q
```

Expected: FAIL because `src.predict.refine` does not exist.

- [ ] **Step 3: Create `src/predict/refine/masks.py`**

Use:

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..prompted import Sam3Predictor


@dataclass(frozen=True)
class RefinedMaskResult:
    mask: np.ndarray
    score: float
    low_res_mask: np.ndarray
    selected_index: int


def select_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    mask_arr = np.asarray(masks)
    score_arr = np.asarray(scores, dtype=np.float32)
    if mask_arr.ndim < 3:
        raise ValueError("masks must have candidate and spatial dimensions")

    candidate_count = int(np.prod(mask_arr.shape[:-2]))
    flat_scores = score_arr.reshape(-1)
    if flat_scores.size != candidate_count:
        raise ValueError(
            f"score count {flat_scores.size} does not match mask count {candidate_count}"
        )

    flat_masks = mask_arr.reshape(candidate_count, *mask_arr.shape[-2:])
    selected_index = int(np.argmax(flat_scores))
    return (
        flat_masks[selected_index].astype(bool),
        float(flat_scores[selected_index]),
        selected_index,
    )


class MaskRefiner:
    def __init__(self, predictor) -> None:
        self.predictor = predictor

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cuda",
    ) -> "MaskRefiner":
        return cls(Sam3Predictor.from_checkpoint(path, device=device))

    def refine(
        self,
        *,
        image=None,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | torch.Tensor,
    ) -> RefinedMaskResult:
        if mask_input is None:
            raise ValueError("mask_input is required for refinement")
        if image is not None:
            self.predictor.set_image(image)
        masks, scores, low_res_masks = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=False,
        )
        mask, score, selected_index = select_best_mask(masks, scores)
        flat_low_res = np.asarray(low_res_masks).reshape(
            -1,
            *np.asarray(low_res_masks).shape[-2:],
        )
        return RefinedMaskResult(
            mask=mask,
            score=score,
            low_res_mask=flat_low_res[selected_index],
            selected_index=selected_index,
        )
```

- [ ] **Step 4: Create `src/predict/refine/__init__.py`**

Use:

```python
from .masks import MaskRefiner as MaskRefiner
from .masks import RefinedMaskResult as RefinedMaskResult
from .masks import select_best_mask as select_best_mask
```

- [ ] **Step 5: Use refinement helper in `scripts/video_memory_reference.py`**

At the top, add:

```python
from src.predict.refine import MaskRefiner, select_best_mask
```

Delete the local `select_best_mask` function from the script.

In `predict_sam_mask_from_prompts`, replace:

```python
            refined_masks, refined_scores, _refined_low_res = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_array,
                mask_input=low_res,
                multimask_output=False,
            )
            refined_mask, refined_score, _refined_index = select_best_mask(
                refined_masks,
                refined_scores,
            )
```

with:

```python
            refined = MaskRefiner(predictor).refine(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_array,
                mask_input=low_res,
            )
            refined_mask = refined.mask
            refined_score = refined.score
```

In `predict_sam_mask_from_box`, replace the analogous second-pass block with:

```python
            refined = MaskRefiner(predictor).refine(
                box=box_array,
                mask_input=low_res,
            )
            refined_mask = refined.mask
            refined_score = refined.score
```

- [ ] **Step 6: Update monkeypatch imports that used root `src.predict`**

In `tests/test_video_memory_reference.py`, replace:

```python
import src.predict
```

with:

```python
import src.predict.prompted as prompted
```

Replace:

```python
monkeypatch.setattr(src.predict, "Sam3Predictor", FakePredictor)
```

with:

```python
monkeypatch.setattr(prompted.Sam3Predictor, "from_checkpoint", FakePredictor.from_checkpoint)
```

In `tests/test_context_predictor.py`, replace the local import:

```python
from src.predict import Sam3Predictor
```

with:

```python
from src.predict.prompted import Sam3Predictor
```

- [ ] **Step 7: Run refinement/script tests**

Run:

```bash
python -m pytest tests/test_video_memory_reference.py tests/test_context_predictor.py::test_reference_prompt_refinement_keeps_original_prompts -q
```

Expected: PASS.

- [ ] **Step 8: Commit refinement workflow**

Run:

```bash
git add src/predict/refine scripts/video_memory_reference.py tests/test_video_memory_reference.py tests/test_context_predictor.py
git commit -m "feat: add explicit mask refinement workflow"
```

## Task 6: Move Next-Frame Prediction Package

**Files:**
- Move: `src/predict/video.py` to `src/predict/next_frame/predictor.py`
- Create: `src/predict/next_frame/__init__.py`
- Modify: `src/predict/next_frame/predictor.py`
- Modify: `tests/test_video_memory.py`
- Modify: `scripts/video_memory_smoke_test.py`
- Modify: `scripts/video_memory_effect_test.py`

- [ ] **Step 1: Move file**

Run:

```bash
mkdir src\predict\next_frame
git mv src\predict\video.py src\predict\next_frame\predictor.py
```

- [ ] **Step 2: Rename class and fix imports in `src/predict/next_frame/predictor.py`**

Change:

```python
from ..model.build import build_model
from ..types import MemoryPrediction, MemoryReference
from .image_transform import preprocess_rgb_images, scale_coords, to_rgb_pil
```

to:

```python
from ...model.build import build_model
from ...types import MemoryPrediction, MemoryReference
from ..prompted.transforms import preprocess_rgb_images, scale_coords, to_rgb_pil
```

Rename:

```python
class VideoMemoryInference:
```

to:

```python
class NextFramePredictor:
```

Update the `from_checkpoint` return annotation and constructor call:

```python
    ) -> "NextFramePredictor":
        model = build_model(
            path=path,
            device=device,
            multiplex_count=multiplex_count,
            max_num_objects=max_num_objects,
        )
        return cls(model=model.video, device=device)
```

- [ ] **Step 3: Create `src/predict/next_frame/__init__.py`**

Use:

```python
from ...types import MemoryPrediction as MemoryPrediction
from ...types import MemoryReference as MemoryReference
from .predictor import NextFramePredictor as NextFramePredictor
```

- [ ] **Step 4: Update next-frame imports in tests and scripts**

In `tests/test_video_memory.py`, replace every:

```python
from src.predict import MemoryReference, VideoMemoryInference
from src.predict import VideoMemoryInference
```

with:

```python
from src.predict.next_frame import MemoryReference, NextFramePredictor
```

Replace all `VideoMemoryInference` references with `NextFramePredictor`.

Update the public API test:

```python
def test_next_frame_public_api_imports() -> None:
    import src.predict.next_frame as next_frame
    from src.model.build import build_model
    from src.predict.next_frame import MemoryReference, NextFramePredictor

    assert NextFramePredictor.__name__ == "NextFramePredictor"
    assert MemoryReference.__name__ == "MemoryReference"
    assert MemoryReference.__module__ == "src.types"
    assert callable(build_model)
    assert next_frame.NextFramePredictor is NextFramePredictor
    assert next_frame.MemoryReference is MemoryReference
    assert not hasattr(next_frame, "__all__")
```

In the video module structure test, replace:

```python
assert (root / "src" / "predict" / "video.py").is_file()
```

with:

```python
assert (root / "src" / "predict" / "next_frame" / "predictor.py").is_file()
assert not (root / "src" / "predict" / "video.py").exists()
```

In `scripts/video_memory_smoke_test.py`, replace:

```python
from src.predict import MemoryReference, VideoMemoryInference
```

with:

```python
from src.predict.next_frame import MemoryReference, NextFramePredictor
```

and replace `VideoMemoryInference.from_checkpoint` with
`NextFramePredictor.from_checkpoint`.

In `scripts/video_memory_effect_test.py`, make the same import and class-name
replacement. Also update type annotations from `VideoMemoryInference` to
`NextFramePredictor`.

- [ ] **Step 5: Run next-frame tests**

Run:

```bash
python -m pytest tests/test_video_memory.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit next-frame move**

Run:

```bash
git add src/predict/next_frame tests/test_video_memory.py scripts/video_memory_smoke_test.py scripts/video_memory_effect_test.py
git add -u src/predict/video.py
git commit -m "refactor: move next frame prediction package"
```

## Task 7: Repository-Wide Import Cleanup And Old Path Removal Check

**Files:**
- Modify any remaining files under `src/`, `tests/`, and `scripts/` that import old paths.

- [ ] **Step 1: Search for old imports**

Run:

```bash
rg -n "src\\.predict\\.(image|masks|reference|video)|from src\\.predict import|import src\\.predict as|from \\.\\.image|from \\.\\.masks|from \\.\\.reference|from \\.\\.video|from \\.image|from \\.video" src tests scripts -S
```

Expected: No references to removed workflow paths. References to
`src.predict.grounding` are allowed.

- [ ] **Step 2: Fix any remaining old imports**

Use these replacements:

```text
src.predict.image -> src.predict.prompted.predictor
src.predict.image_transform -> src.predict.prompted.transforms
src.predict.masks -> src.predict.grid
src.predict.reference -> src.predict.context
src.predict.video -> src.predict.next_frame
from src.predict import Sam3Predictor -> from src.predict.prompted import Sam3Predictor
from src.predict import MemoryReference, VideoMemoryInference -> from src.predict.next_frame import MemoryReference, NextFramePredictor
```

If a remaining import expects `VideoMemoryInference`, rename the call site to
`NextFramePredictor`. Do not add compatibility aliases.

- [ ] **Step 3: Run package structure and focused workflow tests**

Run:

```bash
python -m pytest tests/test_package_structure.py tests/test_predictor_api.py tests/test_auto_mask_generator.py tests/test_context_predictor.py tests/test_reference_guided_masks.py tests/test_video_memory.py tests/test_video_memory_reference.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit cleanup**

Run:

```bash
git add src tests scripts
git commit -m "refactor: remove old predict workflow paths"
```

## Task 8: Final Verification

**Files:**
- No code changes expected unless verification exposes a defect.

- [ ] **Step 1: Run full relevant test set**

Run:

```bash
python -m pytest tests/test_predictor_api.py tests/test_auto_mask_generator.py tests/test_context_predictor.py tests/test_reference_guided_masks.py tests/test_video_memory.py tests/test_video_memory_reference.py tests/test_package_structure.py tests/test_mask_instances.py tests/test_transforms.py tests/test_grounding.py tests/test_grounding_postprocess.py -q
```

Expected: PASS.

- [ ] **Step 2: Run old path search**

Run:

```bash
rg -n "src\\.predict\\.(image|masks|reference|video)|from src\\.predict import|import src\\.predict as" src tests scripts -S
```

Expected: No output.

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected: only the pre-existing unrelated dirty files may remain if they were
not part of the implementation. If implementation tasks modified them, they
should already be committed with the task that required the change.

- [ ] **Step 4: Final implementation report**

Report:

```text
Implemented predict workflow boundaries:
- prompted: point/box/mask prompt prediction
- grid: grid point candidate generation
- refine: explicit mask_input second-pass refinement
- context: find again from reference mask context
- next_frame: one target frame from reference frame memory

Verification:
- `python -m pytest tests/test_predictor_api.py tests/test_auto_mask_generator.py tests/test_context_predictor.py tests/test_reference_guided_masks.py tests/test_video_memory.py tests/test_video_memory_reference.py tests/test_package_structure.py tests/test_mask_instances.py tests/test_transforms.py tests/test_grounding.py tests/test_grounding_postprocess.py -q`: PASS
- `rg -n "src\\.predict\\.(image|masks|reference|video)|from src\\.predict import|import src\\.predict as" src tests scripts -S`: no output

Remaining worktree:
- `git status --short`: only pre-existing unrelated files remain, or clean if those files were intentionally included in task commits
```
