# Grid Finetune Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the single finetuning dataset with more refined frog and leaf masks from `GridPredictor`.

**Architecture:** Add one standalone generator that imports public predictor and sample APIs. It runs grid inference once per source image, assigns candidates to manual class regions, copies confirmed background samples, then writes class JSON and a preview under a separate root.

**Tech Stack:** Python 3.11, NumPy, Pillow, PyTorch, pytest, existing `GridPredictor` and `src.data.sample` APIs.

## Global Constraints

- Never modify `asset/` or `weight/`.
- Keep only `finetune_dataset/`; do not create versioned dataset roots.
- Use `weight/sam3.1_multiplex.pt`; do not use Hugging Face.
- Grid settings are tiles `(1, 2)`, points `(10, 10)`, overlap `0.25`, stability `0.75`, NMS `0.7`, and minimum area `64`.
- Preserve class IDs `0=background`, `1=frog`, and `2=leaf`.

---

### Task 1: Candidate class assignment

**Files:**
- Create: `scripts/make_finetune_candidates.py`
- Create: `tests/test_finetune_candidates.py`

**Interfaces:**
- Consumes: grid item dictionaries returned by `GridPredictor.predict` and manual regions from the existing dataset specification.
- Produces: `assign_class(item: dict, regions: dict[int, list[tuple[int, int, int, int]]], image_shape: tuple[int, int], min_overlap: float = 0.5) -> tuple[int, float] | None`.

- [ ] **Step 1: Write failing tests** for point containment, 50% mask overlap, best-class selection, and tied-class rejection using small boolean ROI candidates.
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_finetune_candidates.py -q` and confirm import/function failure.
- [ ] **Step 3: Implement** full-image mask reconstruction, per-region overlap calculation, unique best selection, and threshold filtering.
- [ ] **Step 4: Run** `.venv/Scripts/python -m pytest tests/test_finetune_candidates.py -q` and confirm assignment tests pass.

### Task 2: Candidate dataset writer

**Files:**
- Modify: `scripts/make_finetune_candidates.py`
- Modify: `tests/test_finetune_candidates.py`

**Interfaces:**
- Consumes: assigned grid candidates, source PIL image, split/name/class regions, existing class-0 JSON.
- Produces: `make_sample(predictor, split: str, filename: str, classes: dict[int, list[tuple[int, int, int, int]]]) -> dict[int, int]` and same-schema class JSON files.

- [ ] **Step 1: Write failing test** with a fake predictor and temporary source/background paths; assert one grid call per image, background copy semantics, class-local object IDs, metrics/metadata, and JSON round-trip through `load`.
- [ ] **Step 2: Run the focused test** and confirm failure because dataset writing is absent.
- [ ] **Step 3: Implement minimal writing** using `Sample`, `Object`, and `save`; fail before inference when source image or background JSON is missing.
- [ ] **Step 4: Run the focused tests** and confirm all pass.

### Task 3: Preview and real generation

**Files:**
- Modify: `scripts/make_finetune_candidates.py`
- Modify: `tests/test_finetune_candidates.py`

**Interfaces:**
- Consumes: source image and assigned class objects.
- Produces: one `preview/<split>_<stem>.jpg`, printed per-class counts, and candidate JSON for every configured source.

- [ ] **Step 1: Write failing preview test** asserting output size and that the candidate root differs from the original dataset root.
- [ ] **Step 2: Run focused tests** and confirm the preview assertion fails.
- [ ] **Step 3: Implement preview drawing** with magenta frog, cyan leaf, IDs, score, stability, and overlap; add `main()` using the fixed grid configuration.
- [ ] **Step 4: Run** `.venv/Scripts/python -m pytest tests/test_finetune_candidates.py tests/test_scripts_standalone.py -q`.
- [ ] **Step 5: Run** `.venv/Scripts/python scripts/make_finetune_candidates.py` on the configured images.
- [ ] **Step 6: Load every generated JSON**, report old/new per-class counts, and visually inspect every preview.
- [ ] **Step 7: Run** `.venv/Scripts/python -m pytest tests` and confirm the full suite passes.
- [ ] **Step 8: Commit and push** source, tests, spec, plan, and the final `finetune_dataset/` JSON/preview files.
