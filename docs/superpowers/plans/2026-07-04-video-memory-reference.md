# Video Memory Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real SAM 3.1 tracker memory support to the `src/` rewrite for multi-reference image+mask conditioning.

**Architecture:** Port the upstream tracker-only multiplex stack into `src` and expose a small local-checkpoint memory predictor. Keep text/HF/detector code out of the public API. Use tests to lock importability, checkpoint key mapping, and multi-reference state behavior before running a GPU smoke.

**Tech Stack:** Python, PyTorch, pytest, PIL/numpy, local `weight/sam3.1_multiplex.pt`.

---

### Task 1: Add RED Tests For Video Memory Surface

**Files:**
- Create: `tests/test_video_memory.py`
- No production code yet

- [ ] Write tests that import the planned public API names:
  `Sam3MemoryPredictor`, `Sam3MemoryReference`, and
  `build_video_memory_model`.
- [ ] Write a checkpoint-filter test that loads synthetic keys with
  `tracker.model.maskmem_backbone.*`, `tracker.model.sam_mask_decoder.*`,
  `tracker.model.transformer.*`, and `detector.backbone.vision_backbone.*`;
  assert those keys are retained for the video loader.
- [ ] Write a multi-reference state test that constructs two
  `Sam3MemoryReference` objects with the same `obj_id` and verifies frame order
  is preserved by the predictor helper.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_video_memory.py -q`
  and confirm it fails because the API does not exist.

### Task 2: Port Tracker Modules From Upstream

**Files:**
- Create: `src/memory.py`
- Create: `src/multiplex_utils.py`
- Create: `src/multiplex_mask_decoder.py`
- Create: `src/tracker_utils.py`
- Create: `src/video_tracking_multiplex.py`
- Create: `src/video_tracking_multiplex_demo.py`
- Modify: `src/model_misc.py` or create `src/decoder_memory.py`

- [ ] Mechanically copy upstream tracker-only modules from `sam3-main`.
- [ ] Rewrite imports from `sam3.model.*` and `sam3.sam.*` to local `src`
  modules.
- [ ] Add a lazy/fallback path for EDT and connected-components helpers so
  importing video memory does not require `triton`.
- [ ] Run the RED import tests and keep working until import errors are gone.

### Task 3: Add Video Memory Builder And Checkpoint Loader

**Files:**
- Create: `src/video_builder.py`
- Create: `src/video_checkpoint.py`
- Modify: `src/__init__.py`

- [ ] Implement `build_video_memory_model(checkpoint_path, device)` using the
  upstream `build_sam3_multiplex_video_model` component graph without language
  or detector modules.
- [ ] Implement video-specific checkpoint remapping that keeps
  `tracker.model.*` memory/tracker keys and vision backbone keys.
- [ ] Export builder and predictor names from `src/__init__.py`.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_video_memory.py -q`
  and confirm checkpoint-loader tests pass.

### Task 4: Add Public Memory Predictor

**Files:**
- Create: `src/memory_predictor.py`
- Test: `tests/test_video_memory.py`

- [ ] Add `Sam3MemoryReference` with `image`, `mask`, `obj_id`, and optional
  prompt metadata.
- [ ] Add `Sam3MemoryPredictor` methods to normalize reference ordering and
  initialize an image-list pseudo-video state.
- [ ] Implement mask-reference insertion through the tracker `add_new_masks`
  path.
- [ ] Add target prediction through tracker propagation.
- [ ] Run the focused pytest file.

### Task 5: Add GPU Smoke Script

**Files:**
- Create: `scripts/video_memory_smoke_test.py`

- [ ] Build two or more pseudo frames from `asset/sample.jpg`.
- [ ] Use the existing image prompt predictor or a deterministic mask box to
  create a reference mask.
- [ ] Add the reference mask to memory, propagate to the target frame, and save
  a visual overlay under `outputs/`.
- [ ] Run the script with local checkpoint:
  `.\.venv\Scripts\python.exe scripts\video_memory_smoke_test.py --checkpoint weight\sam3.1_multiplex.pt`
- [ ] Report runtime and output path.

### Self-Review

- The plan keeps `sam3-main/` reference-only.
- The public API is tracker-memory focused and excludes language/HF.
- Tests cover import, checkpoint routing, multi-reference ordering, and smoke.
- The main remaining risk is upstream tracker size; if a copied helper imports
  optional CUDA/Triton kernels eagerly, replace that import with a local lazy
  fallback before broadening dependencies.
