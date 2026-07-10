# Grounding Reference Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. The original pre-execution checkboxes are retained below; final status and evidence are recorded separately.

**Goal:** Build a box-reference grounding predictor that extracts per-box class features, reuses each image backbone result, decodes class prompts sequentially with exact batch-1 behavior, and returns mask objects compatible with existing JSON.

**Architecture:** Each reference image is encoded once. Pixel `xyxy` boxes are grouped by integer class ID into padded box prompts while every box keeps an individual normalized feature vector. The target image is encoded once, prompt groups are decoded one at a time, and candidates are filtered by maximum same-class feature similarity before within-class NMS.

**Execution status:** Implemented. Batched target decoding was rejected after CUDA
measurement showed shape-dependent BF16 differences; sequential decoding preserves
exact single-prompt behavior while still reusing the target image encoding.

**Tech Stack:** Python 3, PyTorch, NumPy, Pillow, torchvision NMS, pytest, Black, Ruff.

## Global Constraints

- Use `D:\code\sam3\.venv` from the workspace root.
- Work on `codex/grounding-refactor`; do not push unless explicitly requested.
- Never modify or stage `sam3-main/`, `weight/`, or `asset/`.
- Load weights only from explicit local paths; do not load the text encoder or use Hugging Face.
- Keep code direct and short; no compatibility aliases, wrappers, new dataclasses, lazy imports, `__all__`, Meta headers, or future annotations.
- Public references accept pixel `xyxy` boxes and non-negative integer class IDs only.
- One and multiple references use the same non-empty list API.
- Preserve checkpoint keys and single-box decoder math.
- Apply NMS only within one class; preserve cross-class overlap.

---

### Task 1: Add Box Reference and Similarity Math

**Files:**
- Create: `src/predict/ground_ops/reference.py`
- Modify: `src/predict/ground_ops/sim.py`
- Create: `tests/test_ground_reference.py`

**Interfaces:**
- `validate(boxes, class_ids, image_shape) -> tuple[np.ndarray, np.ndarray]`.
- `groups(boxes, class_ids) -> tuple[np.ndarray, list[np.ndarray]]`.
- `feature_bank(references) -> dict[int, torch.Tensor]`.
- `prompt_groups(references) -> tuple[dict[str, torch.Tensor], np.ndarray]`.
- `box_vectors(image, boxes, orig_hw) -> Tensor[N, C]`.
- `mask_vectors(image, masks) -> Tensor[N, C]`.
- `max_scores(reference, target) -> Tensor[N]`.

- [ ] **Step 1: Write failing box validation tests**

```python
def test_validate_clips_boxes_and_preserves_classes():
    boxes, classes = reference.validate(
        [[-2, 1, 5, 7], [4, 2, 12, 9]], [2, 1], (8, 10)
    )
    np.testing.assert_array_equal(boxes, [[0, 1, 5, 7], [4, 2, 10, 8]])
    assert classes.dtype == np.int64


def test_groups_boxes_by_sorted_class():
    boxes = np.array([[0, 0, 2, 2], [2, 2, 4, 4], [4, 4, 6, 6]])
    classes, grouped = reference.groups(boxes, np.array([2, 1, 2]))
    assert classes.tolist() == [1, 2]
    np.testing.assert_array_equal(grouped[0], boxes[[1]])
    np.testing.assert_array_equal(grouped[1], boxes[[0, 2]])
```

Add parametrized failures for empty input, wrong shape, class length mismatch,
non-finite coordinates, reversed/zero-area boxes, fully outside boxes,
non-integer classes, and negative classes.

- [ ] **Step 2: Run tests and confirm the module is missing**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ground_reference.py -q`

Expected: collection fails with `ImportError`.

- [ ] **Step 3: Implement validation and grouping**

Normalize `[4]` to `[1, 4]`, validate finite float coordinates and positive area,
clip to reference image bounds, reject boxes empty after clipping, and return
float32 boxes with int64 classes. Group boxes under sorted unique classes without
averaging.

- [ ] **Step 4: Add failing feature tests**

```python
def test_max_scores_uses_best_exemplar():
    refs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    torch.testing.assert_close(sim.max_scores(refs, target), torch.tensor([0.8, 0.9]))
```

Test `box_vectors()` against an explicit rectangular feature-grid mean and test
that `feature_bank()` concatenates repeated classes across reference images.

- [ ] **Step 5: Implement feature and merge math**

Scale pixel boxes to the final FPN grid using floor starts and ceil ends. Build
rectangular grid masks and reuse a private normalized masked-mean function for box
and target-mask features. Rename old `vectors()`/`scores()` and delete `select()`.
Merge prompt features along batch dimension 1, prompt masks along dimension 0,
and class IDs in the same order.

- [ ] **Step 6: Verify and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ground_reference.py -q
git add src/predict/ground_ops/reference.py src/predict/ground_ops/sim.py tests/test_ground_reference.py
git commit -m "refactor: add grounding box reference math"
```

---

### Task 2: Batch Class Box Prompts

**Files:**
- Modify: `src/data/ground.py`
- Modify: `src/ml/blocks/grounding/image.py`
- Modify: `src/ml/blocks/grounding/decoder.py`
- Modify: `src/ml/model/grounding.py`
- Create: `tests/test_ground_batch.py`
- Modify: `tests/test_data_ground.py`
- Modify: `tests/test_builder.py`

**Interfaces:**
- `build_box_batch(groups, orig_hw, device) -> tuple[Tensor, Tensor, Tensor]`.
- `GroundingImage.expand(image, batch_size) -> dict`.
- `Sam3GroundingModel.encode_box_prompts(image, boxes, labels, box_mask) -> dict`.
- `Sam3GroundingModel.decode(image, prompt)` aligns target image batch to prompt batch.

- [ ] **Step 1: Write failing padded-box tests**

For class groups containing one and two boxes, assert box/label shapes `[2, 2, 4]`
and `[2, 2]`, padding mask shape `[2, 2]`, padding values
`[[False, True], [False, False]]`, and correct normalized center-width-height math.

- [ ] **Step 2: Implement `build_box_batch()`**

Pad class columns to the largest group, normalize pixel `xyxy` to center-width-
height, create label ones, and mark padded positions in `box_mask`. Keep existing
single-prompt point/box builders for direct model parity.

- [ ] **Step 3: Write failing encoded-image batch tests**

Test `GroundingImage.expand()` with tensors and `NestedTensor`, including `None`
masks. It must keep a matching batch, expand only batch 1 without cloning feature
storage, and reject other mismatches. Test cached visual tokens `[32, 1, 256]`
expand to a multi-prompt batch in `GroundingDecoder.prompt_inputs()`.

- [ ] **Step 4: Implement batch alignment**

Expand vision features, masks, position encodings, and FPN nested tensors while
preserving feature sizes. Add `encode_box_prompts()` to expand one encoded reference
image and call the trained box prompt encoder. In `decode()`, expand one target
image to `prompt["features"].shape[1]`. Expand cached language features on dimension
1 and language masks on dimension 0 before prompt concatenation.

- [ ] **Step 5: Verify and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ground_batch.py tests/test_data_ground.py tests/test_ground_blocks.py tests/test_builder.py -q
git add src/data/ground.py src/ml/blocks/grounding/image.py src/ml/blocks/grounding/decoder.py src/ml/model/grounding.py tests/test_ground_batch.py tests/test_data_ground.py tests/test_builder.py
git commit -m "refactor: batch grounding box prompts"
```

---

### Task 3: Add GPU Candidate Filtering and Object Output

**Files:**
- Create: `src/predict/ground_ops/output.py`
- Create: `tests/test_ground_output.py`

**Interfaces:**
- `candidates(out, image, class_ids, bank, orig_hw, score_thr, sim_thr) -> list[dict]`.
- `finish(items, nms_thr, top_k) -> list[dict]`.

- [ ] **Step 1: Write failing candidate tests**

Use a two-prompt fake decoder batch. Assert score filtering occurs first,
similarity uses only the prompt class's feature bank, retained values move to CPU,
and intermediate candidates contain private `nms_box` but no `raw` or GPU tensor.

- [ ] **Step 2: Implement candidate filtering**

Sigmoid model logits, keep score-valid queries, convert selected normalized boxes
to clipped pixel NMS boxes, resize only selected mask logits, compute target mask
features and same-class maximum similarity on GPU, then convert only retained
values to CPU.

- [ ] **Step 3: Write failing finalization tests**

Use identical boxes in classes 1 and 2. Assert same-class duplicates are removed,
cross-class overlap remains, final integer `box` comes from mask bounds, private
`nms_box` is removed, and sequential `object_id` values are assigned.

- [ ] **Step 4: Implement finalization, verify, and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ground_output.py tests/test_ground_reference.py tests/test_math_ops.py -q
git add src/predict/ground_ops/output.py tests/test_ground_output.py
git commit -m "refactor: add grounding object output"
```

---

### Task 4: Replace GroundPredictor Public API

**Files:**
- Modify: `src/predict/ground.py`
- Modify: `tests/test_ground_predict.py`
- Modify: `tests/test_predict_output.py`
- Modify: `tests/test_package_structure.py`

**Interfaces:**
- `encode_reference(image, boxes, class_ids) -> dict`.
- `predict(image, references: list[dict]) -> list[dict]`.
- Constructor options: `score_thr=0.0`, `nms_thr=0.7`, `top_k=None`, `sim_thr=0.0`.

- [ ] **Step 1: Write failing public API tests**

Assert reference classes `[2, 1, 2]` produce prompt classes `[1, 2]`, retain all
three feature classes, and encode the reference image once. Assert five prompt
groups encode the target once and decode batches `[1, 1, 1, 1, 1]`. Reject a bare
reference dict and an empty list.

- [ ] **Step 2: Implement `encode_reference()`**

Encode the image once, validate boxes/classes, compute per-box features, group
boxes, build the padded box batch, encode class prompts once, and return only
prompt features/mask/classes and per-box features/classes.

- [ ] **Step 3: Implement sequential `predict()`**

Merge feature banks and prompt groups, encode target once, decode each prompt with
batch size 1, accumulate filtered candidates, and finalize objects. Never return
or retain raw decoder output.

- [ ] **Step 4: Remove obsolete behavior**

Delete `encode_ref()`, point/mask reference options, name grouping, predictor
format/rerank methods, and old constructor names. Update `from_path()`. Keep direct
model point/box methods for parity scripts.

- [ ] **Step 5: Verify and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ground_predict.py tests/test_predict_output.py tests/test_package_structure.py tests/test_architecture.py -q
git add src/predict/ground.py tests/test_ground_predict.py tests/test_predict_output.py tests/test_package_structure.py
git commit -m "refactor: simplify grounding predictor api"
```

---

### Task 5: Use Existing Sample/Object Reference Boxes

**Files:**
- Modify: `scripts/ground.py`
- Modify: `tests/test_scripts_standalone.py`

**Interfaces:**
- `reference_arrays(sample) -> tuple[np.ndarray, np.ndarray]`.
- `refine(image, objects, device) -> None`.
- `make_result(image, objects) -> Sample`.

- [ ] **Step 1: Write failing data-flow tests**

Assert `reference_arrays()` reads `Object.box` and integer `class_id` without mask
reconstruction and rejects missing/non-integer classes. Update refinement to stack
all object logits into one `SinglePredictor.predict_embed()` call. Update JSON
round-trip assertions for class ID, compact mask, score, similarity, and refined
score.

- [ ] **Step 2: Rewrite `scripts/ground.py`**

Represent local reference annotations as existing `Sample/Object` data, call the
new reference API, batch refinement, convert final masks through `pack.box_roi()`,
and draw only from JSON-reloaded data. Remove reference names and old grouped-array
pack/read helpers.

- [ ] **Step 3: Verify and commit**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_scripts_standalone.py tests/test_ground_predict.py tests/test_data_sample.py -q
git add scripts/ground.py tests/test_scripts_standalone.py
git commit -m "refactor: use sample boxes for grounding"
```

---

### Task 6: Verify Checkpoint Parity and CUDA Performance

**Files:**
- Create: `scripts/bench_ground.py`
- Modify: `scripts/parity_ground.py` only if renamed helpers require it.
- Modify: `tests/test_scripts_standalone.py`

- [ ] **Step 1: Add an import-only benchmark test**

Importing `scripts.bench_ground` must not load weights or run CUDA work.

- [ ] **Step 2: Implement benchmark**

Load one model and cached visual tokens, encode several class box prompts, warm up,
then measure two repeated predictions. Synchronize CUDA, reset/read peak memory,
assert repeated object outputs match, and print latency, peak VRAM, and counts.

- [ ] **Step 3: Run real verification**

```powershell
.venv\Scripts\python.exe scripts/parity_ground.py
.venv\Scripts\python.exe scripts/ground.py
.venv\Scripts\python.exe scripts/bench_ground.py
```

Expected: strict load and raw single-box parity pass; result JSON uses
`sam3.sample.v1`; repeated outputs match with measurements printed.

- [ ] **Step 4: Commit benchmark coverage**

```powershell
git add scripts/bench_ground.py scripts/parity_ground.py tests/test_scripts_standalone.py
git commit -m "test: verify grounding inference performance"
```

---

### Task 7: Document and Complete Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-10-grounding-refactor.md`

- [ ] **Step 1: Update README**

Document the box/class reference API, list-only prediction, per-box features,
class prompts, cross-class overlap, object output, JSON mapping, and mask-to-box as
a separate future helper.

- [ ] **Step 2: Run focused and full tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ground_reference.py tests/test_ground_batch.py tests/test_ground_output.py tests/test_ground_predict.py tests/test_ground_blocks.py tests/test_data_ground.py tests/test_scripts_standalone.py tests/test_builder.py tests/test_architecture.py -q
.venv\Scripts\python.exe -m pytest tests -q
```

- [ ] **Step 3: Run style and scope checks**

```powershell
$files = git diff --name-only --diff-filter=ACMR main...HEAD -- '*.py'
.venv\Scripts\python.exe -m black --check $files
.venv\Scripts\python.exe -m ruff check src tests scripts
git diff main...HEAD --check
git diff --name-only main...HEAD -- sam3-main weight asset
git status --short
```

- [ ] **Step 4: Record evidence and commit**

Record full test count, parity values, repeated latency, peak VRAM, and measurement
limitations in this plan.

```powershell
git add README.md docs/superpowers/plans/2026-07-10-grounding-refactor.md tests
git commit -m "docs: document grounding box workflow"
```

---

## Execution Evidence

- Focused grounding tests: 67 passed.
- Full test suite: 238 passed.
- Official SAM3.1 strict checkpoint load: passed.
- Single-box old/new raw parity: logits, boxes, and masks all had maximum and
  mean differences of zero; final mask XOR was zero.
- Final `scripts/ground.py` CUDA run: 20 objects saved and reloaded through the
  existing JSON path.
- Repeated CUDA benchmark on NVIDIA GeForce RTX 2060: 12 identical objects;
  11891.608 ms and 11834.144 ms; 3874.407 MiB peak allocation for both runs.
- Target batch experiment: BF16 encoder maximum difference `0.25` and object-count
  drift `12 -> 13` between batch sizes 1 and 4. Float32 maximum difference was
  `1.57356e-5`. The public predictor therefore uses exact batch-1 target decoding.
- Changed-file Black, repository Ruff, diff, and protected-path checks passed.

---

## Completion Criteria

- Public reference input is pixel `xyxy` boxes plus integer class IDs.
- Every box retains an individual normalized feature vector.
- Same-image/class boxes form one trained box prompt.
- One and multiple references use the same non-empty list API.
- Each image backbone runs once and target prompts decode sequentially.
- Class similarity uses maximum same-class exemplar cosine similarity.
- Same-class NMS removes duplicates; cross-class overlaps remain.
- Returned objects contain CPU NumPy masks and no raw GPU tensors.
- Existing JSON preserves class IDs, compact masks, and metrics.
- Strict load, raw single-box parity, full tests, style, and protected paths pass.
- CUDA repeated-output equality, latency, and peak VRAM are measured.
