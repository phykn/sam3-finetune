# Embedding Batch Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up crop-pyramid automatic masks without changing mask quality by exposing tensor image embeddings and batching crop image encoding.

**Architecture:** Add `Sam3ImageEmbedding` as the explicit tensor boundary in `src/predictor.py`. Refactor prompt decoding to operate on embeddings, keep `set_image` and `predict` as compatibility wrappers, then let `Sam3AutomaticMaskGenerator` batch crop encoding through `encode_image_batch` / `encode_image_tensor_batch` while preserving the same crop and point prompts.

**Tech Stack:** Python, NumPy, PIL, PyTorch, pytest, local CUDA smoke tests.

---

## File Structure

- Modify: `src/predictor.py`
  Adds `Sam3ImageEmbedding`, tensor-first image encoding methods, embedding-based
  prediction, and compatibility wrappers.
- Modify: `tests/test_predictor_api.py`
  Adds fake-model tests for tensor batch encoding, image batch encoding,
  `predict_from_embedding`, and backward-compatible `predict`.
- Modify: `src/auto_mask_generator.py`
  Adds `crop_encode_batch_size`, uses embedding batches for crop jobs, and falls
  back to single-crop encoding on CUDA OOM.
- Modify: `tests/test_auto_mask_generator.py`
  Adds fake predictor tests for batched crop encode and option validation.
- Modify: `scripts/auto_mask_smoke_test.py`
  Adds `--crop-encode-batch-size` and prints the selected value.
- No changes: `sam3-main/`, `weight/`, `.venv/`, `outputs/`.

### Task 1: Predictor Embedding API

**Files:**
- Modify: `tests/test_predictor_api.py`
- Modify: `src/predictor.py`

- [ ] **Step 1: Write failing predictor embedding tests**

Update `FakeModel.encode_image` in `tests/test_predictor_api.py` so returned
feature batch size follows the input batch:

```python
    def encode_image(self, images):
        self.last_encoded_shape = tuple(images.shape)
        batch_size = images.shape[0]
        return {
            "image_embed": torch.arange(
                batch_size * 256 * 72 * 72,
                dtype=torch.float32,
            ).reshape(batch_size, 256, 72, 72),
            "high_res_features": [
                torch.zeros(batch_size, 32, 288, 288),
                torch.zeros(batch_size, 64, 144, 144),
            ],
        }
```

Update imports:

```python
from src.predictor import Sam3ImageEmbedding, Sam3Predictor
```

Append tests:

```python
def test_encode_image_tensor_batch_returns_one_embedding_per_tensor():
    model = FakeModel()
    predictor = Sam3Predictor(model, device=torch.device("cpu"))
    input_tensor = torch.zeros(2, 3, 1008, 1008)

    embeddings = predictor.encode_image_tensor_batch(
        input_tensor,
        [(10, 20), (30, 40)],
    )

    assert len(embeddings) == 2
    assert isinstance(embeddings[0], Sam3ImageEmbedding)
    assert embeddings[0].image_embed.shape[0] == 1
    assert embeddings[0].orig_hw == (10, 20)
    assert embeddings[1].orig_hw == (30, 40)
    assert model.last_encoded_shape == (2, 3, 1008, 1008)


def test_encode_image_tensor_batch_rejects_empty_batch():
    predictor = Sam3Predictor(FakeModel(), device=torch.device("cpu"))

    try:
        predictor.encode_image_tensor_batch(torch.zeros(0, 3, 1008, 1008), [])
    except ValueError as exc:
        assert "batch" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_encode_image_batch_stacks_preprocessed_images():
    model = FakeModel()
    predictor = Sam3Predictor(model, device=torch.device("cpu"))

    embeddings = predictor.encode_image_batch(
        [
            Image.new("RGB", (20, 10), color=(0, 0, 0)),
            Image.new("RGB", (40, 30), color=(0, 0, 0)),
        ]
    )

    assert len(embeddings) == 2
    assert model.last_encoded_shape == (2, 3, 1008, 1008)
    assert [embedding.orig_hw for embedding in embeddings] == [(10, 20), (30, 40)]


def test_predict_from_embedding_does_not_require_set_image():
    predictor = Sam3Predictor(FakeModel(), device=torch.device("cpu"))
    embedding = predictor.encode_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    masks, scores, low_res = predictor.predict_from_embedding(
        embedding,
        point_coords=np.array([[[10, 5]]], dtype=np.float32),
        point_labels=np.array([[1]], dtype=np.int64),
    )

    assert masks.shape == (1, 10, 20)
    assert scores.shape == (1,)
    assert low_res.shape == (1, 288, 288)
```

- [ ] **Step 2: Run predictor tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_predictor_api.py -q
```

Expected: FAIL because `Sam3ImageEmbedding`, `encode_image_tensor_batch`,
`encode_image_batch`, and `predict_from_embedding` are missing.

- [ ] **Step 3: Implement `Sam3ImageEmbedding` and encode methods**

In `src/predictor.py`:

```python
from contextlib import nullcontext
from dataclasses import dataclass
from collections.abc import Sequence
```

Add:

```python
@dataclass(frozen=True)
class Sam3ImageEmbedding:
    image_embed: torch.Tensor
    high_res_features: tuple[torch.Tensor, ...]
    orig_hw: tuple[int, int]
```

Add methods:

```python
def encode_image_tensor_batch(
    self,
    input_tensor: torch.Tensor,
    orig_hws: Sequence[tuple[int, int]],
    *,
    inference: bool = True,
) -> list[Sam3ImageEmbedding]:
    if input_tensor.ndim != 4 or input_tensor.shape[0] == 0:
        raise ValueError("input_tensor must be a non-empty BCHW batch")
    if input_tensor.shape[0] != len(orig_hws):
        raise ValueError("orig_hws length must match batch size")
    context = torch.inference_mode() if inference else nullcontext()
    with context:
        features = self.model.encode_image(input_tensor.to(self.device))
    image_embed = features["image_embed"]
    high_res_features = tuple(features["high_res_features"])
    embeddings = []
    for index, orig_hw in enumerate(orig_hws):
        embeddings.append(
            Sam3ImageEmbedding(
                image_embed=image_embed[index : index + 1],
                high_res_features=tuple(
                    feature[index : index + 1] for feature in high_res_features
                ),
                orig_hw=orig_hw,
            )
        )
    return embeddings
```

Add convenience methods:

```python
def encode_image(self, image, *, inference=True):
    embeddings = self.encode_image_batch([image], inference=inference)
    return embeddings[0]

def encode_image_batch(self, images, *, inference=True):
    if not images:
        raise ValueError("images batch must be non-empty")
    tensors = []
    orig_hws = []
    for image in images:
        tensor, orig_hw = self.transforms.preprocess_image(image, self.device)
        tensors.append(tensor)
        orig_hws.append(orig_hw)
    input_tensor = torch.cat(tensors, dim=0)
    return self.encode_image_tensor_batch(input_tensor, orig_hws, inference=inference)

def set_image_embedding(self, embedding):
    self._embedding = embedding
```

Update `__init__` to use:

```python
self._embedding: Sam3ImageEmbedding | None = None
```

Update `set_image`:

```python
self.set_image_embedding(self.encode_image(image))
```

- [ ] **Step 4: Refactor predict to embedding path**

Move current `predict` logic into:

```python
def predict_from_embedding(self, embedding: Sam3ImageEmbedding, ...):
```

Replace `self._features["image_embed"]` with `embedding.image_embed`, replace
`self._features["high_res_features"]` with `list(embedding.high_res_features)`,
and replace `self._orig_hw` with `embedding.orig_hw`.

Update `predict`:

```python
if self._embedding is None:
    raise RuntimeError("Call set_image() before predict().")
return self.predict_from_embedding(self._embedding, ...)
```

- [ ] **Step 5: Run predictor tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_predictor_api.py -q
```

Expected: PASS.

- [ ] **Step 6: Run all unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 7: Commit predictor API**

Run:

```powershell
git add src/predictor.py tests/test_predictor_api.py
git commit -m "feat: expose sam3 image embeddings"
```

### Task 2: Generator Batched Crop Encoding

**Files:**
- Modify: `tests/test_auto_mask_generator.py`
- Modify: `src/auto_mask_generator.py`

- [ ] **Step 1: Add failing generator batch encode tests**

In `tests/test_auto_mask_generator.py`, add to `CropAwareFakePredictor`:

```python
        self.encoded_batches = []

    def encode_image_batch(self, images):
        self.encoded_batches.append([image.size for image in images])
        return [{"image": image, "size": image.size} for image in images]

    def predict_from_embedding(
        self,
        embedding,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        self.images.append(embedding["size"])
        return self.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )
```

Append tests:

```python
def test_generator_rejects_invalid_crop_encode_batch_size():
    try:
        Sam3AutomaticMaskGenerator(CropAwareFakePredictor(), crop_encode_batch_size=0)
    except ValueError as exc:
        assert "crop_encode_batch_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_generator_batches_crop_encoding_without_changing_outputs():
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
        filter_crop_edge_masks=False,
        crop_encode_batch_size=2,
    )

    proposals = generator.generate(Image.new("RGB", (8, 8), color=(0, 0, 0)))

    assert predictor.encoded_batches == [[(4, 4), (4, 4)], [(4, 4), (4, 4)]]
    assert len(proposals) == 4
    assert sorted(proposal.crop_index for proposal in proposals) == [0, 1, 2, 3]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: FAIL because `crop_encode_batch_size` is not accepted and generator
does not call `encode_image_batch`.

- [ ] **Step 3: Add generator option and batch crop jobs**

In `Sam3AutomaticMaskGenerator.__init__`, add:

```python
crop_encode_batch_size: int = 1
```

Validate:

```python
if crop_encode_batch_size <= 0:
    raise ValueError("crop_encode_batch_size must be a positive integer")
self.crop_encode_batch_size = crop_encode_batch_size
```

Refactor `generate` so it builds crop jobs per crop grid and processes them in
batches of `self.crop_encode_batch_size`.

- [ ] **Step 4: Decode from embeddings when available**

Add helper:

```python
def _generate_for_crop_embedding(...):
```

It should match `_generate_for_crop`, but call:

```python
self.predictor.predict_from_embedding(embedding, point_coords=..., point_labels=...)
```

instead of `set_image` + `predict`.

Keep `_generate_for_crop` as the single-crop fallback for predictors that do
not expose `encode_image_batch`.

- [ ] **Step 5: Add CUDA OOM fallback**

When calling `predictor.encode_image_batch(crop_images)`, catch only CUDA OOM:

```python
except torch.cuda.OutOfMemoryError:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for job in crop_jobs:
        proposals.extend(self._generate_for_crop(...))
```

Other exceptions should propagate.

- [ ] **Step 6: Run generator tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_mask_generator.py -q
```

Expected: PASS.

- [ ] **Step 7: Run all unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 8: Commit generator batch encoding**

Run:

```powershell
git add src/auto_mask_generator.py tests/test_auto_mask_generator.py
git commit -m "feat: batch crop image embeddings"
```

### Task 3: Smoke CLI And Verification

**Files:**
- Modify: `scripts/auto_mask_smoke_test.py`

- [ ] **Step 1: Add CLI option**

Add:

```python
parser.add_argument("--crop-encode-batch-size", type=int, default=1)
```

Pass it to the generator:

```python
crop_encode_batch_size=args.crop_encode_batch_size,
```

Print it:

```python
print(f"crop_encode_batch_size: {args.crop_encode_batch_size}")
```

- [ ] **Step 2: Run all unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 3: Run baseline speed smoke**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 32 32 --max-masks 200 --crop-encode-batch-size 1
```

Expected: PASS with `proposal_count: 43`.

- [ ] **Step 4: Run batched speed smoke**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 32 32 --max-masks 200 --crop-encode-batch-size 2
```

Expected: PASS with `proposal_count: 43`, or clean fallback to equivalent
outputs if batch encoding OOMs.

- [ ] **Step 5: Commit smoke CLI**

Run:

```powershell
git add scripts/auto_mask_smoke_test.py
git commit -m "test: add crop encode batch smoke option"
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

- [ ] **Step 2: Run existing prompt smoke**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

Expected: PASS with `missing_keys: 0` and `unexpected_keys: 0`.

- [ ] **Step 3: Run routine crop smoke**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 32 32 --max-masks 200 --crop-encode-batch-size 2
```

Expected: PASS, proposal count remains 43, and outputs are written under
`outputs/`.

- [ ] **Step 4: Confirm ignored artifacts are not staged**

Run:

```powershell
git status --short --ignored
```

Expected: `.venv/`, `outputs/`, `sam3-main/`, and `weight/` appear only as
ignored (`!!`) or are absent from tracked/untracked output.

- [ ] **Step 5: Push**

Run:

```powershell
git status --short
git push
```

Expected: clean tracked working tree and successful push to `origin/main`.

## Self-Review

Spec coverage:

- Tensor-first embedding API: Task 1.
- `Sam3ImageEmbedding`: Task 1.
- `predict_from_embedding`: Task 1.
- Backward-compatible `set_image` and `predict`: Task 1.
- Batched crop image encoding: Task 2.
- `crop_encode_batch_size`: Task 2 and Task 3.
- OOM fallback: Task 2.
- Smoke comparison and final verification: Task 3 and Task 4.
- No quality-reducing prompt or crop changes: Task 2 keeps existing crop and point grids.

Placeholder scan:

- No placeholder markers are used.
- Commands and expected outcomes are concrete.

Type consistency:

- `orig_hw` is `(height, width)`.
- `image_size` in proposals remains `(width, height)`.
- `encode_image_tensor_batch` returns `list[Sam3ImageEmbedding]`.
- `crop_encode_batch_size` is a positive integer.
