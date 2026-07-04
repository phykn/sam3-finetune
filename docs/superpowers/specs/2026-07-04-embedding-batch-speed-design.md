# Embedding Batch Speed Design

## Goal

Improve automatic mask generation speed while preserving mask quality. The
optimization must keep the same crop grids, point grids, prompts, thresholds,
and NMS behavior. It should also expose image embeddings as tensor-backed
objects so future fine-tuning or training workflows can reuse the normal
`image -> tensor -> embedding -> prompt decode` path.

Current routine recall smoke target:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 32 32 --max-masks 200
```

Recent baseline on RTX 2060:

- `points_per_batch=64`: `17.76s`, `proposal_count=43`,
  `proposal_count_by_crop_grid={1: 24, 2: 19}`
- `points_per_batch=32`: `18.16s`, same proposal count
- `points_per_batch=128`: `40.26s`, same proposal count

Increasing prompt batch size does not improve speed on this GPU. The next
quality-preserving optimization should target image embedding work.

## Non-Goals

- Do not reduce `crop_points_per_side`.
- Do not drop crop grids from the user-selected configuration.
- Do not change thresholds, mask scoring, crop-edge filtering, or NMS defaults.
- Do not use Hugging Face.
- Do not modify `sam3-main/`; it remains reference-only.
- Do not commit checkpoints, generated outputs, or local environments.
- Do not implement training or fine-tuning yet. This task only exposes the
  embedding tensor boundary needed later.

## Public Embedding API

Add a tensor-backed embedding object in `src/predictor.py`:

```python
@dataclass(frozen=True)
class Sam3ImageEmbedding:
    image_embed: torch.Tensor
    high_res_features: tuple[torch.Tensor, ...]
    orig_hw: tuple[int, int]
```

Field semantics:

- `image_embed`: tensor returned by the image encoder for one image, with a
  leading batch dimension of `1`.
- `high_res_features`: high-resolution decoder feature tensors for the same one
  image, each with a leading batch dimension of `1`.
- `orig_hw`: original image height and width used by coordinate transforms and
  mask postprocessing.

Add predictor methods:

```python
def encode_image_tensor_batch(
    self,
    input_tensor: torch.Tensor,
    orig_hws: Sequence[tuple[int, int]],
    *,
    inference: bool = True,
) -> list[Sam3ImageEmbedding]

def encode_image(
    self,
    image: Image.Image | np.ndarray,
    *,
    inference: bool = True,
) -> Sam3ImageEmbedding

def encode_image_batch(
    self,
    images: Sequence[Image.Image | np.ndarray],
    *,
    inference: bool = True,
) -> list[Sam3ImageEmbedding]

def set_image_embedding(self, embedding: Sam3ImageEmbedding) -> None

def predict_from_embedding(
    self,
    embedding: Sam3ImageEmbedding,
    point_coords: np.ndarray | None = None,
    point_labels: np.ndarray | None = None,
    box: np.ndarray | None = None,
    mask_input: np.ndarray | torch.Tensor | None = None,
    multimask_output: bool = True,
    return_logits: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Tensor API semantics:

- `input_tensor` is the normalized model input tensor with shape
  `(B, 3, 1008, 1008)`.
- `orig_hws` has one `(height, width)` entry per batch item.
- `encode_image_tensor_batch(...)` is the core embedding path. The PIL/NumPy
  convenience methods preprocess inputs and delegate to this tensor method.
- Future fine-tuning can call `encode_image_tensor_batch(..., inference=False)`
  with tensors from a dataloader instead of going through PIL objects.

Compatibility:

- `set_image(image)` becomes a wrapper around `encode_image(image)` and
  `set_image_embedding(...)`.
- `predict(...)` keeps the current API and delegates to the currently stored
  `Sam3ImageEmbedding`.
- Existing point, box, and mask smoke scripts should not need changes.

Fine-tuning readiness:

- `encode_image(..., inference=True)` should preserve current inference behavior
  using `torch.inference_mode()`.
- `encode_image_tensor_batch(..., inference=False)` and the convenience wrappers
  should not use `torch.inference_mode()` so image encoder gradients can be
  enabled later when the caller controls `model.train()` and optimizer state.
- `Sam3ImageEmbedding` should store actual tensors, not detached NumPy arrays.

## Batched Crop Encoding

Add a batch encode path to reduce crop pyramid overhead:

1. Build crop boxes as today.
2. Group crop jobs by `(crop_grid, crop_width, crop_height, points_per_side)`.
3. For each group, preprocess crop images into one tensor batch.
4. Call `predictor.encode_image_tensor_batch(batch_tensor, orig_hws)` once per
   group batch.
5. Split the returned batched tensors into one `Sam3ImageEmbedding` per crop.
6. Decode prompts for each embedding using the existing prompt batches.

New generator option:

```python
crop_encode_batch_size: int = 1
```

Behavior:

- `crop_encode_batch_size=1` preserves the current one-crop-at-a-time embedding
  path.
- `crop_encode_batch_size=2` is the first target for RTX 2060.
- Values larger than available GPU memory may fail. The implementation should
  catch CUDA OOM during crop batch encoding, clear CUDA cache when available,
  and fall back to encoding crops one at a time.
- Prompt decoding still uses `points_per_batch=64` by default because measured
  `128` was slower.

The crop batch path must use `Sam3ImageEmbedding` objects. It should not store
opaque feature dictionaries directly inside the generator.

## Decoder Refactor

Move the body of `Sam3Predictor.predict(...)` into a shared private method that
accepts a `Sam3ImageEmbedding`:

```python
def _predict_with_embedding(
    self,
    embedding: Sam3ImageEmbedding,
    ...
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```

The method uses:

```python
embedding.image_embed
embedding.high_res_features
embedding.orig_hw
```

instead of `self._features` and `self._orig_hw`.

This makes the prompt decoder independent from hidden predictor state and keeps
future fine-tuning experiments closer to ordinary tensor workflows.

## Performance Measurement

Update `scripts/auto_mask_smoke_test.py`:

```powershell
--crop-encode-batch-size 2
```

Print:

- `elapsed_sec`
- `crop_encode_batch_size`
- `proposal_count`
- `proposal_count_by_crop_grid`

Routine comparison commands:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 32 32 --max-masks 200 --crop-encode-batch-size 1
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 32 32 --max-masks 200 --crop-encode-batch-size 2
```

Success criteria:

- `proposal_count` remains `43` on `asset/sample.jpg` with current weights.
- `proposal_count_by_crop_grid` remains `{1: 24, 2: 19}`.
- Top proposal boxes remain materially the same.
- `crop_encode_batch_size=2` is faster than `1`, or falls back cleanly without
  changing outputs if the GPU cannot run batched encoding.

## Tests

Unit tests should not load the checkpoint.

Add fake-model tests for `Sam3Predictor`:

- `encode_image_tensor_batch` accepts a tensor batch and returns one
  `Sam3ImageEmbedding` per input tensor.
- `encode_image_batch` stacks preprocessed images and splits batch features into
  `Sam3ImageEmbedding` instances.
- `set_image` stores a `Sam3ImageEmbedding`.
- `predict_from_embedding` does not require prior `set_image`.
- `predict` remains backward compatible after `set_image`.

Add generator tests:

- With `crop_encode_batch_size=2`, fake predictor receives batched crop encode
  calls for `2x2` crops.
- Proposal metadata and ROI masks match the existing per-crop encode path.
- Invalid `crop_encode_batch_size <= 0` raises `ValueError`.

## Error Handling

- Empty image batch passed to `encode_image_batch` raises `ValueError`.
- `crop_encode_batch_size <= 0` raises `ValueError`.
- If batched crop encoding raises CUDA OOM, generator falls back to single-crop
  encoding and continues.
- Other exceptions should propagate.

## Git And Artifact Rules

- Commit code, tests, scripts, and docs only.
- Do not stage or commit `sam3-main/`, `weight/`, `.venv/`, or `outputs/`.
- Generated PNGs remain ignored under `outputs/`.

## Open Decisions Resolved

- Preserve mask quality by keeping crop grids, point grids, thresholds, and NMS.
- Do not increase `points_per_batch` beyond 64 by default.
- Expose image embeddings as first-class tensor objects.
- Use batched crop image encoding as the first speed optimization.
- Routine crop smoke uses `[1, 2]`, not `[1, 2, 4]`.
