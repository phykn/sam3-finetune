# Cross-Crop Decoder Batch Design

## Context

Automatic mask generation spends most runtime in repeated mask decoder calls. The
model internals match the upstream SAM mask decoder and should remain unchanged.
The optimization target is therefore the caller-side data flow: reduce decoder
call count and avoid extra host/device transfers while preserving masks,
thresholds, crop grids, and NMS behavior.

Current `[1, 2]` crop recall smoke runs five crops and 80 prompt decode calls:
16 point batches for the full image plus 16 batches for each of four 2x2 crops.
Image embedding batch encode is already supported, but each crop still decodes
its point batches independently.

## Design

Add a predictor API that accepts several prompt batches attached to image
embeddings and decodes them in one mask decoder call when their prompt layout is
compatible. For automatic masks this means point-only prompts with one point per
prompt row, which is the normal grid sampling path.

The predictor will:

- prepare prompt tensors per embedding using the existing coordinate transform;
- concatenate compatible prompt batches along the prompt batch dimension;
- repeat or expand image embeddings and high-resolution features to match the
  concatenated prompt rows;
- call the existing mask decoder with `repeat_image=False`;
- split the tensor outputs back into one result tuple per original prompt batch.

The automatic mask generator will:

- keep crop image batch encoding as-is;
- create point decode jobs for encoded crops;
- submit multiple point decode jobs to the new predictor batch API;
- preserve per-crop proposal filtering, crop-local NMS, global NMS, sorting, and
  output structures;
- fall back to the existing per-crop decode path if the predictor does not
  expose the batch API or if CUDA OOM occurs during batched decoding.

## Data Optimization

The main data optimization is batching prompt embeddings before the decoder, not
changing the model. This reduces Python/model call overhead and lets the decoder
process a larger batch per call. CPU conversion remains at the split result
boundary for now so proposal filtering can stay unchanged and low risk.

The implementation must avoid changing model weights, mask decoder code,
threshold semantics, or returned ROI mask geometry.

## Verification

Unit tests will cover:

- predictor prompt batch decoding calls mask decoder once for multiple prompt
  batches and splits outputs correctly;
- automatic mask generator uses batched decoder calls without changing proposal
  metadata;
- fallback behavior remains compatible with predictors that only implement the
  older methods.

Runtime verification will use the sample image and local checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\auto_mask_smoke_test.py --crop-grids 1 2 --crop-points-per-side 32 32 --max-masks 200 --crop-encode-batch-size 2
```

The expected quality invariant is the same proposal count by crop grid as the
previous baseline unless floating point ordering changes produce a documented,
small difference.
