# Finetune, Resume, and DDP Refactor Design

Date: 2026-07-10
Status: Approved and implemented on `codex/finetune-ddp-refactor`

## 1. Purpose

Refactor the image finetuning path into a short, one-way pipeline while preserving
the mixture-of-experts design. Correct the training objectives, expose per-mask
class-head results during prediction, add resumable training checkpoints, and
support single-server multi-GPU training with `torchrun` and DDP.

The intended application is battery SEM imagery. Class output index 0 always means
"this mask is a real object/particle." Later indices are independent binary
attributes whose meanings may change between projects.

## 2. Goals

- Keep the image/condition/prompt-dependent expert router.
- Keep the SAM3.1 base model frozen and load it from an explicit local path.
- Make the finetune flow read in one direction:

  ```text
  batch
    -> frozen image and prompt encoders
    -> expert router
    -> feature adapters and decoder LoRA
    -> mask, IoU, and mask token
    -> class head
    -> loss
    -> trainer
  ```

- Replace ambiguous or disconnected loss behavior with explicit equations.
- Return class logits and probabilities for every predicted mask.
- Preserve class probabilities through grid refinement and JSON output.
- Resume a new-format training checkpoint with model, optimizer, and global step.
- Run unchanged in one process or under single-server multi-GPU DDP.
- Keep code flat, short, and narrowly scoped.

## 3. Non-goals

- Loading historical LoRA/adapter checkpoints for inference.
- Maintaining compatibility with the current finetune checkpoint parameter names.
- Exact resume of RNG, augmentation, worker, sampler, or DataLoader position.
- Automatic detection or filtering based on class score. The predictor will expose
  the score, but automatic filtering remains a later task.
- Multi-node scheduler integrations such as SLURM.
- Refactoring the full dataset and augmentation subsystem. Only changes required by
  the finetune loss and DDP sampler are in scope.
- Hugging Face loading or runtime text/VLM loading.

## 4. Problems in the Current Code

### 4.1 Mixed responsibilities

`FinetuneModel` currently owns base-model delegation, prompt conversion, router
input normalization, adapter injection, decoding, and class prediction.
`FinetuneTrainer` owns iteration, device transfer, loss construction, logging,
checkpoint serialization, and execution policy.

### 4.2 Unused class output

The class head is trained in `FinetuneModel.forward`, but `SinglePredictor` and
`GridPredictor` discard it. The trained classifier therefore cannot be consumed by
inference or saved in result JSON.

### 4.3 Loss disconnects

- Brier mask loss has weak logit gradients for confidently wrong pixels.
- IoU loss exists but is not used, although the predictor publishes decoder IoU as
  its mask score.
- Automatic-background class weighting currently uses the SAM object score. That
  score is not a validated SEM particle/background classifier.
- Confirmed and automatically sampled backgrounds both skip mask loss, which is
  desired, but the reason is implicit in the `has_mask` flag.
- Per-rank local means would give biased DDP gradients when valid mask or class
  counts differ between ranks.

### 4.4 Module and file structure

The three small files under `src/finetune/layers/` do not need an intermediate
folder. LoRA modules are also registered through both the wrapped decoder and
`linear_layers`, creating unnecessary duplicate module paths.

### 4.5 No practical distributed entrypoint or resume loader

The trainer can save partial state but has no strict loader, no stable format
version, no distributed sampler, no rank-aware side effects, and no runnable
`torchrun` entrypoint.

## 5. Target Structure

```text
src/finetune/
  __init__.py
  adapter.py
  router.py
  prompt.py
  model.py
  loss.py
  checkpoint.py
  ddp.py
  trainer.py

scripts/
  finetune.py

config/
  finetune.yaml
```

Responsibilities:

- `adapter.py`: expert feature adapters and expert LoRA linear layers.
- `router.py`: image/condition/prompt input to normalized expert weights.
- `prompt.py`: dataset prompt dictionaries to SAM prompt tensors.
- `model.py`: frozen SAM3.1, adapter injection, forward/decode, and class head.
- `loss.py`: mask BCE, Dice, IoU regression, class BCE, and DDP-safe reduction.
- `checkpoint.py`: atomic save, strict trainable-state load, optimizer resume.
- `ddp.py`: process-group state, rank helpers, and scalar reductions.
- `trainer.py`: train/validation steps and run scheduling.
- `scripts/finetune.py`: YAML loading, construction, optional resume, and cleanup.

`src/data/dataloader.py` receives only the minimal sampler changes required for
DDP. `src/predict/single.py` and `src/predict/grid.py` receive only the changes
required to propagate class output.

## 6. Adapter and Router Mathematics

For image feature map `x`, expert feature adapter `e`, and normalized router weight
`a_e`, compute

```text
x' = x + (alpha / rank) * sum_e a_e * U_e(D_e(x))
```

where `D_e` and `U_e` are bias-free 1x1 convolutions. Every `U_e` starts at zero,
so the initial adapted feature equals the frozen base feature exactly.

For a frozen linear layer `W x + b`, expert LoRA computes

```text
y = W x + b + (alpha / rank) * sum_e a_e * B_e(A_e(x))
```

`A_e` uses Kaiming initialization and `B_e` starts at zero. The existing default
scale is preserved unless a numerical test proves it wrong. The base linear weight
and bias remain frozen.

The router uses

```text
router_input = concat(global_average_pool(image_feature), condition_embedding,
                      prompt_type_embedding)
expert_weights = softmax(router_mlp(router_input), dim=-1)
```

Every sample receives one non-negative expert-weight vector that sums to one.
The same vector is passed explicitly to all feature adapters and decoder LoRA
layers. No mutable `set_mix` state is used.

## 7. Model Behavior

### 7.1 Frozen base

- All SAM3.1 parameters have `requires_grad=False`.
- Calling `FinetuneModel.train()` keeps the SAM3.1 base in eval mode.
- Router, adapters, and class head remain trainable.
- Image and prompt encoding run without gradient tracking.
- Decoder computation still tracks gradients through adapters and LoRA.
- Wrapped LoRA modules are registered only through their decoder positions. A
  second `linear_layers` module list is not kept.

### 7.2 Class head

The class head is a linear projection of each returned mask token:

```text
class_logits = class_head(mask_tokens)  # [B, M, C]
class_scores = sigmoid(class_logits)
```

Index 0 is object/particle presence. Indices 1 through `C - 1` are independent
binary attributes. Softmax and single-class `class_id` semantics are not used.

The SAM mask decoder already returns mask tokens aligned with selected masks:

- single-mask decode: one mask and one token;
- multimask decode: each selected mask has its corresponding token.

The finetune decode path adds class logits without changing mask or IoU values.
At zero initialization, mask and IoU output must match the base image model.

### 7.3 Training output

The model returns tensors with stable names and shapes:

```text
mask_logits   [B, 1, H, W]
iou_scores    [B, 1]
class_logits  [B, 1, C]
```

Training uses single-mask decode. Prediction may return `M > 1`.

## 8. Data Meaning

The training boundary uses `mask_valid` instead of the ambiguous `has_mask` name.

- Confirmed particle: `mask_valid=True`, class index 0 target 1.
- Confirmed background/non-particle: `mask_valid=False`, class index 0 target 0
  with weight 1.
- Automatically sampled background: `mask_valid=False`, `is_auto_bg=True`, class
  index 0 target 0 with a confidence-derived weight.
- When class index 0 is zero, later attribute weights are zero.

Background masks are not optimized. Background is learned by class index 0. This
keeps segmentation focused on confirmed particle masks and leaves filtering to the
class result.

The finetune entrypoint requires class targets, class weights, and conditions with
dimensions matching the configured model. Missing or inconsistent fields fail
before the first optimizer step.

## 9. Loss Design

### 9.1 Mask BCE

For each valid particle mask, use pixel-mean binary cross entropy with logits:

```text
L_mask_bce(i) = mean_pixels BCEWithLogits(z_i, y_i)
```

The resized target may contain soft boundary values, which BCEWithLogits supports.
Unlike Brier loss, its logit gradient remains useful for confidently wrong pixels.

### 9.2 Soft Dice

For probability `q = sigmoid(z)`:

```text
L_dice(i) = 1 - (2 * sum(q * y) + 1) / (sum(q) + sum(y) + 1)
```

Only samples with `mask_valid=True` contribute to mask BCE and Dice.

### 9.3 IoU regression

The actual binary IoU is computed without gradients from

```text
predicted_mask = mask_logits > 0
target_mask = target > 0.5
```

The IoU score loss is mean squared error between decoder IoU prediction and actual
IoU. Only valid particle masks contribute. This keeps the predictor's published
mask score calibrated as adapters change the masks.

### 9.4 Independent class BCE

Use elementwise BCEWithLogits for every active class attribute. Dataset weights
select known attributes.

For automatically sampled background only, define

```text
p = sigmoid(class_logits[..., 0]).detach()
auto_background_weight = 1 - p
```

Multiply the first class weight by this value. A candidate the class head considers
likely to be a particle receives less negative pressure; a likely background
candidate receives more. `detach()` prevents the model from reducing loss by
changing the weight through its gradient path.

Confirmed background samples keep first-class weight 1 and prevent the uncertain
automatic-background branch from becoming an all-particle escape route. SAM3.1's
original object score is not used as a training target or weight.

### 9.5 Total objective

Start with equal explicit component weights:

```text
L = L_mask_bce + L_dice + L_iou + L_class
```

No extra component-weight configuration is introduced before training evidence
shows it is necessary. Each component is logged separately.

### 9.6 Correct DDP normalization

Let local component sum on rank `r` be `S_r`, global active count or weight be
`N`, and world size be `W`. Each rank backpropagates

```text
L_r = W * S_r / max(N, 1)
```

PyTorch DDP averages gradients over `W`, producing the same gradient as

```text
sum_r S_r / max(N, 1)
```

on one global batch. Numerators and denominators are all-reduced for logging.
If the global denominator is zero, that component returns a graph-connected zero.

## 10. Predictor and JSON Output

Plain `Sam3ImageModel` prediction keeps its existing output keys. A finetune model
adds:

```text
class_logits
class_scores
```

The class dimensions follow the score/mask axes:

- one image with multiple masks: `[M, C]`;
- multiple prompts with one mask each: values are flattenable to `[N, C]` in the
  grid pipeline.

`GridPredictor` attaches the matching class-score vector to every candidate and
replaces it with the refined prediction's vector after refinement.

Finetune JSON examples store probabilities, not logits:

```json
{
  "metrics": {
    "score": 0.91,
    "class_scores": [0.97, 0.12, 0.81]
  }
}
```

The existing sample schema already allows list-valued metrics, so the schema
version does not change. `class_id` remains unused because outputs are multi-label.

## 11. Configuration and Entrypoint

Merge model, data, and training configuration into one file:

```text
config/finetune.yaml
  model
  data.train
  data.valid
  train
```

Remove `config/finetune_model.yaml`. The YAML contains relative or user-supplied
paths only; repository-local absolute paths are not written into config.

The entrypoint accepts only the essential overrides:

```bash
python scripts/finetune.py --config config/finetune.yaml

torchrun --standalone --nproc-per-node=4 scripts/finetune.py \
  --config config/finetune.yaml

torchrun --standalone --nproc-per-node=4 scripts/finetune.py \
  --config config/finetune.yaml \
  --resume run/example/checkpoints/last.pt
```

Without torchrun environment variables, the same code runs in one process.

## 12. DDP Design

- Read `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` from torchrun.
- Set the local CUDA device before constructing the model.
- Use NCCL for CUDA and Gloo for CPU tests.
- Wrap the finetune model with `DistributedDataParallel` and
  `find_unused_parameters=False`.
- Use `DistributedSampler` for both training and validation.
- Shuffle only training data and call `sampler.set_epoch` at each iterator reset.
- Let all ranks perform forward, backward, optimizer step, and validation.
- All-reduce loss numerators, denominators, and validation statistics.
- Only rank 0 creates tqdm, TensorBoard writer, run directories, and checkpoints.
- Destroy the process group in a `finally` path.

The initial target is one Linux server with multiple CUDA GPUs. Environment-based
torchrun setup also permits later multi-node execution, but cluster scheduler setup
is not part of this work.

## 13. Checkpoint and Resume

The only supported format is:

```text
format: sam3.finetune.v1
step: integer global step
model: trainable parameter tensors
optimizer: optimizer state dict
config: resolved model/data/train configuration
```

Checkpoint rules:

- Unwrap DDP before enumerating named parameters.
- Save only parameters whose `requires_grad` is true.
- Do not write `module.` prefixes.
- On load, compare the exact expected key set before copying tensors.
- Validate every parameter shape.
- Use the optimizer's native state loader after model validation.
- Return the restored global step to the trainer.
- Do not restore RNG or DataLoader position.
- Each rank reads the same resume file on the single-server filesystem.
- Only rank 0 saves.
- Save `last.pt` through a temporary file and atomic replace.
- Save interval files as `step-000010.pt` and similar.

Run output uses

```text
run/<run-id>/
  checkpoints/
    last.pt
    step-000010.pt
  log/
    events...
```

This avoids confusing run checkpoints with the protected root `weight/` directory.

## 14. Error Handling

Fail early and directly for:

- non-positive rank, expert count, condition count, or class count;
- condition indices outside the configured embedding range;
- target/weight dimensions that do not equal the class-head dimension;
- inconsistent mask, IoU, mask-token, and class axes;
- missing checkpoint fields or unsupported format version;
- missing, extra, or shape-mismatched trainable checkpoint tensors;
- non-finite loss on any rank.

A synchronized finite flag is reduced before backward so all ranks stop together.
An empty global mask set produces zero mask/Dice/IoU loss while class training
continues.

Do not add broad exception wrappers or silent checkpoint fallbacks.

## 15. Verification

### 15.1 Adapter and router math

- Compare expert adapter output with an explicit weighted-sum calculation.
- Verify router weights are non-negative and sum to one per sample.
- Verify zero-initialized adapters preserve base mask and IoU output.
- Verify base parameters stay frozen and the base remains in eval mode.
- Verify LoRA modules have only one registered parameter path.

### 15.2 Loss math

- Check mask BCE and Dice against hand-computed small tensors.
- Check binary IoU thresholding and MSE.
- Check confirmed particle, confirmed background, and automatic-background cases.
- Check that automatic-background weights use class index 0 and are detached.
- Check zero behavior when no valid masks exist.
- Compare simulated multi-rank normalization with a single global batch.

### 15.3 Model and predictor

- Verify single-mask and multimask class axes align with masks and IoU scores.
- Verify plain SAM prediction has no class keys.
- Verify finetune single prediction contains logits and probabilities.
- Verify grid candidates keep refined class scores.
- Verify `metrics["class_scores"]` survives JSON round-trip.

### 15.4 Checkpoint and DDP

- Train a tiny model for one step, save, mutate state, and strictly resume model,
  optimizer, and step.
- Reject missing, extra, and shape-mismatched trainable tensors.
- Run a two-process CPU/Gloo smoke test with a tiny model.
- Verify DDP parameters remain equal and only rank 0 writes a checkpoint.
- Verify single-process execution without process-group initialization.

### 15.5 Project verification

- Run the full pytest suite.
- Run Black check and Ruff.
- Strictly load the existing local SAM3.1 checkpoint.
- Run image inference and confirm base mask/IoU parity at adapter initialization.
- State explicitly that NCCL multi-GPU execution cannot be verified on the current
  computer.
- Provide the Linux `torchrun` smoke command and expected rank-0 artifacts.

## 16. Compatibility and Safety

- Existing SAM3.1 base checkpoint loading must continue to work.
- Existing plain image inference output remains unchanged.
- Finetune masks and IoU values match the base at zero adapter initialization.
- New class output is additive for finetune prediction.
- Historical finetune/LoRA checkpoint loading is intentionally unsupported.
- Do not modify, stage, commit, or push `sam3-main/`, `weight/`, or `asset/`.
- Do not push unless explicitly requested.
