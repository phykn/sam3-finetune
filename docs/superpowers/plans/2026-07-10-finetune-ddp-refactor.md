# Finetune DDP Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a direct, mathematically checked finetune pipeline with per-mask class scores, strict resume checkpoints, and single-server multi-GPU DDP.

**Architecture:** Flatten adapter and router code, keep SAM3.1 frozen, and make `FinetuneModel` produce aligned mask, IoU, and independent class outputs. Move loss, checkpoint, distributed helpers, and execution policy into focused modules; keep prediction additive and use one YAML-driven entrypoint for single-process and `torchrun` execution.

**Tech Stack:** Python 3, PyTorch, torch.distributed/DDP, NumPy, Pillow, PyYAML, pytest, TensorBoard, Black, Ruff.

**Execution Status:** Completed on `codex/finetune-ddp-refactor`; final verification recorded in the branch history.

## Global Constraints

- Use the workspace `.venv` and run commands from `D:\code\sam3`.
- Load SAM3.1 weights only from explicit local paths, normally under `weight/`.
- Do not use Hugging Face or load the runtime text/VLM encoder.
- Keep code simple and short; avoid unnecessary wrappers, dataclasses, options, lazy imports, and defensive fallbacks.
- Keep the direction one-way: adapters/router/prompt -> model -> loss -> trainer -> predictor/entrypoint.
- Class output index 0 always means object/particle presence; later indices are independent sigmoid attributes.
- Preserve the mixture-of-experts router and zero-initialized adapter behavior.
- Existing SAM3.1 base checkpoint loading and plain image inference must continue to work.
- Historical finetune/LoRA checkpoint loading is intentionally unsupported.
- Resume restores trainable model state, optimizer state, and global step, but not RNG or DataLoader position.
- DDP targets one Linux server with multiple CUDA GPUs through `torchrun`; NCCL cannot be verified on this computer.
- Do not modify, stage, commit, or push `sam3-main/`, `weight/`, or `asset/`.
- Do not push unless explicitly requested.

---

## File Map

- `src/finetune/adapter.py`: `FeatureAdapter` and `LoraLinear` expert residual math.
- `src/finetune/router.py`: `Router` expert weights.
- `src/finetune/prompt.py`: dataset prompt dictionaries to SAM tensors.
- `src/finetune/model.py`: frozen SAM3.1 orchestration and per-mask class head.
- `src/finetune/loss.py`: BCE, Dice, IoU, class weighting, and global normalization.
- `src/finetune/checkpoint.py`: strict trainable state and atomic checkpoint I/O.
- `src/finetune/ddp.py`: rank/world helpers, initialization, reductions, broadcasts.
- `src/finetune/trainer.py`: step/validation/log/checkpoint schedule.
- `src/data/dataloader.py`: train/valid loader construction and distributed sampler reset.
- `src/predict/single.py`: optional class output formatting.
- `src/predict/grid.py`: class-score propagation through candidates and refinement.
- `src/predict/mask/format.py`: tensor-to-NumPy class output conversion.
- `src/build.py`: builders for the unified configuration.
- `scripts/finetune.py`: single-process and torchrun training entrypoint.
- `scripts/finetune_single.py`, `scripts/finetune_grid.py`: class scores in result JSON.
- `config/finetune.yaml`: unified model/data/train configuration.
- `requirements.txt`: explicit PyYAML dependency.
- `README.md`: launch, resume, output, and verification notes.

---

### Task 1: Flatten and Verify Expert Adapters and Router

**Files:**
- Create: `src/finetune/adapter.py`
- Create: `src/finetune/router.py`
- Modify: `src/finetune/model.py`
- Modify: `tests/test_finetune_layers.py`
- Delete: `src/finetune/layers/__init__.py`
- Delete: `src/finetune/layers/image.py`
- Delete: `src/finetune/layers/linear.py`
- Delete: `src/finetune/layers/router.py`

**Interfaces:**
- Produces: `FeatureAdapter(channels, rank, num_experts, alpha=1.0)`.
- Produces: `LoraLinear(base, rank, num_experts, alpha=1.0)`.
- Produces: `Router(image_dim, num_conditions, num_experts, hidden_dim=128, embed_dim=16)`.
- Both adapters consume `mix: Tensor[B, E]` and preserve the input/output base shape.

- [ ] **Step 1: Change imports and tests to the flat modules, then add explicit expert-math assertions**

```python
from src.finetune.adapter import FeatureAdapter, LoraLinear
from src.finetune.router import Router


def test_feature_adapter_matches_weighted_expert_sum():
    adapter = FeatureAdapter(2, rank=1, num_experts=2, alpha=1.0)
    for down, up in zip(adapter.down, adapter.up):
        torch.nn.init.ones_(down.weight)
        torch.nn.init.ones_(up.weight)
    x = torch.ones(2, 2, 1, 1)
    mix = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
    expected = x.clone()
    for expert, (down, up) in enumerate(zip(adapter.down, adapter.up)):
        expected += up(down(x)) * mix[:, expert, None, None, None]
    assert torch.allclose(adapter(x, mix), expected)


def test_lora_linear_matches_weighted_expert_sum():
    base = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.ones_(base.weight)
    layer = LoraLinear(base, rank=1, num_experts=2, alpha=1.0)
    for down, up in zip(layer.down, layer.up):
        torch.nn.init.ones_(down.weight)
        torch.nn.init.ones_(up.weight)
    x = torch.ones(2, 2)
    mix = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
    expected = base(x)
    for expert, (down, up) in enumerate(zip(layer.down, layer.up)):
        expected += up(down(x)) * mix[:, expert, None]
    assert torch.allclose(layer(x, mix), expected)
```

- [ ] **Step 2: Run the focused tests and confirm the flat imports fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_layers.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.finetune.adapter'`.

- [ ] **Step 3: Merge the adapter implementations and move the router**

```python
# src/finetune/adapter.py
import math

import torch
from torch import nn


class FeatureAdapter(nn.Module):
    def __init__(self, channels, rank, num_experts, alpha=1.0):
        super().__init__()
        if rank <= 0 or num_experts <= 0:
            raise ValueError("rank and num_experts must be positive")
        self.scale = alpha / rank
        self.down = nn.ModuleList(
            nn.Conv2d(channels, rank, 1, bias=False) for _ in range(num_experts)
        )
        self.up = nn.ModuleList(
            nn.Conv2d(rank, channels, 1, bias=False) for _ in range(num_experts)
        )
        for up in self.up:
            nn.init.zeros_(up.weight)

    def forward(self, x, mix):
        delta = torch.zeros_like(x)
        for index, (down, up) in enumerate(zip(self.down, self.up)):
            delta = delta + up(down(x)) * mix[:, index, None, None, None]
        return x + delta * self.scale


class LoraLinear(nn.Module):
    def __init__(self, base, rank, num_experts, alpha=1.0):
        super().__init__()
        if rank <= 0 or num_experts <= 0:
            raise ValueError("rank and num_experts must be positive")
        self.base = base
        self.scale = alpha / rank
        self.down = nn.ModuleList(
            nn.Linear(base.in_features, rank, bias=False) for _ in range(num_experts)
        )
        self.up = nn.ModuleList(
            nn.Linear(rank, base.out_features, bias=False)
            for _ in range(num_experts)
        )
        for param in base.parameters():
            param.requires_grad = False
        for down, up in zip(self.down, self.up):
            nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
            nn.init.zeros_(up.weight)

    def forward(self, x, mix=None):
        out = self.base(x)
        if mix is None:
            return out
        delta = torch.zeros_like(out)
        shape = [mix.shape[0]] + [1] * (out.ndim - 1)
        for index, (down, up) in enumerate(zip(self.down, self.up)):
            delta = delta + up(down(x)) * mix[:, index].view(shape)
        return out + delta * self.scale
```

Move `PROMPT_IDS` and `Router` unchanged except for positive constructor checks into
`src/finetune/router.py`, update `src/finetune/model.py` imports, and delete the old
`layers` files.

- [ ] **Step 4: Run focused and package-structure tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_layers.py tests/test_package_structure.py -q`

Expected: all selected tests pass and no test imports `src.finetune.layers`.

- [ ] **Step 5: Commit the flat adapter/router unit**

```bash
git add src/finetune tests/test_finetune_layers.py tests/test_package_structure.py
git commit -m "refactor: flatten finetune adapters"
```

---

### Task 2: Separate Prompt Conversion and Align Class Output Per Mask

**Files:**
- Create: `src/finetune/prompt.py`
- Modify: `src/finetune/model.py`
- Modify: `src/build.py`
- Create: `tests/test_finetune_model.py`
- Modify: `tests/test_builder.py`
- Modify: `tests/test_finetune_layers.py`

**Interfaces:**
- Consumes: `FeatureAdapter`, `LoraLinear`, and `Router` from Task 1.
- Produces: `build_prompt(item, image_size, mask_size, device)`.
- Produces: `FinetuneModel.decode_masks(...) -> (masks, ious, tokens, object_logits, class_logits)`.
- Produces: `FinetuneModel.forward(batch)` with single-mask tensors for later loss integration.

- [ ] **Step 1: Add failing tests for per-mask classes, frozen eval state, and unique LoRA paths**

```python
def test_decode_returns_one_class_vector_per_returned_mask():
    model, base = make_fake_finetune_model(num_labels=3, returned_masks=3)
    out = model.decode_masks(
        torch.ones(1, 256, 2, 2),
        (torch.ones(1, 32, 8, 8), torch.ones(1, 64, 4, 4)),
        base.encode_prompt(points=(torch.zeros(1, 1, 2), torch.ones(1, 1))),
        base.get_image_position_encoding(),
        multimask=True,
        cond=0,
        prompt_type="point",
    )
    assert out[0].shape[1] == 3
    assert out[4].shape == (1, 3, 3)


def test_train_keeps_frozen_base_in_eval_mode():
    model, base = make_fake_finetune_model(num_labels=2, returned_masks=1)
    model.train()
    assert model.training is True
    assert base.training is False
    assert model.router.training is True
    assert model.class_head.training is True


def test_lora_parameters_have_one_named_path():
    model, _base = make_fake_finetune_model(num_labels=2, returned_masks=1)
    names = [name for name, _param in model.named_parameters() if ".down." in name]
    assert names
    assert all(not name.startswith("linear_layers.") for name in names)
```

- [ ] **Step 2: Run the new model tests and confirm class alignment fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_model.py -q`

Expected: failures because `class_head`, five-value decode, and base-eval override do
not exist yet.

- [ ] **Step 3: Extract prompt conversion into pure functions**

```python
# src/finetune/prompt.py
import torch

from ..data import prompt as prompt_data


def build_prompt(item, image_size, mask_size, device):
    points = prompt_data.build_points(
        item["points"], item["point_labels"], (image_size, image_size), image_size, device
    )
    box = prompt_data.build_box(
        item["box"], (image_size, image_size), image_size, device
    )
    if box is None:
        point_prompt = points
    elif points is None:
        point_prompt = box
    else:
        point_prompt = (
            torch.cat([box[0], points[0]], dim=1),
            torch.cat([box[1], points[1]], dim=1),
        )
    mask_prompt = prompt_data.build_mask(item["mask"], mask_size, device)
    if point_prompt is None:
        batch = 1 if mask_prompt is None else mask_prompt.shape[0]
        point_prompt = (
            torch.zeros(batch, 1, 2, device=device),
            -torch.ones(batch, 1, dtype=torch.int, device=device),
        )
    return point_prompt, mask_prompt
```

- [ ] **Step 4: Rewrite model orchestration around the five aligned decode outputs**

Implement these exact behaviors in `src/finetune/model.py`:

```python
def decode_masks(self, image_embed, high_res_features, prompt, image_pe,
                 multimask=True, repeat_image=False, cond=None, prompt_type=None):
    mix = self._mix(image_embed, cond, prompt_type)
    image_embed, high_res = self._adapt(image_embed, high_res_features, mix)
    masks, ious, tokens, objects = self.model.decode_masks(
        image_embed, high_res, prompt, image_pe, multimask, repeat_image, mix=mix
    )
    class_logits = self.class_head(tokens)
    if class_logits.shape[:2] != masks.shape[:2]:
        raise ValueError("class tokens must align with masks")
    return masks, ious, tokens, objects, class_logits


def train(self, mode=True):
    super().train(mode)
    self.model.eval()
    return self
```

Freeze the base before wrapping, rename `label_head` to `class_head`, remove
`linear_layers`, and keep adapters discoverable through their decoder positions.
Use `build_prompt` in the batch loop. Validate positive model dimensions and
condition indices.

- [ ] **Step 5: Update builders and existing tests for `FeatureAdapter` and `class_head`**

Change `src/build.py` and `tests/test_builder.py` to import from the flat modules and
assert `model.class_head.out_features`. Remove old model tests from
`tests/test_finetune_layers.py` after they are represented in
`tests/test_finetune_model.py`.

- [ ] **Step 6: Run model, builder, and core inference tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_model.py tests/test_builder.py tests/test_single_predict.py -q`

Expected: all selected tests pass; plain `SinglePredictor` behavior remains unchanged.

- [ ] **Step 7: Commit the model/prompt boundary**

```bash
git add src/finetune src/build.py tests/test_finetune_model.py tests/test_builder.py tests/test_finetune_layers.py
git commit -m "refactor: simplify finetune model flow"
```

---

### Task 3: Replace and Globally Normalize Finetune Losses

**Files:**
- Create: `src/finetune/ddp.py`
- Modify: `src/finetune/loss.py`
- Modify: `src/finetune/trainer.py`
- Modify: `src/data/dataset.py`
- Modify: `src/data/dataloader.py`
- Create: `tests/test_finetune_loss.py`
- Modify: `tests/test_finetune_trainer.py`
- Modify: `tests/test_data_dataset.py`
- Modify: `tests/test_data_dataloader.py`
- Modify: `tests/test_finetune_layers.py`

**Interfaces:**
- Produces: `world_size() -> int`, `sum_value(tensor) -> tensor`.
- Produces: `mean_loss(local_sum, local_weight) -> tuple[Tensor, float]`.
- Produces: `finetune_loss(batch, out) -> tuple[Tensor, dict[str, float]]`.
- Renames batch field `has_mask` to `mask_valid`.

- [ ] **Step 1: Add failing numerical loss and background-weight tests**

```python
def test_auto_background_uses_detached_particle_probability():
    logits = torch.tensor([[2.0, -1.0], [-2.0, 1.0]], requires_grad=True)
    targets = torch.zeros_like(logits)
    weights = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    auto = torch.tensor([True, False])
    adjusted = class_weights(weights, logits, auto)
    assert torch.allclose(adjusted[0, 0], 1 - logits[0, 0].sigmoid())
    assert adjusted[1, 0] == 1
    assert adjusted.requires_grad is False


def test_background_samples_do_not_contribute_mask_or_iou_loss():
    batch, out = make_loss_batch(mask_valid=[1.0, 0.0])
    total, stats = finetune_loss(batch, out)
    changed = {key: value.clone() for key, value in out.items()}
    changed["mask_logits"][1] = 100
    changed["iou_scores"][1] = 100
    changed_total, changed_stats = finetune_loss(batch, changed)
    assert torch.allclose(total, changed_total)
    assert stats == changed_stats


def test_mask_bce_keeps_gradient_for_confident_wrong_logit():
    logits = torch.tensor([[-10.0]], requires_grad=True)
    loss = mask_bce(logits, torch.ones_like(logits)).sum()
    loss.backward()
    assert logits.grad.item() < -0.9
```

- [ ] **Step 2: Run loss tests and confirm missing interfaces fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_loss.py -q`

Expected: collection fails because `class_weights`, `finetune_loss`, and `mask_bce`
are not implemented.

- [ ] **Step 3: Add minimal distributed sum and world-size helpers**

```python
# src/finetune/ddp.py
import torch.distributed as dist


def active():
    return dist.is_available() and dist.is_initialized()


def world_size():
    return dist.get_world_size() if active() else 1


def sum_value(value):
    out = value.detach().clone()
    if active():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out
```

- [ ] **Step 4: Implement component sums and DDP-correct means**

```python
def mean_loss(local_sum, local_weight):
    weight = torch.as_tensor(local_weight, device=local_sum.device,
                             dtype=local_sum.dtype)
    global_weight = sum_value(weight)
    if global_weight.item() <= 0:
        return local_sum * 0, 0.0
    backward = local_sum * (world_size() / global_weight)
    logged = sum_value(local_sum) / global_weight
    return backward, float(logged.cpu())


def class_weights(weights, logits, is_auto_bg):
    out = weights.detach().clone()
    auto = is_auto_bg.to(device=out.device, dtype=torch.bool).flatten()
    particle = logits[:, 0].detach().sigmoid()
    out[auto, 0] *= 1 - particle[auto]
    return out
```

Implement per-sample pixel-mean BCEWithLogits, soft Dice, binary IoU target at
`mask_logits > 0` and `target > 0.5`, IoU MSE, and weighted class BCE. Compute
four globally normalized terms and return stats named `loss`, `mask_bce`,
`mask_dice`, `iou_loss`, and `class_loss`.

- [ ] **Step 5: Rename the data contract and integrate trainer loss**

Replace `has_mask` with `mask_valid` in dataset items, collate output, tests, and
trainer batches. Preserve the existing rule: confirmed particle is valid; confirmed
background and automatic background are invalid. Replace trainer `_loss` logic with
`finetune_loss(batch, out)` and update tiny test-model output names to
`mask_logits`, `iou_scores`, and `class_logits`.

- [ ] **Step 6: Run loss, trainer, and data tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_loss.py tests/test_finetune_trainer.py tests/test_data_dataset.py tests/test_data_dataloader.py -q`

Expected: all selected tests pass, including the zero-valid-mask case.

- [ ] **Step 7: Commit the corrected objective**

```bash
git add src/finetune src/data tests/test_finetune_loss.py tests/test_finetune_trainer.py tests/test_data_dataset.py tests/test_data_dataloader.py tests/test_finetune_layers.py
git commit -m "fix: correct finetune loss flow"
```

---

### Task 4: Expose Per-Mask Class Scores Through Prediction and JSON

**Files:**
- Modify: `src/predict/mask/format.py`
- Modify: `src/predict/single.py`
- Modify: `src/predict/grid.py`
- Modify: `scripts/finetune_single.py`
- Modify: `scripts/finetune_grid.py`
- Modify: `tests/test_predict_output.py`
- Modify: `tests/test_single_predict.py`
- Modify: `tests/test_grid_predict.py`
- Modify: `tests/test_scripts_standalone.py`

**Interfaces:**
- Consumes: five-value finetune decode from Task 2.
- Produces: optional predictor keys `class_logits` and `class_scores`.
- Produces: grid candidate key `class_scores` only for finetune output.
- Produces: JSON metric `class_scores: list[float]`.

- [ ] **Step 1: Add failing predictor and JSON tests**

```python
def test_finetune_prediction_adds_per_mask_class_scores():
    model = FakeModelWithClasses(classes=torch.tensor([[[2.0, -2.0]]]))
    out = SinglePredictor(model, device="cpu").predict(
        Image.new("RGB", (20, 10)),
        point_coords=np.array([[10.0, 5.0]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask=False,
    )
    assert out["class_logits"].shape == (1, 2)
    np.testing.assert_allclose(
        out["class_scores"], 1 / (1 + np.exp(-out["class_logits"]))
    )


def test_plain_prediction_does_not_add_class_keys():
    out = SinglePredictor(FakeModel(), device="cpu").predict(
        Image.new("RGB", (20, 10)),
        point_coords=np.array([[10.0, 5.0]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
    )
    assert "class_logits" not in out
    assert "class_scores" not in out
```

Add grid tests that initial and refined candidates retain the refined vector, and
script tests that loaded JSON contains `metrics["class_scores"]`.

- [ ] **Step 2: Run predictor tests and confirm class keys are absent**

Run: `.venv\Scripts\python.exe -m pytest tests/test_predict_output.py tests/test_single_predict.py tests/test_grid_predict.py tests/test_scripts_standalone.py -q`

Expected: new class-output assertions fail.

- [ ] **Step 3: Add optional class formatting without changing plain output**

```python
def make_classes(logits):
    logits = logits.squeeze(0).float().detach().cpu()
    return {
        "class_logits": logits.numpy(),
        "class_scores": logits.sigmoid().numpy(),
    }
```

In `SinglePredictor._decode`, retain the full decode tuple, treat element 5 as
class logits only when present, and merge `make_classes` into the formatted output.
Do not add `None` class keys for plain models.

- [ ] **Step 4: Carry class rows through grid creation and refinement**

Normalize class arrays with

```python
def _class_rows(out, count):
    values = out.get("class_scores")
    if values is None:
        return [None] * count
    values = np.asarray(values).reshape(count, -1)
    return [row.copy() for row in values]
```

Attach rows to candidates only when non-`None`. During refinement, overwrite the
old row with the refined row.

- [ ] **Step 5: Save probabilities in finetune example JSON**

Set

```python
metrics = {"score": float(score)}
if class_scores is not None:
    metrics["class_scores"] = np.asarray(class_scores, dtype=float).tolist()
```

in both finetune JSON packers. Keep `class_id=None`.

- [ ] **Step 6: Run predictor and script tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_predict_output.py tests/test_single_predict.py tests/test_grid_predict.py tests/test_scripts_standalone.py -q`

Expected: all selected tests pass and plain prediction fixtures remain unchanged.

- [ ] **Step 7: Commit additive class output**

```bash
git add src/predict scripts/finetune_single.py scripts/finetune_grid.py tests/test_predict_output.py tests/test_single_predict.py tests/test_grid_predict.py tests/test_scripts_standalone.py
git commit -m "feat: expose finetune class scores"
```

---

### Task 5: Add Strict Atomic Checkpoint and General Resume

**Files:**
- Create: `src/finetune/checkpoint.py`
- Create: `tests/test_finetune_checkpoint.py`
- Modify: `src/finetune/trainer.py`
- Modify: `tests/test_finetune_trainer.py`

**Interfaces:**
- Produces: `trainable_state(model) -> dict[str, Tensor]`.
- Produces: `save_checkpoint(path, model, optimizer, step, config) -> None`.
- Produces: `load_checkpoint(path, model, optimizer) -> tuple[int, dict]`.
- Checkpoint format constant: `sam3.finetune.v1`.

- [ ] **Step 1: Add round-trip and strict-failure tests**

```python
def test_checkpoint_restores_trainable_state_optimizer_and_step(tmp_path):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_once(model, optimizer)
    expected = {name: value.clone() for name, value in trainable_state(model).items()}
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, step=7, config={"train": {"steps": 9}})
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.zero_()
    step, config = load_checkpoint(path, model, optimizer)
    assert step == 7
    assert config == {"train": {"steps": 9}}
    for name, value in trainable_state(model).items():
        assert torch.equal(value, expected[name])


def test_checkpoint_rejects_missing_trainable_key(tmp_path):
    path = write_checkpoint_without_one_key(tmp_path)
    with pytest.raises(ValueError, match="trainable keys"):
        load_checkpoint(path, TinyModel(), make_optimizer())
```

Add equivalent extra-key, wrong-shape, and unsupported-format tests.

- [ ] **Step 2: Run checkpoint tests and confirm the module is missing**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_checkpoint.py -q`

Expected: collection fails for missing `src.finetune.checkpoint`.

- [ ] **Step 3: Implement strict trainable parameter state**

```python
FORMAT = "sam3.finetune.v1"


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def trainable_state(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in unwrap(model).named_parameters()
        if param.requires_grad
    }


def load_trainable_state(model, state):
    expected = {
        name: param
        for name, param in unwrap(model).named_parameters()
        if param.requires_grad
    }
    if set(state) != set(expected):
        raise ValueError("checkpoint trainable keys do not match model")
    with torch.no_grad():
        for name, param in expected.items():
            value = state[name]
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(f"checkpoint shape mismatch: {name}")
            param.copy_(value.to(device=param.device, dtype=param.dtype))
```

- [ ] **Step 4: Implement atomic save and resume**

Create parent directories, save to `path.with_suffix(path.suffix + ".tmp")`, and
replace with `Path.replace(path)`. Store `format`, `step`, `model`, `optimizer`, and
`config`. On load, require every field, validate format before model copy, load the
optimizer state, and return integer step plus config.

- [ ] **Step 5: Make trainer save `last.pt` and interval files through the module**

Replace trainer-local serialization with `save_checkpoint`. Use
`run_dir/checkpoints/last.pt` and `step-{step:06d}.pt`. Accept resolved config in the
trainer constructor and initialize `self.step` from the entrypoint after resume.

- [ ] **Step 6: Run checkpoint and trainer tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_checkpoint.py tests/test_finetune_trainer.py -q`

Expected: round-trip, strict failures, atomic path cleanup, and interval save tests pass.

- [ ] **Step 7: Commit checkpoint/resume support**

```bash
git add src/finetune/checkpoint.py src/finetune/trainer.py tests/test_finetune_checkpoint.py tests/test_finetune_trainer.py
git commit -m "feat: add strict finetune resume"
```

---

### Task 6: Add DDP Runtime, Distributed Loaders, and Rank-Aware Trainer

**Files:**
- Modify: `src/finetune/ddp.py`
- Modify: `src/data/dataloader.py`
- Modify: `src/build.py`
- Modify: `src/finetune/trainer.py`
- Create: `tests/test_finetune_ddp.py`
- Modify: `tests/test_data_dataloader.py`
- Modify: `tests/test_builder.py`
- Modify: `tests/test_finetune_trainer.py`

**Interfaces:**
- Produces: `init() -> torch.device`, `rank()`, `local_rank()`, `is_main()`, `finish()`.
- Produces: `broadcast_object(value)`, `all_finite(value)`.
- Produces: `make_finetune_loader(config, train, rank=0, world_size=1)`.
- `InfiniteLoader` advances `DistributedSampler.set_epoch` on reset.

- [ ] **Step 1: Add single-process and two-process distributed tests**

```python
def test_ddp_helpers_default_to_single_process():
    assert ddp.active() is False
    assert ddp.rank() == 0
    assert ddp.world_size() == 1
    assert ddp.is_main() is True


def test_infinite_loader_advances_distributed_sampler_epoch():
    sampler = FakeSampler()
    loader = InfiniteLoader([[1]], sampler=sampler)
    assert next(loader) == [1]
    assert next(loader) == [1]
    assert sampler.epochs == [1]
```

Create a spawned two-process Gloo worker that initializes a file-based process
group, runs one SGD step on a tiny DDP model, gathers parameters, and lets only rank
0 create a marker file. Assert gathered parameters are equal and exactly one marker
exists.

- [ ] **Step 2: Run DDP tests and confirm missing helpers fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_ddp.py tests/test_data_dataloader.py -q`

Expected: failures for missing initialization, rank, finite, and sampler-epoch behavior.

- [ ] **Step 3: Implement environment-based process-group lifecycle**

```python
def init():
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        backend = "nccl"
        device = torch.device("cuda", local)
    else:
        backend = "gloo"
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, init_method="env://")
    return device


def all_finite(value):
    flag = torch.tensor(int(torch.isfinite(value).all()), device=value.device)
    if active():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())
```

Add direct `rank`, `local_rank`, `is_main`, `broadcast_object`, and guarded
`finish` functions. Keep all functions no-op safe outside torchrun.

- [ ] **Step 4: Add train/valid DistributedSampler construction**

Build `TrainDataset` for train and `ValidDataset` for validation. When
`world_size > 1`, construct `DistributedSampler(dataset, num_replicas=world_size,
rank=rank, shuffle=train, drop_last=train)`, set DataLoader shuffle only when no
sampler, and let `InfiniteLoader` increment its epoch and call `set_epoch` whenever
it resets.

- [ ] **Step 5: Make trainer side effects rank-aware and failures synchronized**

Create run/checkpoint/log directories, tqdm, SummaryWriter, and checkpoint files
only on `ddp.is_main()`. All ranks still train and validate. Call `ddp.all_finite`
before backward and raise `FloatingPointError` on every rank when false. Keep
component statistics already global from Task 3.

- [ ] **Step 6: Run DDP, loader, trainer, and builder tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_ddp.py tests/test_data_dataloader.py tests/test_finetune_trainer.py tests/test_builder.py -q`

Expected: single-process tests and two-process CPU/Gloo smoke pass.

- [ ] **Step 7: Commit DDP runtime support**

```bash
git add src/finetune/ddp.py src/data/dataloader.py src/build.py src/finetune/trainer.py tests/test_finetune_ddp.py tests/test_data_dataloader.py tests/test_finetune_trainer.py tests/test_builder.py
git commit -m "feat: add finetune ddp runtime"
```

---

### Task 7: Add Unified YAML Configuration and Training Entrypoint

**Files:**
- Modify: `config/finetune.yaml`
- Delete: `config/finetune_model.yaml`
- Create: `scripts/finetune.py`
- Modify: `src/build.py`
- Modify: `requirements.txt`
- Create: `tests/test_finetune_script.py`
- Modify: `tests/test_scripts_standalone.py`
- Modify: `tests/test_package_structure.py`

**Interfaces:**
- Produces CLI: `scripts/finetune.py --config PATH [--resume PATH]`.
- Config sections: `model`, `data.train`, `data.valid`, `train`.
- `train.steps` is the final global step, not additional steps after resume.

- [ ] **Step 1: Add failing config and entrypoint-construction tests**

```python
def test_finetune_config_has_one_complete_tree():
    config = yaml.safe_load(Path("config/finetune.yaml").read_text(encoding="utf-8"))
    assert set(config) == {"model", "data", "train"}
    assert set(config["data"]) == {"train", "valid"}
    assert "path" in config["model"]
    assert "steps" in config["train"]


def test_resume_run_dir_is_checkpoint_run_dir(tmp_path):
    path = tmp_path / "run-a" / "checkpoints" / "last.pt"
    assert finetune.resume_run_dir(path) == tmp_path / "run-a"
```

Add a construction test with monkeypatched model, loaders, optimizer, trainer,
checkpoint loader, and DDP wrapper. Assert resume loads before DDP wrapping and the
restored step reaches the trainer.

- [ ] **Step 2: Run entrypoint tests and confirm script/config failures**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_script.py tests/test_scripts_standalone.py tests/test_package_structure.py -q`

Expected: missing `scripts.finetune` and old split config assertions fail.

- [ ] **Step 3: Add PyYAML and replace the split config with one tree**

Add `pyyaml` to `requirements.txt`. Use this exact top-level structure:

```yaml
model:
  path: null
  num_conditions: 1
  num_experts: 4
  num_classes: 3
  lora_rank: 8
  feature_rank: 16
data:
  train:
    paths: []
    conds: []
    labels: []
    batch_size: 4
    num_workers: 4
  valid:
    paths: []
    conds: []
    labels: []
    batch_size: 4
    num_workers: 4
train:
  steps: 1000
  valid_steps: 10
  learning_rate: 0.0001
  save_every: 100
  clip_grad_norm: 1.0
  amp: true
  run_root: run
```

Rename builder configuration from `num_labels` to `num_classes` consistently.

- [ ] **Step 4: Implement the essential CLI and launch order**

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def resume_run_dir(path):
    return Path(path).resolve().parent.parent
```

In `main`: initialize DDP, load YAML, build local-rank model and loaders, create
AdamW over `model.trainable_parameters()`, load resume before wrapping, wrap with
`DistributedDataParallel` only when active, broadcast a rank-0 timestamped run
directory for new runs, construct the trainer with restored step, run training in a
`try`, close the trainer, and call `ddp.finish()` in `finally`.

- [ ] **Step 5: Run entrypoint and standalone script tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_script.py tests/test_scripts_standalone.py tests/test_package_structure.py -q`

Expected: all selected tests pass and importing the script performs no training.

- [ ] **Step 6: Commit the runnable training interface**

```bash
git add config requirements.txt scripts/finetune.py src/build.py tests/test_finetune_script.py tests/test_scripts_standalone.py tests/test_package_structure.py
git commit -m "feat: add finetune training entrypoint"
```

---

### Task 8: Integration, Documentation, and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `src/finetune/__init__.py`
- Modify: structure and integration tests as required by final public names
- Verify only: `weight/sam3.1_multiplex.pt`

**Interfaces:**
- Documents single-process, torchrun, resume, class-score JSON, and server checks.
- Exports only public finetune construction/training names that callers need.

- [ ] **Step 1: Add final architecture and integration assertions**

Add tests that assert:

```python
assert not Path("src/finetune/layers").exists()
assert Path("src/finetune/checkpoint.py").is_file()
assert Path("src/finetune/ddp.py").is_file()
assert Path("scripts/finetune.py").is_file()
assert not Path("config/finetune_model.yaml").exists()
```

Add one fake end-to-end test from batch -> `FinetuneModel.forward` ->
`finetune_loss` -> backward that asserts every trainable parameter used by the graph
has a finite gradient.

- [ ] **Step 2: Run the integration tests and fix only uncovered contract gaps**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_model.py tests/test_finetune_loss.py tests/test_finetune_checkpoint.py tests/test_finetune_ddp.py tests/test_finetune_script.py tests/test_package_structure.py -q`

Expected: all selected tests pass.

- [ ] **Step 3: Update README with exact public behavior and commands**

Document:

```bash
python scripts/finetune.py --config config/finetune.yaml
torchrun --standalone --nproc-per-node=4 scripts/finetune.py --config config/finetune.yaml
torchrun --standalone --nproc-per-node=4 scripts/finetune.py --config config/finetune.yaml --resume run/example/checkpoints/last.pt
```

State that class index 0 is particle presence, later indices are independent
attributes, JSON uses `metrics.class_scores`, old LoRA checkpoints are unsupported,
and NCCL multi-GPU remains to be run on the Linux server.

- [ ] **Step 4: Run the narrow finetune suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_finetune_layers.py tests/test_finetune_model.py tests/test_finetune_loss.py tests/test_finetune_trainer.py tests/test_finetune_checkpoint.py tests/test_finetune_ddp.py tests/test_finetune_script.py tests/test_single_predict.py tests/test_grid_predict.py tests/test_scripts_standalone.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Run the full suite and formatting checks**

Run: `.venv\Scripts\python.exe -m pytest tests -q`

Expected: all tests pass.

Run: `.venv\Scripts\python.exe -m black --check src tests scripts`

Expected: `All done!` with no files requiring changes.

Run: `.venv\Scripts\python.exe -m ruff check src tests scripts`

Expected: `All checks passed!`.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 6: Verify the real local SAM3.1 checkpoint and zero-init parity**

Run a short script through `.venv\Scripts\python.exe` that loads
`weight/sam3.1_multiplex.pt` into both `Sam3ImageModel` and `FinetuneModel`, encodes
one local test image, decodes the same prompt with adapters at zero initialization,
and prints maximum absolute mask and IoU differences.

Expected:

```text
checkpoint: strict load passed
mask max abs diff: 0.0
iou max abs diff: 0.0
class shape: (1, 1, C)
```

- [ ] **Step 7: Record the Linux server smoke command**

Document this command without claiming it ran locally:

```bash
torchrun --standalone --nproc-per-node=2 scripts/finetune.py \
  --config config/finetune.yaml
```

Expected server evidence: both ranks initialize NCCL, one rank-0 TensorBoard log is
created, one `checkpoints/last.pt` is written, and resume continues from its stored
global step.

- [ ] **Step 8: Inspect final scope and commit integration docs**

Run: `git status --short`

Expected: no entries under `sam3-main/`, `weight/`, or `asset/`.

```bash
git add README.md src tests scripts config requirements.txt docs/superpowers/plans/2026-07-10-finetune-ddp-refactor.md
git commit -m "docs: document finetune ddp workflow"
```

---

## Completion Criteria

- All eight task commits exist on `codex/finetune-ddp-refactor`.
- Existing SAM3.1 local checkpoint loading and plain image prediction still work.
- Zero-initialized adapters preserve base mask and IoU output.
- Mask, Dice, IoU, and class losses match the approved equations.
- Confirmed background uses full class-negative weight; automatic background uses
  detached `1 - particle_probability`; neither contributes mask or IoU loss.
- Per-mask class scores reach SinglePredictor, GridPredictor, refinement, and JSON.
- New-format checkpoint resumes trainable state, optimizer, and global step strictly.
- Single-process and two-process CPU/Gloo tests pass.
- Full pytest, Black, Ruff, and diff checks pass.
- NCCL multi-GPU is clearly marked as pending Linux-server verification.
