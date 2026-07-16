# SAM3 Finetune

A local-checkpoint SAM 3.1 rewrite for image segmentation, reference grounding,
video propagation, and LoRA finetuning. Runtime code lives under `src/` and does
not depend on Hugging Face.

## Repository

```text
config/            model and finetuning configuration templates
finetune_dataset/  small frog/leaf finetuning dataset and previews
notebooks/         dataset, forward, loss, and grid-prediction checks
src/data/          dataset, augmentation, prompts, and JSON sample format
src/finetune/      LoRA model, router, loss, checkpoint, DDP, and trainer
src/ml/            SAM 3.1 components -> workflow blocks -> assembled models
src/predict/       single-image, grid, grounding, and video predictors
tests/             runtime and mathematical regression tests
```

`weight/visual_token.pt` is included for no-text grounding. `asset/`, `docs/`,
`scripts/`, `outputs/`, other weight files, and the upstream `sam3-main/`
checkout are local-only and are not part of the remote repository.

## Setup

Create a virtual environment and install the repository dependencies:

```bash
python -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\python.exe` instead. Install `torch` and
`torchvision` separately using the CUDA wheel that matches the machine.

Place the official checkpoint at an explicit local path, for example:

```text
weight/sam3.1_multiplex.pt
```

Weights are never downloaded automatically.

## Dataset

Samples use the `sam3.sample.v2` JSON schema. Images and object ROIs are stored
as base64 PNG data, so the included example dataset has no external image-file
dependency.

The finetuning loader accepts class folders:

```yaml
data:
  train:
    folders:
      - path: finetune_dataset/train/0_background
        cond: 0
        target: [0, 0, 0]
        weight: [1, 0, 0]
      - path: finetune_dataset/train/1_frog
        cond: 0
        target: [1, 1, 0]
        weight: [1, 1, 1]
      - path: finetune_dataset/train/2_leaf
        cond: 1
        target: [1, 0, 1]
        weight: [1, 1, 1]
    batch_size: 2
    num_workers: 0
```

Class-head outputs are independent sigmoid attributes per mask. Index `0` is
object presence. Later indices are task-defined attributes; in the included
dataset they distinguish frog and leaf samples.

Direct `paths` configurations must provide one `cond` and one label entry per
path. Conditions must be valid model condition indices; label targets are in
`[0, 1]` and label weights are finite non-negative values.

```python
from src.build import build_finetune_loader

loader = build_finetune_loader(
    config["data"]["train"],
    num_classes=config["model"]["num_classes"],
    num_conditions=config["model"]["num_conditions"],
    train=True,
)
batch = next(loader)
```

Training and prediction both resize images directly to `1008 x 1008`; target
masks and decoder logits are `288 x 288`.
Background samples have `mask_valid=0` and train the class head without applying
mask or IoU loss. Automatically sampled background prompts focus their presence
loss on high-confidence false positives.

## LoRA Finetuning

`build_finetune_model` loads the base checkpoint, freezes SAM, and adds the LoRA
experts, feature adapters, router, and class head.

```python
from pathlib import Path

import torch
import yaml

from src.build import build_finetune_loader, build_finetune_model
from src.finetune.trainer import FinetuneTrainer

config = yaml.safe_load(Path("config/finetune.yaml").read_text())
config["model"]["path"] = "weight/sam3.1_multiplex.pt"
config["model"]["device"] = "cuda"

model = build_finetune_model(config["model"])
num_classes = config["model"]["num_classes"]
num_conditions = config["model"]["num_conditions"]
train_loader = build_finetune_loader(
    config["data"]["train"],
    num_classes=num_classes,
    num_conditions=num_conditions,
    train=True,
)
valid_loader = build_finetune_loader(
    config["data"]["valid"],
    num_classes=num_classes,
    num_conditions=num_conditions,
    train=False,
)
optimizer = torch.optim.AdamW(
    model.trainable_parameters(),
    lr=config["train"]["learning_rate"],
)

trainer = FinetuneTrainer(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    steps=config["train"]["steps"],
    valid_steps=config["train"]["valid_steps"],
    device="cuda",
    run_root=config["train"]["run_root"],
    save_every=config["train"]["save_every"],
    clip_grad_norm=config["train"]["clip_grad_norm"],
    amp=config["train"]["amp"],
    config=config,
)
try:
    stats = trainer.train()
finally:
    trainer.close()
```

Only trainable parameters, optimizer state, the completed global step, and the
configuration are stored in `sam3.finetune.v1` checkpoints. Resume before
constructing DDP and pass the restored step to the trainer:

```python
from pathlib import Path

from src.finetune.checkpoint import load_checkpoint

checkpoint_path = Path("run/example/checkpoints/last.pt")
step, _saved_config = load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    config,
)
trainer = FinetuneTrainer(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    steps=config["train"]["steps"],
    valid_steps=config["train"]["valid_steps"],
    device="cuda",
    step=step,
    run_dir=checkpoint_path.parent.parent,
    save_every=config["train"]["save_every"],
    clip_grad_norm=config["train"]["clip_grad_norm"],
    amp=config["train"]["amp"],
    config=config,
)
```

Pressing `Ctrl+C` during `trainer.train()` writes the latest completed step to
`last.pt` before exiting. Resume continues from that step but does not restore
random-number or sampler state. Model, data, and optimization settings must
match the checkpoint; `steps`, `valid_steps`, `save_every`, `run_root`, and the
runtime device may change. The new `steps` value must not be lower than the
checkpoint step.

The DDP helpers under `src.finetune.ddp` target single-server multi-GPU training.
CPU/Gloo behavior is covered by tests; NCCL execution must be verified on the
target Linux server.

## LoRA Grid Prediction

Load the base checkpoint and finetuning checkpoint directly into a predictor:

```python
from src.predict import GridPredictor

predictor = GridPredictor.from_finetune(
    "weight/sam3.1_multiplex.pt",
    "run/example/checkpoints/last.pt",
    device="cuda",
    cond=0,
    tiles=(1, 2),
    points_per_side=(10, 10),
    batch_size=4,
)
objects = predictor.predict(image)
```

Use condition `0` for frog and `1` for leaf with the included example mapping.
Grid objects contain mask geometry, quality metrics, and class logits/scores when
the underlying model is a `FinetuneModel`. Grid proposals are filtered by raw
predicted IoU and object-presence score before refinement; tune `iou_thr` and
`presence_thr` when the defaults are not appropriate for a dataset.

## Other Prediction APIs

Public predictor entry points are exported from `src.predict`:

- `SinglePredictor`: point, box, or mask-prompt image segmentation
- `GridPredictor`: automatic grid proposals with refinement and NMS
- `GroundPredictor`: reference-box visual grounding
- `VideoPredictor`: forward video-mask propagation with explicit add/remove APIs

Model builders are exported from `src.build`:

- `build_image_model`
- `build_grounding_model`
- `build_video_model`
- `build_finetune_model`
- `build_finetune_loader`

All builders load weights only from explicit local paths.

`SinglePredictor` and `GroundPredictor` expose `encode()` plus `predict_embed()`
for reusing image features. Grounding decodes reference classes in bounded
prompt batches, and Single/Grid accept both PIL and NumPy images.

## Notebooks

The committed notebooks contain executed outputs:

| Notebook | Purpose |
| --- | --- |
| `01_dataset.ipynb` | Inspect training images, prompts, targets, and classes |
| `02_forward.ipynb` | Compare LoRA-model prediction masks with targets |
| `03_loss.ipynb` | Compute the finetuning loss breakdown on loader batches |
| `04_predict.ipynb` | Run LoRA-model grid prediction and inspect masks |

Set `LORA_PATH` in notebooks `02` through `04` to inspect a trained checkpoint.
With `LORA_PATH=None`, the LoRA structure exists but its zero-initialized adapter
delta has not been trained.

## Tests

```bash
./.venv/bin/python -m pytest tests
```

The test suite covers model structure, checkpoint loading, data, finetuning
math, prediction, grounding, and video state. Local-only scripts and their
script-specific tests are intentionally not part of the remote repository.
