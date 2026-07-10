# SAM3 Finetune

Lightweight SAM 3 rewrite for local finetuning and inference experiments.

The project keeps the upstream SAM 3 checkout as reference code and builds a
smaller runtime under `src/`.

## Overview

The model code is organized as a one-way stack:

```text
components -> blocks -> models -> predictors
```

Main model entrypoints:

- `Sam3ImageModel`: prompt-based image segmentation
- `Sam3GroundingModel`: visual-token grounding
- `Sam3VideoModel`: video mask propagation

Prediction workflows:

- `single`: one image with point, box, or mask prompts
- `grid`: automatic mask proposals from grid prompts
- `ground`: reference-guided grounding
- `video`: mask propagation across frames

## Architecture

Runtime dependencies move in one direction:

```text
data/io/ops/runtime -> components -> blocks -> model -> build/predict -> scripts
```

- `components` contain reusable mathematical modules.
- `blocks` assemble components for one model stage.
- `model` connects blocks for image, grounding, or video workflows.
- `predict` uses public model methods without importing model internals.

## Repository Layout

```text
config/       model config
scripts/      runnable examples and parity checks
src/data/     input preprocessing
src/io/       checkpoint and video frame loading
src/ml/components/  reusable neural-network components
src/ml/blocks/      image, grounding, and video model blocks
src/ml/model/       assembled image, grounding, and video models
src/ops/      shared tensor and box operations
src/predict/  prediction workflows
src/finetune/ finetune model, losses, checkpoints, DDP, and trainer
tests/        unit tests
```

`sam3-main/`, `weight/`, and `asset/` are local-only directories and are not
part of the committed runtime.

## Setup

Create a virtual environment and install the Python dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install `torch` and `torchvision` separately with the CUDA wheel/index that
matches your machine.

## Build Models

```python
from src.build import build_grounding_model, build_image_model, build_video_model

config = {
    "path": "weight/sam3.pt",
    "visual_path": "weight/visual_token.pt",
    "device": "cuda",
}

image_model = build_image_model(config)
grounding_model = build_grounding_model(config)
video_model = build_video_model(config)
```

## Run Examples

```powershell
.\.venv\Scripts\python.exe scripts\single.py
.\.venv\Scripts\python.exe scripts\grid.py
.\.venv\Scripts\python.exe scripts\ground.py
.\.venv\Scripts\python.exe scripts\video.py
```

The video predictor tracks forward from mask prompts. Objects can be added on
the latest cached frame or removed explicitly:

```python
from src.predict import VideoPredictor

predictor = VideoPredictor.from_path("weight/sam3.1_multiplex.pt")
state = predictor.start(first_frame, first_mask, obj_id=1)
out = predictor.predict(next_frame, state)
ids = predictor.add_masks(state, new_masks, [2, 3])
ids = predictor.remove_objects(state, [2])
```

Automatic discovery is separate from tracking. It requires grounding results
to be associated with active tracks before calling `add_masks`.

Parity checks against the reference implementation:

```powershell
.\.venv\Scripts\python.exe scripts\parity_image.py
.\.venv\Scripts\python.exe scripts\parity_ground.py
.\.venv\Scripts\python.exe scripts\parity_video.py
.\.venv\Scripts\python.exe scripts\parity_video_dynamic.py
```

## Finetune

`config/finetune.yaml` contains the model, training/validation data, and trainer
settings. Set the local SAM3.1 checkpoint and sample paths before running it.

Single-process training:

```powershell
.\.venv\Scripts\python.exe scripts\finetune.py --config config\finetune.yaml
```

Single-server multi-GPU training on Linux:

```bash
torchrun --standalone --nproc-per-node=4 scripts/finetune.py \
  --config config/finetune.yaml
```

Resume restores the trainable parameters, optimizer, and global step. Data order
and augmentation randomness start from a new loader sequence:

```bash
torchrun --standalone --nproc-per-node=4 scripts/finetune.py \
  --config config/finetune.yaml \
  --resume run/example/checkpoints/last.pt
```

Class-head outputs are independent sigmoid attributes for each mask. Index 0 is
always particle/object presence; later indices can represent SEM attributes such
as particle size. `SinglePredictor` and `GridPredictor` return `class_logits` and
`class_scores` when used with `FinetuneModel`. Finetune example JSON stores the
probability vector in `object.metrics.class_scores`.

Checkpoints use the new `sam3.finetune.v1` format under
`run/<run-id>/checkpoints/`. Older LoRA checkpoints are not supported. CPU/Gloo
multi-process behavior is tested locally; NCCL multi-GPU execution must be checked
on the target Linux server.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

## Notes

- Official SAM3.1 checkpoints are loaded strictly from explicit local paths.
- Checkpoint key translation is owned by `src/io/checkpoint.py`.
- Checkpoints produced by the previous LoRA rewrite are not supported.
- Hugging Face loading is intentionally not used in this rewrite path.
- Cached visual tokens are used for no-text grounding.
