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

## Repository Layout

```text
config/       model config
scripts/      runnable examples and parity checks
src/data/     input preprocessing
src/io/       checkpoint and video frame loading
src/ml/       model components, blocks, and assembled models
src/ops/      shared tensor and box operations
src/predict/  prediction workflows
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

Parity checks against the reference implementation:

```powershell
.\.venv\Scripts\python.exe scripts\parity_image.py
.\.venv\Scripts\python.exe scripts\parity_ground.py
.\.venv\Scripts\python.exe scripts\parity_video.py
```

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

## Notes

- Checkpoints are loaded from explicit local paths.
- Hugging Face loading is intentionally not used in this rewrite path.
- Cached visual tokens are used for no-text grounding.
