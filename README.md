# SAM3 Finetune Rewrite

This workspace is a local SAM 3 rewrite and finetune experiment. The upstream
`sam3-main/` tree is kept as reference material. New rewrite work belongs under
`src/`, with the prediction API split by workflow.

## Workspace Rules

- `src/` contains the maintained rewrite code.
- `sam3-main/` is reference-only for this rewrite.
- `weight/` is local checkpoint storage.
- Do not commit or publish `sam3-main/`, `weight/`, checkpoints, datasets,
  generated logs, or large media artifacts.
- Load model weights only from explicit local paths, normally under `weight/`.
- Do not use Hugging Face loading in the rewrite path.

## Environment

Use the workspace-local virtual environment from the repository root. The full
inference and training target is Python 3.12 or newer, PyTorch 2.7 or newer,
and a CUDA-compatible GPU with CUDA 12.6 or newer. Lightweight API tests can run
on CPU when the imported dependencies are installed.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install `torch` and `torchvision` separately with the CUDA wheel/index that
matches the machine. They are intentionally not pinned in `requirements.txt`.

Run tests with the `.venv` Python. The system `python` may not have the required
packages installed.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Public API

Import workflow classes from their workflow package. The root `src` and
`src.predict` packages do not re-export workflow classes.

```python
from src.predict.prompted import Sam3Predictor
from src.predict.grid import AutomaticMaskGenerator
from src.predict.refine import MaskRefiner
from src.predict.context import ContextMatcher, ReferenceGuidedMaskGenerator
from src.predict.next_frame import NextFramePredictor
from src.types import (
    ContextReference,
    MaskInstance,
    MemoryReference,
    ReferenceExample,
    Sam3ImageEmbedding,
    Sam3PromptBatch,
)
```

Do not use old root imports such as:

```python
from src.predict import Sam3Predictor
```

## Prompted Prediction

`Sam3Predictor` uses a stateless embedding API. Encode an image, then pass the
embedding explicitly to prediction or low-resolution decode methods.

```python
import numpy as np
from PIL import Image

from src.predict.prompted import Sam3Predictor

image = Image.open("asset/frog_target.jpg").convert("RGB")
predictor = Sam3Predictor.from_checkpoint("weight/sam3.1_multiplex.pt")
embedding = predictor.encode_image(image)

masks, scores, low_res_masks = predictor.predict_from_embedding(
    embedding,
    point_coords=np.array([[560.0, 500.0]], dtype=np.float32),
    point_labels=np.array([1], dtype=np.int64),
    multimask_output=False,
)
```

The stateful SAM-style API is not the current public API:

```python
predictor.set_image(image)
predictor.predict(...)
```

## Grid Prediction

`AutomaticMaskGenerator` sweeps point prompts over an image and returns mask
proposals or `MaskInstance` values. It does not own second-pass refinement or
reference-context matching.

```python
from PIL import Image

from src.predict.grid import AutomaticMaskGenerator
from src.predict.prompted import Sam3Predictor

image = Image.open("asset/frog_target.jpg").convert("RGB")
predictor = Sam3Predictor.from_checkpoint("weight/sam3.1_multiplex.pt")
generator = AutomaticMaskGenerator(predictor, points_per_side=32)

proposals = generator.generate(image)
instances = generator.generate_instances(image, source="auto")
```

## Refinement

`MaskRefiner` owns explicit `mask_input` second-pass refinement. Reuse an image
embedding when a caller already has one.

```python
from src.predict.prompted import Sam3Predictor
from src.predict.refine import MaskRefiner

predictor = Sam3Predictor.from_checkpoint("weight/sam3.1_multiplex.pt")
embedding = predictor.encode_image(image)

first_masks, first_scores, first_low_res = predictor.predict_from_embedding(
    embedding,
    point_coords=point_coords,
    point_labels=point_labels,
)

refined = MaskRefiner(predictor).refine(
    embedding=embedding,
    point_coords=point_coords,
    point_labels=point_labels,
    mask_input=first_low_res[0],
)
```

## Context Prediction

`ContextMatcher` finds an object again from reference image and mask context.
It should be described as reference-guided matching, not copying the old mask.

```python
from src.predict.context import ContextMatcher
from src.types import ContextReference

matcher = ContextMatcher.from_checkpoint("weight/sam3.1_multiplex.pt")
predictions = matcher.predict(
    target_image=target_image,
    references=[
        ContextReference(
            image=reference_image,
            mask=reference_mask,
        )
    ],
)
```

`ReferenceGuidedMaskGenerator` reranks candidate masks using reference context.

## Next-Frame Prediction

`NextFramePredictor` predicts one target frame from reference frame memory. The
current scope is not a full streaming tracker.

```python
from src.predict.next_frame import MemoryReference, NextFramePredictor

predictor = NextFramePredictor.from_checkpoint("weight/sam3.1_multiplex.pt")
prediction = predictor.predict(
    target_image=target_image,
    references=[
        MemoryReference(
            image=reference_frame,
            mask=reference_mask,
            obj_id=1,
        )
    ],
)
```

## Scripts

The scripts are thin command-line entrypoints for fixed local workflows.

```powershell
.\.venv\Scripts\python.exe scripts\prompt.py --device auto
.\.venv\Scripts\python.exe scripts\grid.py --device auto
.\.venv\Scripts\python.exe scripts\context.py --device auto
.\.venv\Scripts\python.exe scripts\refine.py --device auto
.\.venv\Scripts\python.exe scripts\next.py --device auto
```

Default script inputs expect local files under `asset/` and `weight/`.

## Focused Tests

Use focused checks while refactoring.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_context_predictor.py tests\test_reference_guided_masks.py tests\test_video_memory_reference.py tests\test_grounding.py tests\test_grounding_postprocess.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_package_structure.py tests\test_predictor_api.py tests\test_refine_predictor.py -q
```

Before broader completion claims, run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```
