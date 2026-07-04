# SAM3 Src Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a flat `src/` implementation that copies only the needed SAM 3.1 interactive segmentation code, loads local weights from `weight/sam3.1_multiplex.pt`, and verifies point/box/mask prompted segmentation on `asset/sample.jpg`.

**Architecture:** `sam3-main/` remains read-only reference source. The new `src/` package owns a minimal image encoder, prompt encoder, mask decoder, checkpoint remapper, and predictor facade. Runtime never uses Hugging Face, detector language paths, training, eval, or video propagation.

**Tech Stack:** Python, PyTorch, torchvision, PIL, NumPy, timm, iopath, pytest, local CUDA through `.venv`.

---

## File Structure

- Create: `.gitignore`
  Protects local environment, generated outputs, upstream reference source, and local weights from upload.
- Modify: `AGENTS.md`
  Keeps persistent local instructions: use `.venv`, use local weights only, never commit `sam3-main/` or `weight/`.
- Modify: `requirements.txt`
  Lists non-PyTorch runtime/test dependencies without fixed versions unless required.
- Create: `src/__init__.py`
  Marks `src` as a package first, then exports `Sam3Predictor` after
  `src/predictor.py` exists.
- Create: `src/checkpoint.py`
  Loads local checkpoint files and remaps SAM 3.1 checkpoint prefixes into local module names.
- Create: `src/transforms.py`
  Handles image normalization, point/box coordinate scaling, mask postprocessing, and output visualization helpers.
- Create: `src/common.py`
  Copied/adapted SAM common layers such as `MLPBlock` and `LayerNorm2d`.
- Create: `src/rope.py`
  Copied/adapted rotary position embedding utilities.
- Create: `src/transformer.py`
  Copied/adapted SAM two-way transformer.
- Create: `src/prompt_encoder.py`
  Copied/adapted SAM prompt encoder.
- Create: `src/mask_decoder.py`
  Copied/adapted SAM mask decoder.
- Create: `src/data_misc.py`
  Minimal copied/adapted `NestedTensor` and `interpolate` utilities.
- Create: `src/model_misc.py`
  Copied/adapted model utilities required by `vit.py`, especially `AttentionType` and `LayerScale`.
- Create: `src/fused.py`
  Copied/adapted `addmm_act` fallback used by ViT MLPs.
- Create: `src/position_encoding.py`
  Copied/adapted sine positional encoding for the neck.
- Create: `src/vit.py`
  Copied/adapted ViT backbone.
- Create: `src/neck.py`
  Copied/adapted `Sam3TriViTDetNeck`.
- Create: `src/image_encoder.py`
  Minimal wrapper around the tri-head ViT neck that returns interactive image embeddings and high-resolution features.
- Create: `src/builder.py`
  Constructs the local model with SAM 3.1 hyperparameters and local checkpoint loading.
- Create: `src/predictor.py`
  Public API for `set_image()` and `predict()`.
- Create: `tests/test_checkpoint.py`
  Unit tests checkpoint key remapping without loading the 3.2 GB real checkpoint.
- Create: `tests/test_transforms.py`
  Unit tests coordinate scaling and mask output shape behavior.
- Create: `scripts/smoke_test.py`
  Runs the real local checkpoint and writes output images under `outputs/`.

### Runtime Module Shape

Use `src` as the package name. Import from the workspace root:

```python
from src.predictor import Sam3Predictor
```

Do not add `src/` itself to `sys.path`; add the workspace root.

---

### Task 1: Repository Guardrails

**Files:**
- Create: `.gitignore`
- Modify: `AGENTS.md`
- Verify: `docs/superpowers/specs/2026-07-04-sam3-src-rewrite-design.md`

- [ ] **Step 1: Ensure `.gitignore` protects local-only paths**

Create or update `.gitignore` with exactly these required entries plus any existing safe entries:

```gitignore
# Local environment
.venv/
__pycache__/
*.py[cod]
.pytest_cache/

# Generated outputs
outputs/

# Reference source and local model artifacts: never upload these.
sam3-main/
weight/
```

- [ ] **Step 2: Verify protected paths are present**

Run:

```powershell
Select-String -Path .gitignore -Pattern "^sam3-main/$|^weight/$|^.venv/$|^outputs/$"
```

Expected: one match for each protected path.

- [ ] **Step 3: Initialize Git only if needed**

Run:

```powershell
git rev-parse --is-inside-work-tree
```

Expected if already initialized: `true`.

If it fails with `fatal: not a git repository`, run:

```powershell
git init
git remote add origin https://github.com/phykn/sam3-finetune.git
```

If `origin` already exists but points elsewhere, stop and report the existing remote before changing it.

- [ ] **Step 4: Verify ignored local directories are not stageable**

Run:

```powershell
git status --short --ignored
```

Expected: `sam3-main/` and `weight/` appear as ignored (`!!`) or are absent from untracked/staged output. They must not appear as `??`, `A`, or `M`.

- [ ] **Step 5: Commit guardrail files if Git is initialized**

Run:

```powershell
git status --short
git add .gitignore AGENTS.md requirements.txt docs/superpowers/specs/2026-07-04-sam3-src-rewrite-design.md docs/superpowers/plans/2026-07-04-sam3-src-rewrite.md
git status --short
```

Expected: staged files do not include `sam3-main/` or `weight/`.

Commit:

```powershell
git commit -m "docs: add sam3 src rewrite plan"
```

If Git has no user identity configured, stop and report the exact Git error.

---

### Task 2: Checkpoint Remapping

**Files:**
- Create: `src/__init__.py`
- Create: `src/checkpoint.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write checkpoint remapping tests**

Create `tests/test_checkpoint.py`:

```python
import torch

from src.checkpoint import filter_and_remap_state_dict


def test_filter_and_remap_state_dict_keeps_interactive_prompt_decoder_and_backbone():
    source = {
        "tracker.model.interactive_sam_prompt_encoder.point_embeddings.0.weight": torch.zeros(1, 256),
        "tracker.model.interactive_sam_mask_decoder.iou_token.weight": torch.zeros(1, 256),
        "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight": torch.zeros(1024, 3, 14, 14),
        "detector.backbone.language_backbone.encoder.token_embedding.weight": torch.zeros(1, 1),
        "detector.transformer.decoder.layers.0.ca_text.in_proj_weight": torch.zeros(1, 1),
    }

    remapped, ignored = filter_and_remap_state_dict(source)

    assert "prompt_encoder.point_embeddings.0.weight" in remapped
    assert "mask_decoder.iou_token.weight" in remapped
    assert "image_encoder.vision_backbone.trunk.patch_embed.proj.weight" in remapped
    assert "detector.backbone.language_backbone.encoder.token_embedding.weight" in ignored
    assert "detector.transformer.decoder.layers.0.ca_text.in_proj_weight" in ignored


def test_filter_and_remap_state_dict_accepts_nested_model_key():
    source = {
        "model": {
            "tracker.model.interactive_sam_prompt_encoder.no_mask_embed.weight": torch.zeros(1, 256),
        }
    }

    remapped, ignored = filter_and_remap_state_dict(source)

    assert ignored == []
    assert list(remapped.keys()) == ["prompt_encoder.no_mask_embed.weight"]
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_checkpoint.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `src.checkpoint` or missing `filter_and_remap_state_dict`.

- [ ] **Step 3: Create package marker**

Create `src/__init__.py`:

```python
__all__ = []
```

The public `Sam3Predictor` export is added in Task 7 after `src/predictor.py`
exists.

- [ ] **Step 4: Implement local checkpoint utilities**

Create `src/checkpoint.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


@dataclass(frozen=True)
class LoadReport:
    checkpoint_path: Path
    loaded_keys: int
    ignored_keys: int
    missing_keys: list[str]
    unexpected_keys: list[str]
    ignored_key_examples: list[str]


def _unwrap_state_dict(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"]
    return checkpoint


def filter_and_remap_state_dict(checkpoint: Mapping) -> tuple[dict[str, torch.Tensor], list[str]]:
    state = _unwrap_state_dict(checkpoint)
    remapped: dict[str, torch.Tensor] = {}
    ignored: list[str] = []

    prefix_map = {
        "tracker.model.interactive_sam_prompt_encoder.": "prompt_encoder.",
        "tracker.model.interactive_sam_mask_decoder.": "mask_decoder.",
        "detector.backbone.vision_backbone.": "image_encoder.vision_backbone.",
    }

    for key, value in state.items():
        target_key = None
        for source_prefix, local_prefix in prefix_map.items():
            if key.startswith(source_prefix):
                target_key = local_prefix + key[len(source_prefix) :]
                break
        if target_key is None:
            ignored.append(key)
            continue
        remapped[target_key] = value

    return remapped, ignored


def load_local_checkpoint(path: str | Path) -> Mapping:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def load_weights(model: torch.nn.Module, path: str | Path, strict: bool = False) -> LoadReport:
    checkpoint_path = Path(path)
    checkpoint = load_local_checkpoint(checkpoint_path)
    remapped, ignored = filter_and_remap_state_dict(checkpoint)
    result = model.load_state_dict(remapped, strict=strict)
    return LoadReport(
        checkpoint_path=checkpoint_path,
        loaded_keys=len(remapped),
        ignored_keys=len(ignored),
        missing_keys=list(result.missing_keys),
        unexpected_keys=list(result.unexpected_keys),
        ignored_key_examples=ignored[:20],
    )
```

- [ ] **Step 5: Run checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_checkpoint.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit checkpoint utility**

Run:

```powershell
git status --short
git add src/__init__.py src/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: add local checkpoint remapping"
```

Keep `src/__init__.py` as `__all__ = []` until Task 7 restores the public export.

---

### Task 3: Transforms And Coordinate Handling

**Files:**
- Create: `src/transforms.py`
- Create: `tests/test_transforms.py`

- [ ] **Step 1: Write transform tests**

Create `tests/test_transforms.py`:

```python
import numpy as np
import torch
from PIL import Image

from src.transforms import Sam3Transforms


def test_transform_coords_scales_pixel_points_to_model_resolution():
    transforms = Sam3Transforms(resolution=1008)
    coords = np.array([[50.0, 25.0]], dtype=np.float32)

    out = transforms.transform_coords(coords, orig_hw=(100, 200))

    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[252.0, 252.0]]))


def test_transform_box_scales_xyxy_to_two_corner_points():
    transforms = Sam3Transforms(resolution=1008)
    box = np.array([20.0, 10.0, 180.0, 90.0], dtype=np.float32)

    out = transforms.transform_box(box, orig_hw=(100, 200))

    assert out.shape == (1, 2, 2)
    assert torch.allclose(out[0, 0], torch.tensor([100.8, 100.8]))
    assert torch.allclose(out[0, 1], torch.tensor([907.2, 907.2]))


def test_preprocess_image_returns_batch_tensor_and_original_hw():
    transforms = Sam3Transforms(resolution=1008)
    image = Image.new("RGB", (20, 10), color=(255, 0, 0))

    tensor, orig_hw = transforms.preprocess_image(image, device=torch.device("cpu"))

    assert orig_hw == (10, 20)
    assert tensor.shape == (1, 3, 1008, 1008)
    assert tensor.dtype == torch.float32
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_transforms.py -q
```

Expected: FAIL with missing `src.transforms` or `Sam3Transforms`.

- [ ] **Step 3: Implement transforms**

Create `src/transforms.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Normalize, Resize, ToTensor


class Sam3Transforms:
    def __init__(self, resolution: int = 1008, mask_threshold: float = 0.0) -> None:
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.to_tensor = ToTensor()
        self.resize = Resize((resolution, resolution))
        self.normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def preprocess_image(
        self, image: Image.Image | np.ndarray, device: torch.device
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if isinstance(image, Image.Image):
            image_rgb = image.convert("RGB")
            width, height = image_rgb.size
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("NumPy images must have shape HxWx3")
            image_rgb = Image.fromarray(image.astype(np.uint8), mode="RGB")
            height, width = image.shape[:2]
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")

        tensor = self.normalize(self.resize(self.to_tensor(image_rgb))).unsqueeze(0)
        return tensor.to(device=device), (height, width)

    def transform_coords(
        self, coords: np.ndarray | torch.Tensor, orig_hw: tuple[int, int]
    ) -> torch.Tensor:
        coords_t = torch.as_tensor(coords, dtype=torch.float32).clone()
        if coords_t.shape[-1] != 2:
            raise ValueError("Point coordinates must end with dimension 2")
        h, w = orig_hw
        coords_t[..., 0] = coords_t[..., 0] / float(w)
        coords_t[..., 1] = coords_t[..., 1] / float(h)
        return coords_t * float(self.resolution)

    def transform_box(
        self, box: np.ndarray | torch.Tensor, orig_hw: tuple[int, int]
    ) -> torch.Tensor:
        box_t = torch.as_tensor(box, dtype=torch.float32)
        if box_t.numel() != 4:
            raise ValueError("Box must contain four values: x0, y0, x1, y1")
        return self.transform_coords(box_t.reshape(1, 2, 2), orig_hw)

    def postprocess_masks(
        self, masks: torch.Tensor, orig_hw: tuple[int, int], return_logits: bool = False
    ) -> torch.Tensor:
        masks = F.interpolate(masks.float(), orig_hw, mode="bilinear", align_corners=False)
        if return_logits:
            return masks
        return masks > self.mask_threshold


def save_mask_png(mask: np.ndarray, path: str | Path) -> None:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_uint8, mode="L").save(path)


def save_overlay_png(image: Image.Image, mask: np.ndarray, path: str | Path) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha = (mask.astype(np.uint8) * 120)
    overlay.putalpha(Image.fromarray(alpha, mode="L"))
    Image.alpha_composite(base, overlay).save(path)
```

- [ ] **Step 4: Run transform tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_transforms.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit transforms**

Run:

```powershell
git add src/transforms.py tests/test_transforms.py
git commit -m "feat: add image and prompt transforms"
```

---

### Task 4: Copy SAM Prompt And Mask Modules

**Files:**
- Create: `src/common.py`
- Create: `src/rope.py`
- Create: `src/transformer.py`
- Create: `src/prompt_encoder.py`
- Create: `src/mask_decoder.py`

- [ ] **Step 1: Copy source files**

Run from `D:\code\sam3`:

```powershell
Copy-Item -LiteralPath sam3-main\sam3\sam\common.py -Destination src\common.py
Copy-Item -LiteralPath sam3-main\sam3\sam\rope.py -Destination src\rope.py
Copy-Item -LiteralPath sam3-main\sam3\sam\transformer.py -Destination src\transformer.py
Copy-Item -LiteralPath sam3-main\sam3\sam\prompt_encoder.py -Destination src\prompt_encoder.py
Copy-Item -LiteralPath sam3-main\sam3\sam\mask_decoder.py -Destination src\mask_decoder.py
```

- [ ] **Step 2: Replace SAM package imports with local imports**

In `src/transformer.py`, replace:

```python
from sam3.sam.rope import apply_rotary_enc, apply_rotary_enc_real, compute_axial_cis
```

with:

```python
from .rope import apply_rotary_enc, apply_rotary_enc_real, compute_axial_cis
```

The existing relative imports in `prompt_encoder.py` and `mask_decoder.py` should remain:

```python
from .common import LayerNorm2d
```

and:

```python
from .common import MLPBlock
```

- [ ] **Step 3: Verify copied modules import**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from src.prompt_encoder import PromptEncoder; from src.mask_decoder import MaskDecoder; from src.transformer import TwoWayTransformer; print('sam modules import ok')"
```

Expected: `sam modules import ok`.

- [ ] **Step 4: Commit copied SAM modules**

Run:

```powershell
git add src/common.py src/rope.py src/transformer.py src/prompt_encoder.py src/mask_decoder.py
git commit -m "feat: copy sam prompt and mask modules"
```

---

### Task 5: Copy Image Encoder Dependencies

**Files:**
- Create: `src/data_misc.py`
- Create: `src/model_misc.py`
- Create: `src/fused.py`
- Create: `src/position_encoding.py`
- Create: `src/vit.py`
- Create: `src/neck.py`

- [ ] **Step 1: Copy source files**

Run:

```powershell
Copy-Item -LiteralPath sam3-main\sam3\model\data_misc.py -Destination src\data_misc.py
Copy-Item -LiteralPath sam3-main\sam3\model\model_misc.py -Destination src\model_misc.py
Copy-Item -LiteralPath sam3-main\sam3\perflib\fused.py -Destination src\fused.py
Copy-Item -LiteralPath sam3-main\sam3\model\position_encoding.py -Destination src\position_encoding.py
Copy-Item -LiteralPath sam3-main\sam3\model\vitdet.py -Destination src\vit.py
Copy-Item -LiteralPath sam3-main\sam3\model\necks.py -Destination src\neck.py
```

- [ ] **Step 2: Replace image module imports**

In `src/vit.py`, replace:

```python
from sam3.model.data_misc import NestedTensor
from sam3.model.model_misc import AttentionType, LayerScale
from sam3.perflib.fused import addmm_act
from sam3.sam.rope import apply_rotary_enc_real, VisionRotaryEmbeddingVE
```

with:

```python
from .data_misc import NestedTensor
from .model_misc import AttentionType, LayerScale
from .fused import addmm_act
from .rope import apply_rotary_enc_real, VisionRotaryEmbeddingVE
```

In `src/neck.py`, replace:

```python
from sam3.model.data_misc import NestedTensor
```

with:

```python
from .data_misc import NestedTensor
```

- [ ] **Step 3: Verify image modules import**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from src.vit import ViT; from src.neck import Sam3TriViTDetNeck; from src.position_encoding import PositionEmbeddingSine; print('image modules import ok')"
```

Expected: `image modules import ok`. A warning about Flash Attention being disabled on RTX 2060 is acceptable.

- [ ] **Step 4: Commit copied image modules**

Run:

```powershell
git add src/data_misc.py src/model_misc.py src/fused.py src/position_encoding.py src/vit.py src/neck.py
git commit -m "feat: copy image encoder dependencies"
```

---

### Task 6: Build The Minimal Model

**Files:**
- Create: `src/image_encoder.py`
- Create: `src/builder.py`
- Create: `tests/test_builder.py`

- [ ] **Step 1: Write builder smoke unit test**

Create `tests/test_builder.py`:

```python
import torch

from src.builder import build_model


def test_build_model_has_expected_submodules():
    model = build_model(device=torch.device("cpu"))

    assert hasattr(model, "image_encoder")
    assert hasattr(model, "prompt_encoder")
    assert hasattr(model, "mask_decoder")
    assert model.image_size == 1008
    assert model.backbone_stride == 14
```

- [ ] **Step 2: Run builder test to verify it fails before implementation**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_builder.py -q
```

Expected: FAIL with missing `src.builder` or `build_model`.

- [ ] **Step 3: Implement image encoder**

Create `src/image_encoder.py`:

```python
from __future__ import annotations

import torch
from torch import nn

from .data_misc import NestedTensor
from .neck import Sam3TriViTDetNeck


class InteractiveImageEncoder(nn.Module):
    def __init__(self, vision_backbone: Sam3TriViTDetNeck) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone

    def forward(self, images: torch.Tensor, mask_decoder: nn.Module) -> dict[str, object]:
        _, _, interactive_features, _interactive_pos, _, _ = self.vision_backbone(
            images,
            need_sam3_out=False,
            need_interactive_out=True,
            need_propagation_out=False,
        )

        if len(interactive_features) < 3:
            raise RuntimeError("Interactive encoder expected three feature levels")

        interactive_features[0].tensors = mask_decoder.conv_s0(
            interactive_features[0].tensors
        )
        interactive_features[1].tensors = mask_decoder.conv_s1(
            interactive_features[1].tensors
        )

        image_embed = interactive_features[-1].tensors
        high_res_features = [
            interactive_features[0].tensors,
            interactive_features[1].tensors,
        ]
        return {
            "image_embed": image_embed,
            "high_res_features": high_res_features,
        }
```

- [ ] **Step 4: Implement builder**

Create `src/builder.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn

from .checkpoint import LoadReport, load_weights
from .image_encoder import InteractiveImageEncoder
from .mask_decoder import MaskDecoder
from .neck import Sam3TriViTDetNeck
from .position_encoding import PositionEmbeddingSine
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer
from .vit import ViT


class Sam3PromptedSegmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_size = 1008
        self.backbone_stride = 14
        self.hidden_dim = 256
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        position_encoding = PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=1008,
        )
        trunk = ViT(
            img_size=1008,
            pretrain_img_size=336,
            patch_size=14,
            embed_dim=1024,
            depth=32,
            num_heads=16,
            mlp_ratio=4.625,
            norm_layer="LayerNorm",
            drop_path_rate=0.1,
            qkv_bias=True,
            use_abs_pos=True,
            tile_abs_pos=True,
            global_att_blocks=(7, 15, 23, 31),
            rel_pos_blocks=(),
            use_rope=True,
            use_interp_rope=True,
            window_size=24,
            pretrain_use_cls_token=True,
            retain_cls_token=False,
            ln_pre=True,
            ln_post=False,
            return_interm_layers=False,
            bias_patch_embed=False,
            compile_mode=None,
            use_fa3=False,
            use_rope_real=False,
        )
        tri_neck = Sam3TriViTDetNeck(
            trunk=trunk,
            position_encoding=position_encoding,
            d_model=256,
            scale_factors=[4.0, 2.0, 1.0],
        )
        self.image_encoder = InteractiveImageEncoder(tri_neck)
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=False,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
        )

    def encode_image(self, images: torch.Tensor) -> dict[str, object]:
        return self.image_encoder(images, self.mask_decoder)


def build_model(
    checkpoint_path: str | None = None,
    device: torch.device | str = "cuda",
) -> tuple[Sam3PromptedSegmenter, LoadReport | None] | Sam3PromptedSegmenter:
    model = Sam3PromptedSegmenter().to(device)
    model.eval()
    if checkpoint_path is None:
        return model
    report = load_weights(model, checkpoint_path, strict=False)
    model.to(device)
    model.eval()
    return model, report
```

- [ ] **Step 5: Run builder test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_builder.py -q
```

Expected: PASS.

- [ ] **Step 6: Run a real checkpoint load summary**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from src.builder import build_model; model, report = build_model('weight/sam3.1_multiplex.pt', device='cuda'); print(report)"
```

Expected: command completes without Hugging Face access. Missing keys should correspond to local model keys that are not present or intentionally excluded; unexpected keys should be empty because unmapped checkpoint keys are counted as ignored before `load_state_dict`.

- [ ] **Step 7: Commit builder**

Run:

```powershell
git add src/image_encoder.py src/builder.py tests/test_builder.py
git commit -m "feat: build minimal prompted segmenter"
```

---

### Task 7: Predictor API

**Files:**
- Create: `src/predictor.py`
- Modify: `src/__init__.py`
- Create: `tests/test_predictor_api.py`

- [ ] **Step 1: Write predictor API tests with a fake model**

Create `tests/test_predictor_api.py`:

```python
import numpy as np
import torch
from PIL import Image

from src.predictor import Sam3Predictor


class FakePromptEncoder(torch.nn.Module):
    mask_input_size = (288, 288)

    def forward(self, points=None, boxes=None, masks=None):
        sparse = torch.zeros(1, 3, 256)
        dense = torch.zeros(1, 256, 72, 72)
        return sparse, dense

    def get_dense_pe(self):
        return torch.zeros(1, 256, 72, 72)


class FakeMaskDecoder(torch.nn.Module):
    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        repeat_image,
        high_res_features,
    ):
        return (
            torch.ones(1, 1, 288, 288),
            torch.tensor([[0.9]]),
            torch.zeros(1, 1, 256),
            torch.ones(1, 1),
        )


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_encoder = FakePromptEncoder()
        self.mask_decoder = FakeMaskDecoder()

    def encode_image(self, images):
        return {
            "image_embed": torch.zeros(1, 256, 72, 72),
            "high_res_features": [
                torch.zeros(1, 32, 288, 288),
                torch.zeros(1, 64, 144, 144),
            ],
        }


def test_predictor_accepts_box_and_returns_numpy_outputs():
    predictor = Sam3Predictor(FakeModel(), device=torch.device("cpu"))
    predictor.set_image(Image.new("RGB", (20, 10), color=(0, 0, 0)))

    masks, scores, low_res = predictor.predict(box=np.array([2, 1, 18, 9], dtype=np.float32))

    assert masks.shape == (1, 10, 20)
    assert scores.tolist() == [0.9]
    assert low_res.shape == (1, 288, 288)
```

- [ ] **Step 2: Run predictor test to verify it fails before implementation**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_predictor_api.py -q
```

Expected: FAIL with missing `src.predictor` or `Sam3Predictor`.

- [ ] **Step 3: Implement predictor**

Create `src/predictor.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .builder import build_model
from .checkpoint import LoadReport
from .transforms import Sam3Transforms


class Sam3Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cuda",
        load_report: LoadReport | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.transforms = Sam3Transforms(resolution=1008, mask_threshold=0.0)
        self.load_report = load_report
        self._features: dict[str, object] | None = None
        self._orig_hw: tuple[int, int] | None = None

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: torch.device | str = "cuda"
    ) -> "Sam3Predictor":
        model, report = build_model(str(checkpoint_path), device=device)
        return cls(model=model, device=device, load_report=report)

    @torch.inference_mode()
    def set_image(self, image: Image.Image | np.ndarray) -> None:
        input_tensor, orig_hw = self.transforms.preprocess_image(image, self.device)
        self._features = self.model.encode_image(input_tensor)
        self._orig_hw = orig_hw

    @torch.inference_mode()
    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._features is None or self._orig_hw is None:
            raise RuntimeError("Call set_image() before predict().")

        concat_points = None
        if point_coords is not None:
            if point_labels is None:
                raise ValueError("point_labels must be supplied with point_coords")
            coords = self.transforms.transform_coords(point_coords, self._orig_hw).to(
                self.device
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if coords.ndim == 2:
                coords = coords[None, ...]
                labels = labels[None, ...]
            concat_points = (coords, labels)

        if box is not None:
            box_coords = self.transforms.transform_box(box, self._orig_hw).to(self.device)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=self.device)
            if concat_points is None:
                concat_points = (box_coords, box_labels)
            else:
                concat_points = (
                    torch.cat([box_coords, concat_points[0]], dim=1),
                    torch.cat([box_labels, concat_points[1]], dim=1),
                )

        mask_prompt = None
        if mask_input is not None:
            mask_prompt = torch.as_tensor(mask_input, dtype=torch.float32, device=self.device)
            if mask_prompt.ndim == 3:
                mask_prompt = mask_prompt[None, :, :, :]
            if mask_prompt.shape[-2:] != self.model.prompt_encoder.mask_input_size:
                mask_prompt = F.interpolate(
                    mask_prompt,
                    size=self.model.prompt_encoder.mask_input_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )

        if concat_points is None and mask_prompt is None:
            raise ValueError("Provide at least one point, box, or mask prompt.")

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_prompt,
        )
        low_res_masks, iou_predictions, _tokens, _obj_scores = self.model.mask_decoder(
            image_embeddings=self._features["image_embed"],
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=self._features["high_res_features"],
        )
        masks = self.transforms.postprocess_masks(
            low_res_masks, self._orig_hw, return_logits=return_logits
        )
        return (
            masks.squeeze(0).detach().cpu().numpy(),
            iou_predictions.squeeze(0).float().detach().cpu().numpy(),
            torch.clamp(low_res_masks, -32.0, 32.0).squeeze(0).float().detach().cpu().numpy(),
        )
```

- [ ] **Step 4: Restore package export**

Update `src/__init__.py`:

```python
from .predictor import Sam3Predictor

__all__ = ["Sam3Predictor"]
```

- [ ] **Step 5: Run predictor API test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_predictor_api.py -q
```

Expected: PASS.

- [ ] **Step 6: Run unit test set**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 7: Commit predictor**

Run:

```powershell
git add src/predictor.py src/__init__.py tests/test_predictor_api.py
git commit -m "feat: add prompted segmentation predictor"
```

---

### Task 8: Real Smoke Test

**Files:**
- Create: `scripts/smoke_test.py`

- [ ] **Step 1: Create smoke test script**

Create `scripts/smoke_test.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predictor import Sam3Predictor
from src.transforms import save_mask_png, save_overlay_png


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    image_path = ROOT / "asset" / "sample.jpg"
    checkpoint_path = ROOT / "weight" / "sam3.1_multiplex.pt"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    box = np.array(
        [width * 0.25, height * 0.25, width * 0.75, height * 0.75],
        dtype=np.float32,
    )
    point_coords = np.array([[width * 0.5, height * 0.5]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int64)

    predictor = Sam3Predictor.from_checkpoint(checkpoint_path, device="cuda")
    predictor.set_image(image)
    masks, scores, low_res = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(bool)

    mask_path = output_dir / "smoke_mask.png"
    overlay_path = output_dir / "smoke_overlay.png"
    save_mask_png(best_mask, mask_path)
    save_overlay_png(image, best_mask, overlay_path)

    report = predictor.load_report
    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    if report is not None:
        print(f"loaded_keys: {report.loaded_keys}")
        print(f"ignored_keys: {report.ignored_keys}")
        print(f"missing_keys: {len(report.missing_keys)}")
        print(f"unexpected_keys: {len(report.unexpected_keys)}")
        print(f"ignored_key_examples: {report.ignored_key_examples[:5]}")
    print(f"masks_shape: {masks.shape}")
    print(f"low_res_shape: {low_res.shape}")
    print(f"scores: {scores.tolist()}")
    print(f"mask_path: {mask_path}")
    print(f"overlay_path: {overlay_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

Expected: script prints checkpoint path, RTX 2060 device name, key summary, mask shape, score values, and output paths.

- [ ] **Step 3: Verify output files exist**

Run:

```powershell
Test-Path outputs\smoke_mask.png
Test-Path outputs\smoke_overlay.png
```

Expected: both commands print `True`.

- [ ] **Step 4: Commit smoke test**

Run:

```powershell
git add scripts/smoke_test.py
git commit -m "test: add local sam3 smoke test"
```

Do not add `outputs/`, `sam3-main/`, or `weight/`.

---

### Task 9: Final Verification And Push Readiness

**Files:**
- Verify: all changed files

- [ ] **Step 1: Run all unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 2: Run real smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

Expected: PASS and output files created.

- [ ] **Step 3: Verify forbidden directories are ignored**

Run:

```powershell
git status --short --ignored
```

Expected: `sam3-main/`, `weight/`, `.venv/`, and `outputs/` are ignored (`!!`) or absent. They must not be staged.

- [ ] **Step 4: Verify remote**

Run:

```powershell
git remote -v
```

Expected: `origin` points to `https://github.com/phykn/sam3-finetune.git`.

- [ ] **Step 5: Push only after status is safe**

Run:

```powershell
git status --short
```

Expected: no uncommitted tracked changes, and no staged or untracked `sam3-main/` or `weight/`.

Push:

```powershell
git push -u origin main
```

If the default branch is not `main`, run `git branch --show-current` and push that branch instead:

```powershell
git push -u origin HEAD
```

Report the pushed branch and the exact verification commands that passed.

---

## Self-Review

Spec coverage:

- New flat `src/` layout: Tasks 2 through 7.
- Copy needed code from `sam3-main/`: Tasks 4 and 5.
- No nested `sam3lite/`: file structure uses only `src/`.
- No Hugging Face: Tasks 2, 6, and 8 use local checkpoint paths only.
- Local weight loading: Tasks 2, 6, and 8.
- Point, box, and mask prompt API: Tasks 3 and 7.
- `asset/sample.jpg` smoke test: Task 8.
- Do not upload `sam3-main/` or `weight/`: Tasks 1 and 9.

Placeholder scan:

- No placeholder markers are used.
- Code steps include exact file contents or exact copy/import replacement commands.

Type consistency:

- The public API consistently uses `Sam3Predictor.from_checkpoint()`, `set_image()`, and `predict()`.
- Checkpoint remapping targets `image_encoder`, `prompt_encoder`, and `mask_decoder`, matching the builder module names.
- Smoke test imports from `src.predictor`, matching the flat package design.
