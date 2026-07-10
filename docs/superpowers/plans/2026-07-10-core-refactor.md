# SAM3 Core Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the shared math and image/grounding model core into a clear one-way architecture while retaining strict official SAM3.1 loading and exact CUDA inference parity.

**Architecture:** Keep the dependency flow `data/io/ops/runtime -> components -> blocks -> model -> build/predict -> scripts`. Components own reusable math, blocks own module assembly, and model files only coordinate blocks. Move active video transformer primitives only where the core split requires it; leave the video tracker state machine unchanged.

**Tech Stack:** Python 3.11, PyTorch, torchvision, pytest, Ruff, Black, local SAM3.1 checkpoint and cached visual tokens.

## Global Constraints

- Run commands from `D:\code\sam3` with `.venv\Scripts\python.exe`.
- Keep code simple and short; do not add unnecessary wrappers, dataclasses, options, lazy imports, Meta headers, `__all__`, or `from __future__ import annotations`.
- Do not modify, stage, commit, or push `sam3-main/`, `weight/`, or `asset/`.
- Load weights only from explicit local paths. Do not add Hugging Face loading or instantiate the VLM/text encoder.
- Official SAM3.1 base checkpoints must load strictly; old rewrite LoRA checkpoint keys are unsupported.
- Preserve exact image and grounding CUDA parity. Structural movement must not change active formulas.
- Preserve the measured zero-difference video parity while shared transformer code moves.
- Keep boolean attention-mask semantics `True = excluded` and spatial flattening `(B,C,H,W) -> (H*W,B,C)`.
- Use Black line length 88 and leave unrelated video formatting untouched.
- Do not create commits unless the user separately authorizes commits. End each task with a status/diff checkpoint instead.

---

### Task 1: Rename Stateless Math Operations and Add Invariant Tests

**Files:**

- Create: `tests/test_math_ops.py`
- Modify: `src/ops/box.py`
- Modify: `src/ops/tensor.py`
- Modify: all current `invert_sigmoid`, `convert_to_xyxy`, and `filter_boxes` call sites reported by `rg`

**Interfaces:**

- Produces: `inverse_sigmoid(tensor: torch.Tensor, eps: float = 1e-3) -> torch.Tensor`
- Produces: `cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor`
- Produces: `nms_indices(boxes, scores, iou_threshold: float) -> list[int]`
- Preserves: the exact formulas and NMS ordering of the current functions

- [ ] **Step 1: Write failing tests for the new names and formulas**

```python
import numpy as np
import torch

from src.ops.box import cxcywh_to_xyxy, nms_indices
from src.ops.tensor import inverse_sigmoid


def test_inverse_sigmoid_round_trip():
    values = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    torch.testing.assert_close(torch.sigmoid(inverse_sigmoid(values)), values)


def test_cxcywh_to_xyxy_uses_center_and_size():
    boxes = torch.tensor([[0.5, 0.25, 0.4, 0.2]])
    expected = torch.tensor([[0.3, 0.15, 0.7, 0.35]])
    torch.testing.assert_close(cxcywh_to_xyxy(boxes), expected)


def test_nms_indices_accepts_numpy_and_keeps_score_order():
    boxes = np.array([[0, 0, 2, 2], [0, 0, 2, 2], [4, 4, 5, 5]])
    scores = np.array([0.8, 0.9, 0.7])
    assert nms_indices(boxes, scores, 0.5) == [1, 2]
```

- [ ] **Step 2: Run the focused tests and verify import failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_math_ops.py -q
```

Expected: collection fails because the new names do not exist.

- [ ] **Step 3: Rename the functions without changing their formulas**

```python
def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def nms_indices(boxes, scores, iou_threshold: float) -> list[int]:
    boxes_t = _as_float_tensor(boxes)
    scores_t = _as_float_tensor(scores)
    if boxes_t.numel() == 0:
        return []
    if boxes_t.ndim != 2 or boxes_t.shape[1] != 4:
        raise ValueError("boxes must have shape Nx4")
    if scores_t.ndim != 1 or scores_t.shape[0] != boxes_t.shape[0]:
        raise ValueError("scores must have shape N")
    keep = nms(boxes_t, scores_t, float(iou_threshold))
    return [int(index) for index in keep.detach().cpu().tolist()]


def inverse_sigmoid(tensor: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    tensor = tensor.clamp(min=0, max=1)
    numerator = tensor.clamp(min=eps)
    denominator = (1 - tensor).clamp(min=eps)
    return torch.log(numerator / denominator)
```

Update every call site in `src`, `tests`, and `scripts`; do not leave compatibility aliases.

- [ ] **Step 4: Run focused and architecture-sensitive tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_math_ops.py tests\test_ground_blocks.py tests\test_ground_predict.py tests\test_grid_predict.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Record the task checkpoint**

```powershell
git status --short
git diff --check
```

Expected: only Task 1 source and test files are changed; whitespace check passes.

---

### Task 2: Make Position Caching Device-Aware and Lock RoPE Invariants

**Files:**

- Create: `tests/test_position_math.py`
- Modify: `src/ml/components/nn/position.py`
- Modify: `src/ml/components/sam/rope.py`

**Interfaces:**

- Preserves: `PositionEmbeddingSine.forward(x) -> Tensor[B,C,H,W]`
- Preserves: complex and real RoPE public functions
- Adds: cache keys containing spatial size and input device
- Adds: explicit even-channel and integral repeat-ratio validation

- [ ] **Step 1: Add CPU-cache and RoPE equivalence tests**

```python
import torch

from src.ml.components.nn.position import PositionEmbeddingSine
from src.ml.components.sam.rope import (
    apply_rotary_enc,
    apply_rotary_enc_real,
    compute_axial_cis,
)


def test_position_encoding_cache_uses_input_device():
    encoder = PositionEmbeddingSine(num_pos_feats=8)
    image = torch.zeros(2, 1, 8, 8)
    output = encoder(image)
    assert output.device == image.device
    assert output.shape == (2, 8, 8, 8)


def test_real_and_complex_rope_match_and_preserve_norm():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 4, 8)
    freqs = compute_axial_cis(dim=8, end_x=2, end_y=2)
    complex_query, complex_key = apply_rotary_enc(query, key, freqs)
    real_query, real_key = apply_rotary_enc_real(
        query, key, freqs.real, freqs.imag
    )
    torch.testing.assert_close(real_query, complex_query)
    torch.testing.assert_close(real_key, complex_key)
    torch.testing.assert_close(complex_query.norm(dim=-1), query.norm(dim=-1))
```

- [ ] **Step 2: Verify the CPU cache test fails on the current CUDA-only cache**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_position_math.py -q
```

Expected before implementation: the precomputed cache path returns CUDA data
for a CPU input or construction fails on a machine without CUDA.

- [ ] **Step 3: Replace constructor-time CUDA tensors with an input-device cache**

Use this cache contract in `PositionEmbeddingSine`:

```python
self.cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def forward(self, x):
    height, width = x.shape[-2:]
    key = (height, width, x.device)
    if key not in self.cache:
        self.cache[key] = self._encode_grid(
            batch_size=1,
            height=height,
            width=width,
            device=x.device,
        )[0].detach()
    return self.cache[key][None].expand(x.shape[0], -1, -1, -1)
```

Move the existing grid formula unchanged into `_encode_grid`. Do not precompute
on CPU and copy to CUDA because that can change the exact GPU trigonometric
results used by the parity contract.

- [ ] **Step 4: Add RoPE validation without changing valid-path math**

```python
if dim % 4:
    raise ValueError(f"RoPE dim must be divisible by 4, got {dim}")

if repeat_freqs_k and xk_.shape[-2] % xq_.shape[-2]:
    raise ValueError("key sequence length must be a multiple of query length")
```

Apply the repeat-ratio validation to both complex and real implementations.

- [ ] **Step 5: Run focused tests and an image parity checkpoint**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_position_math.py tests\test_builder.py -q
.venv\Scripts\python.exe scripts\parity_image.py
```

Expected: tests pass and every reported image difference remains `0.00000000`.

---

### Task 3: Rename and Minimize the Active Attention Module

**Files:**

- Create: `src/ml/components/nn/attention.py`
- Delete: `src/ml/components/nn/modules.py`
- Modify: `src/ml/components/backbone/vit.py`
- Modify: current imports of `AttentionType`, `MultiheadAttention`, and `MultiheadAttentionWrapper`
- Create: `tests/test_attention_math.py`

**Interfaces:**

- Produces: `MultiheadAttention` with the current Vanilla SDPA state-dict layout
- Removes: XFormers, Sparse, Deformable, and FA3 branches and options
- Preserves: packed/separate projection behavior, masks, dropout, and output projection

- [ ] **Step 1: Add a fixed-weight attention equivalence test**

```python
import torch

from src.ml.components.nn.modules import MultiheadAttention


def test_custom_attention_matches_torch_attention():
    torch.manual_seed(0)
    custom = MultiheadAttention(8, 2, dropout=0.0).eval()
    reference = torch.nn.MultiheadAttention(8, 2, dropout=0.0).eval()
    reference.load_state_dict(custom.state_dict())
    query = torch.randn(5, 2, 8)
    padding = torch.tensor([[False, False, False, True, True]]).expand(2, -1)
    actual, _ = custom(query, query, query, key_padding_mask=padding)
    expected, _ = reference(query, query, query, key_padding_mask=padding)
    torch.testing.assert_close(actual, expected)


def test_attention_backward_is_finite():
    torch.manual_seed(0)
    attention = MultiheadAttention(8, 2, dropout=0.0)
    query = torch.randn(5, 2, 8, requires_grad=True)
    output, _ = attention(query, query, query)
    output.square().mean().backward()
    assert torch.isfinite(query.grad).all()
```

- [ ] **Step 2: Run the equivalence test before structural changes**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_attention_math.py -q
```

Expected: pass, establishing the active Vanilla-path contract.

- [ ] **Step 3: Create `attention.py` with only the active path**

Move `multi_head_attention_forward` and `MultiheadAttention` to the new file.
Remove the optional `xformers` import, `AttentionType`, `attn_type`,
`attn_sparsity`, `sparsity`, and `use_fa3`. Keep the current projection, mask
normalization, SDPA call, output projection, and optional weight computation in
the same order. Delete `MultiheadAttentionWrapper`; import `MultiheadAttention`
directly. Update `tests/test_attention_math.py` to import the new module path.

- [ ] **Step 4: Make ViT attention Vanilla-only**

Remove `attn_type` and `use_fa3` parameters from `Attention`, `Block`, and `ViT`.
Keep `use_rope_real`. Replace the backend branch with the current Vanilla SDPA
body and retain the existing query/key/value layout and RoPE application.

- [ ] **Step 5: Update imports and verify no removed backend remains**

Run:

```powershell
rg -n "xformers|AttentionType|attn_type|attn_sparsity|use_fa3|MultiheadAttentionWrapper" src\ml\components\nn src\ml\components\backbone src\ml\components\grounding
```

Expected: no matches in the listed core paths.

- [ ] **Step 6: Run attention, backbone, and grounding tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_attention_math.py tests\test_backbone_package.py tests\test_ground_blocks.py tests\test_builder.py -q
```

Expected: all tests pass.

---

### Task 4: Split Active Grounding and Video Transformer Implementations

**Files:**

- Modify: `src/ml/components/transformer/decoder.py`
- Create: `src/ml/components/transformer/video.py`
- Rename by recreation: `src/ml/components/transformer/wrapper.py` to `src/ml/components/transformer/model.py`
- Modify: `src/ml/components/video/create.py`
- Modify: `src/ml/components/video/tracking_model.py`
- Create: `tests/test_transformer_structure.py`

**Interfaces:**

- Keeps in `decoder.py`: `TransformerDecoderLayer`, `TransformerDecoder`
- Produces in `video.py`: `RotaryAttention`, `VideoDecoderLayer`, `VideoTransformerEncoder`
- Produces in `model.py`: `Transformer`
- Removes: unused decoder v1/v2 and unused cross-attention encoder

- [ ] **Step 1: Add structure tests for the target classes**

```python
import src.ml.components.transformer.decoder as decoder
from src.ml.components.transformer.model import Transformer
from src.ml.components.transformer.video import (
    RotaryAttention,
    VideoDecoderLayer,
    VideoTransformerEncoder,
)


def test_transformer_implementations_are_split_by_use():
    assert not hasattr(decoder, "TransformerDecoderLayerv1")
    assert not hasattr(decoder, "TransformerDecoderLayerv2")
    assert not hasattr(decoder, "TransformerEncoderCrossAttention")
    assert RotaryAttention.__module__.endswith("transformer.video")
    assert VideoDecoderLayer.__module__.endswith("transformer.video")
    assert VideoTransformerEncoder.__module__.endswith("transformer.video")
    assert Transformer.__module__.endswith("transformer.model")
```

- [ ] **Step 2: Verify the target imports fail**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_transformer_structure.py -q
```

Expected: collection fails because `model.py` and `video.py` do not exist.

- [ ] **Step 3: Trim `decoder.py` to the two active grounding classes**

Keep lines/classes corresponding to the current `TransformerDecoderLayer` and
`TransformerDecoder`. Preserve their method bodies and parameter names except
for the already approved math-operation imports. Remove the unused
`TransformerEncoderCrossAttention`, `TransformerDecoderLayerv1`, and
`TransformerDecoderLayerv2` definitions.

- [ ] **Step 4: Move and rename the active video definitions**

Move the current `functional_attention`, `SimpleRoPEAttention`,
`DecoupledTransformerDecoderLayerv2`, and
`TransformerEncoderDecoupledCrossAttention` bodies to `video.py`, renaming them
to `_scaled_attention`, `RotaryAttention`, `VideoDecoderLayer`, and
`VideoTransformerEncoder`. Remove `use_fa3`; the helper always uses the current
SDPA backend list. Keep `use_rope_real` and all active tensor operations intact.

- [ ] **Step 5: Rename `TransformerWrapper` to `Transformer`**

```python
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, d_model: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = d_model
        self.reset_parameters()

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1 and not any(
                key in name for key in ("box_embed", "query_embed", "reference_points")
            ):
                nn.init.xavier_uniform_(parameter)
```

- [ ] **Step 6: Update video construction imports and names**

`components/video/create.py` imports the three classes from `transformer.video`
and `Transformer` from `transformer.model`. Its transformer constructor no
longer accepts or passes `use_fa3`. `tracking_model.py` also drops that option.

- [ ] **Step 7: Run focused tests and all three predictor tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_transformer_structure.py tests\test_reviewed_bugs.py tests\test_single_predict.py tests\test_ground_predict.py tests\test_video_predict.py -q
```

Expected: all tests pass, including presence-logit clamping.

---

### Task 5: Reorganize Image and Grounding Blocks Around Their Responsibilities

**Files:**

- Create: `src/ml/blocks/image/__init__.py`
- Create: `src/ml/blocks/image/features.py`
- Create: `src/ml/blocks/image/prompt.py`
- Create: `src/ml/blocks/image/masks.py`
- Create: `src/ml/blocks/grounding/__init__.py`
- Create: `src/ml/blocks/grounding/tokens.py`
- Create: `src/ml/blocks/grounding/image.py`
- Create: `src/ml/blocks/grounding/prompt.py`
- Create: `src/ml/blocks/grounding/decoder.py`
- Modify: `src/ml/blocks/vision.py`
- Modify: `src/ml/blocks/video_feat.py`
- Delete: old flat `cond.py`, `ground_*.py`, and `sam_*.py` block files
- Delete: `src/ml/components/backbone/create.py`
- Delete: `src/ml/components/grounding/create.py`
- Modify: `src/ml/components/video/tracking_model.py`
- Modify: block imports in tests

**Interfaces:**

- Produces: `VisionEncoder`, `ImageFeatures`, `ImagePromptEncoder`, `ImageMaskDecoder`
- Produces: `VisualTokens`, `GroundingImage`, `GroundingPromptEncoder`, `GroundingDecoder`
- Preserves: existing block state-dict field names during this task
- Preserves: optional geometry prompt for visual-token-only grounding

- [ ] **Step 1: Change package-structure tests to the approved block names**

Add assertions that each target module exists, each target class reports the
target module, and the old flat files do not exist. Update `test_ground_blocks`
to import `GroundingDecoder` from `src.ml.blocks.grounding.decoder`.

- [ ] **Step 2: Verify the structure tests fail before the move**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_package_structure.py tests\test_ground_blocks.py -q
```

Expected: target module imports fail.

- [ ] **Step 3: Recreate image blocks with approved names**

Move bodies without formula changes:

```text
sam_image.py  -> image/features.py : SamImage  -> ImageFeatures
sam_prompt.py -> image/prompt.py   : SamPrompt -> ImagePromptEncoder
sam_mask.py   -> image/masks.py    : SamMask   -> ImageMaskDecoder
```

Rename `tensor` helpers to `unwrap_tensor` and `from_ckpt` methods to
`load_weights`. During this task, keep the current checkpoint prefixes so each
move remains behavior-compatible.

- [ ] **Step 4: Recreate grounding blocks with approved names**

```text
cond.py          -> grounding/tokens.py  : VisualCond   -> VisualTokens
ground_image.py  -> grounding/image.py   : GroundImage  -> GroundingImage
ground_prompt.py -> grounding/prompt.py  : GroundPrompt -> GroundingPromptEncoder
ground_dec.py    -> grounding/decoder.py : GroundDec    -> GroundingDecoder
```

Rename `seq` to `flatten_spatial`, `tensor` to `unwrap_tensor`, `out` to
`unpack_output`, `score_and_box` to `predict_detections`, and `segment` to
`predict_masks`. Keep the current probability, box-refinement, and segmentation
formulas unchanged.

- [ ] **Step 5: Move construction to the owning blocks**

- `vision.py` owns `make_vision_backbone` and `VisionEncoder`.
- `grounding/prompt.py` owns `_make_geometry_encoder`.
- `grounding/decoder.py` owns `_make_attention`, `_make_transformer`,
  `_make_scorer`, and `_make_segmentation_head`.
- `video_feat.py` imports `make_vision_backbone` from the same block layer.
- `components/video/tracking_model.py` requires its injected `backbone` instead
  of importing the deleted component-level backbone factory.

- [ ] **Step 6: Run block, builder, and checkpoint tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_ground_blocks.py tests\test_builder.py tests\test_checkpoint.py tests\test_package_structure.py -q
```

Expected: all tests pass with the reorganized blocks.

- [ ] **Step 7: Run image and grounding CUDA parity**

```powershell
.venv\Scripts\python.exe scripts\parity_image.py
.venv\Scripts\python.exe scripts\parity_ground.py
```

Expected: every reported difference remains zero.

---

### Task 6: Split the Model Module into a One-Way Model Package

**Files:**

- Create: `src/ml/model/__init__.py`
- Create: `src/ml/model/image.py`
- Create: `src/ml/model/grounding.py`
- Create: `src/ml/model/video.py`
- Delete: `src/ml/model.py`
- Modify: `src/build.py`
- Modify: model imports in tests and parity scripts

**Interfaces:**

- Preserves package imports: `from src.ml.model import Sam3ImageModel`
- Produces separate `Sam3ImageModel`, `Sam3GroundingModel`, and `Sam3VideoModel` modules
- Keeps predictors dependent only on public model methods

- [ ] **Step 1: Update model-location tests to the new package**

```python
from src.ml.model import Sam3GroundingModel, Sam3ImageModel, Sam3VideoModel


def test_models_are_split_by_workflow():
    assert Sam3ImageModel.__module__ == "src.ml.model.image"
    assert Sam3GroundingModel.__module__ == "src.ml.model.grounding"
    assert Sam3VideoModel.__module__ == "src.ml.model.video"
```

- [ ] **Step 2: Verify the location test fails**

Run the exact new test. Expected: classes still report `src.ml.model`.

- [ ] **Step 3: Recreate each model in its own file**

Move each current class body to its matching file and update block imports to
Task 5 names. Rename model `from_ckpt` methods to `load_weights`. Keep
`encode_image`, `encode_prompt`, `decode_masks`, `decode`, video runtime
delegation, and forward behavior unchanged. Rename the retrieval method
`image_pe` to `get_image_position_encoding` and update predictor call sites.

`model/__init__.py` uses explicit re-exports without `__all__`:

```python
from .grounding import Sam3GroundingModel as Sam3GroundingModel
from .image import Sam3ImageModel as Sam3ImageModel
from .video import Sam3VideoModel as Sam3VideoModel
```

- [ ] **Step 4: Update builders and parity imports**

Keep the public `src.build` function names. Update only module paths and renamed
model load methods.

- [ ] **Step 5: Run model and predictor tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_builder.py tests\test_single_predict.py tests\test_ground_predict.py tests\test_video_predict.py tests\test_architecture.py -q
```

Expected: all tests pass and predictors do not import model internals.

---

### Task 7: Centralize Strict Official SAM3.1 Loading

**Files:**

- Modify: `src/io/checkpoint.py`
- Modify: every block/model `load_weights` method created in Tasks 5 and 6
- Modify: `tests/test_checkpoint.py`
- Modify: relevant builder tests

**Interfaces:**

- Produces: `Checkpoint.load_block(name: str, module: nn.Module) -> None`
- Produces canonical logical prefixes for image vision/features/prompt/masks and grounding prompt/decoder
- Rejects missing/unexpected official weights
- Does not map old LoRA adapter keys

- [ ] **Step 1: Add strict-load and canonical-prefix tests**

```python
import pytest
import torch
from torch import nn

from src.io.checkpoint import Checkpoint


def test_checkpoint_load_block_is_strict_and_names_the_block():
    checkpoint = Checkpoint(state={"image.features.weight": torch.ones(1)}, ignored=[])
    module = nn.Linear(1, 1, bias=False)
    checkpoint.load_block("image.features", module)
    assert module.weight.item() == 1

    with pytest.raises(RuntimeError, match="image.features"):
        Checkpoint(state={}, ignored=[]).load_block("image.features", module)
```

Update representative remap assertions to the new logical prefixes.

- [ ] **Step 2: Verify tests fail because `load_block` is absent**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_checkpoint.py -q
```

Expected: failure on missing method or old logical prefixes.

- [ ] **Step 3: Implement one strict loading owner**

```python
def load_block(self, name: str, module: torch.nn.Module) -> None:
    state = self.block_state(name)
    if not state:
        raise RuntimeError(f"SAM3.1 checkpoint block is empty: {name}")
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(f"failed to load SAM3.1 block {name}: {error}") from error
```

Update aliases to canonical prefixes such as `image.vision`,
`image.features`, `image.prompt`, `image.masks`, `grounding.prompt`, and
`grounding.decoder`. Do not add any LoRA alias.

- [ ] **Step 4: Delegate all base-weight loading to `Checkpoint.load_block`**

Each image, grounding, and video block/model uses exactly one logical block
name and does not accept a `strict=False` option. `VisualTokens.load_weights`
continues to load the separate cached token mapping directly.

- [ ] **Step 5: Run unit tests and full local load parity**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_checkpoint.py tests\test_builder.py -q
.venv\Scripts\python.exe scripts\parity_image.py
.venv\Scripts\python.exe scripts\parity_ground.py
```

Expected: strict loads succeed and both parity scripts report zero differences.

---

### Task 8: Replace Historical Structure Blacklists with Dependency Invariants

**Files:**

- Modify: `tests/test_architecture.py`
- Modify: `tests/test_package_structure.py`
- Modify: `tests/test_backbone_package.py`
- Modify: `tests/test_sam_package.py`
- Modify: `README.md`

**Interfaces:**

- Produces: an AST import-rank test for the approved one-way dependency flow
- Produces: concise target-path assertions rather than a historical blacklist
- Documents: the new repository layout and model import paths

- [ ] **Step 1: Rewrite the architecture ranks**

Use these logical ranks:

```python
LAYERS = {
    "src.data": 0,
    "src.io": 0,
    "src.ops": 0,
    "src.ml.runtime": 0,
    "src.ml.components": 1,
    "src.ml.blocks": 2,
    "src.ml.model": 3,
    "src.build": 4,
    "src.predict": 4,
    "scripts": 5,
}
```

Keep the longest-prefix rank lookup and fail with exact source/import pairs.

- [ ] **Step 2: Replace the path blacklist**

Assert only the approved target directories/files, the absence of the replaced
flat block files, and public predictor/model exports. Remove assertions for
every abandoned historical name.

- [ ] **Step 3: Update README layout and entry points**

Document `src/ml/model/`, `src/ml/blocks/image/`, and
`src/ml/blocks/grounding/`. Keep the official SAM3.1 local-path and cached
visual-token notes.

- [ ] **Step 4: Run all structure and package tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_architecture.py tests\test_package_structure.py tests\test_backbone_package.py tests\test_sam_package.py tests\test_scripts_standalone.py -q
```

Expected: all tests pass with no upward import.

---

### Task 9: Format, Run the Full Suite, and Prove Final Parity

**Files:**

- Modify only files already touched when formatting

**Interfaces:**

- Produces: a verified core refactor with exact current-machine parity
- Leaves: protected reference, checkpoint, and asset trees untouched

- [ ] **Step 1: Format changed Python files with Black**

Pass the explicit changed Python file list from `git diff --name-only` to Black;
do not format unrelated video tracker files.

- [ ] **Step 2: Run Ruff on changed Python files**

Expected: zero errors. Fix only errors caused or exposed in touched files.

- [ ] **Step 3: Run the complete test suite**

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all updated and newly added tests pass.

- [ ] **Step 4: Run CPU construction smoke checks**

Construct `Sam3ImageModel(path=None)` and `Sam3GroundingModel(path=None)` one at
a time, assert their parameters and position encodings remain on CPU, then
delete each model before constructing the next.

Expected: neither constructor attempts a CUDA allocation.

- [ ] **Step 5: Run all parity scripts**

```powershell
.venv\Scripts\python.exe scripts\parity_image.py
.venv\Scripts\python.exe scripts\parity_ground.py
.venv\Scripts\python.exe scripts\parity_video.py
```

Expected: image and grounding differences remain exactly zero. Video behavior
also remains at its pre-refactor parity because only shared transformer paths
and the thin wrapper moved.

- [ ] **Step 6: Verify repository boundaries and final diff**

```powershell
git status --short
git diff --check
git diff --stat
```

Expected: no changes under `sam3-main/`, `weight/`, or `asset/`; no whitespace
errors; only the approved core, boundary imports, tests, README, and planning
documents are changed.
