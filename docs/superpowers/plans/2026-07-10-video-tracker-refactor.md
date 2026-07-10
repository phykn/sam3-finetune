# SAM3.1 Video Tracker Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the option-heavy, method-injected video tracker with a direct forward-only inference runtime while preserving strict SAM3.1 loading, dynamic mask additions, object removal, and exact parity.

**Architecture:** Keep weight-bearing tensor helpers under `src/ml/components/video/`, but move stateful orchestration into a `src/ml/model/video/` package. `Sam3VideoModel` owns one fixed `VideoRuntime`; `VideoPredictor` exposes streaming start, predict, mask addition, and object removal without detector coupling.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Pillow, pytest, Black, Ruff

**Status:** Implemented on `codex/video-tracker-refactor`; final verification is recorded by the executing session.

## Global Constraints

- Work from the repository root with `.venv\Scripts\python.exe`.
- Keep the dependency direction `components -> blocks -> model -> predict`.
- Preserve official SAM3.1 video checkpoint parameter names and strict local loading.
- Preserve forward inference, multiplexing, mask-based additions, removal, and output parity.
- Do not load Hugging Face, the VLM, or the text encoder.
- Do not add old LoRA checkpoint compatibility.
- Do not modify, stage, commit, or push `sam3-main/`, `weight/`, or `asset/`.
- Keep code short and explicit; do not add dataclasses, wrappers, lazy imports, `__all__`, Meta headers, or `from __future__ import annotations`.
- Do not commit or push unless the user explicitly requests it.
- Every production change follows a witnessed red-green test cycle.

---

## File Structure

### Create

- `src/ml/model/video/__init__.py`: export `Sam3VideoModel`.
- `src/ml/model/video/model.py`: assemble blocks, runtime, and checkpoint loading.
- `src/ml/model/video/runtime.py`: fixed inference-only neural runtime.
- `src/ml/model/video/state.py`: create state, register ids, validate cached frames.
- `src/ml/model/video/masks.py`: mask-based object addition and reconditioning.
- `src/ml/model/video/objects.py`: explicit removal and output slicing.
- `src/ml/model/video/propagate.py`: preflight and forward-only propagation.
- `tests/test_video_runtime_structure.py`: dependency and removed-path contract.
- `tests/test_video_state.py`: state, id, frame, and validation invariants.
- `tests/test_video_multiplex_math.py`: mux/demux and removal invariants.
- `scripts/parity_video_dynamic.py`: upstream parity for add/remove flow.

### Modify

- `src/ml/model/__init__.py`: import the video package.
- `src/ml/components/video/tracker/frame/inference.py`: remove reverse and point-only inputs from the supported call path.
- `src/ml/components/video/tracker/runtime/step.py`: remove correction sampling and implement the direct mask/propagation step.
- `src/ml/components/video/tracker/runtime/initial.py`: keep only mask-as-output and propagation modes.
- `src/ml/components/video/tracker/runtime/output.py`: keep output and memory writes used by inference.
- `src/ml/components/video/tracker/runtime/init.py`: initialize only supported inference attributes.
- `src/ml/components/video/tracker/runtime/params.py`: retain only active inference parameter initialization.
- `src/ml/components/video/tracker/interaction/masks.py`: move orchestration to the model package or reduce it to tensor helpers.
- `src/ml/components/video/tracker/interaction/objects.py`: move orchestration to the model package or reduce it to tensor helpers.
- `src/ml/components/video/tracker/interaction/propagation.py`: replace with model-owned forward propagation.
- `src/predict/video.py`: add public mask-add and object-remove methods.
- `src/predict/video_ops/session.py`: validate and adapt predictor state operations.
- `scripts/parity_video.py`: use the final forward-only API.
- `tests/test_architecture.py`: enforce model ownership of stateful video flow.
- `tests/test_builder.py`: follow the video model package and runtime constructor.
- `tests/test_package_structure.py`: lock final paths.
- `tests/test_video_predict.py`: cover predictor add/remove behavior.
- `tests/test_checkpoint.py`: retain strict video mapping coverage.
- `README.md`: document the final video predictor API and automatic-detection boundary.

### Delete After Import Proof

- `src/ml/model/video.py`
- `src/ml/components/video/tracking_model.py`
- `src/ml/components/video/tracker/model.py`
- `src/ml/components/video/tracker/tracking.py`
- `src/ml/components/video/tracker/runtime/compile.py`
- `src/ml/components/video/tracker/runtime/correction.py`
- `src/ml/components/video/tracker/runtime/loop.py`
- `src/ml/components/video/tracker/prompt/inputs.py`
- `src/ml/components/video/tracker/prompt/order.py`
- `src/ml/components/video/tracker/prompt/sampling.py`
- `src/ml/components/video/tracker/prompt/utils.py`
- point-only interaction files that have no remaining importer

---

### Task 1: Lock the New Package and Public API

**Files:**
- Create: `tests/test_video_runtime_structure.py`
- Modify: `tests/test_package_structure.py`
- Modify: `tests/test_video_predict.py`

**Interfaces:**
- Consumes: existing `Sam3VideoModel`, `VideoPredictor`, and fake predictor model.
- Produces: the required package paths and `add_masks` / `remove_objects` signatures.

- [ ] **Step 1: Write the failing package test**

```python
from inspect import signature
from pathlib import Path

from src.ml.model.video.runtime import VideoRuntime
from src.predict.video import VideoPredictor


ROOT = Path(__file__).resolve().parents[1] / "src" / "ml"


def test_video_runtime_has_only_inference_dependencies():
    params = list(signature(VideoRuntime).parameters)
    assert params == [
        "backbone",
        "transformer",
        "maskmem_backbone",
        "multiplex_controller",
    ]
    assert not hasattr(VideoRuntime, "forward_tracking")
    assert not hasattr(VideoRuntime, "prepare_prompt_inputs")


def test_stateful_video_flow_lives_in_model_package():
    assert (ROOT / "model" / "video" / "runtime.py").exists()
    assert (ROOT / "model" / "video" / "state.py").exists()
    assert not (ROOT / "components" / "video" / "tracker" / "tracking.py").exists()


def test_video_predictor_exposes_dynamic_mask_api():
    assert list(signature(VideoPredictor.add_masks).parameters) == [
        "self",
        "state",
        "masks",
        "obj_ids",
        "frame_idx",
    ]
    assert list(signature(VideoPredictor.remove_objects).parameters) == [
        "self",
        "state",
        "obj_ids",
        "strict",
    ]
```

- [ ] **Step 2: Run the structural test and witness RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_video_runtime_structure.py -q
```

Expected: collection fails because `src.ml.model.video` is not a package and `VideoRuntime` does not exist.

- [ ] **Step 3: Extend the fake model and write failing predictor tests**

Add to `FakeVideoModel` in `tests/test_video_predict.py`:

```python
def add_masks(self, state, frame_idx, obj_ids, masks):
    self.calls.append(("add_masks", frame_idx, list(obj_ids), tuple(masks.shape)))
    state["obj_ids"] = list(dict.fromkeys([*state.get("obj_ids", []), *obj_ids]))
    return frame_idx, state["obj_ids"], None, masks[:, None]

def remove_objects(self, state, obj_ids, strict=True):
    self.calls.append(("remove_objects", list(obj_ids), strict))
    state["obj_ids"] = [x for x in state.get("obj_ids", []) if x not in obj_ids]
    return state["obj_ids"], []
```

Add tests:

```python
def test_video_predictor_adds_masks_on_latest_cached_frame():
    model = FakeVideoModel()
    predictor = VideoPredictor(model, device="cpu")
    state = predictor.start(Image.new("RGB", (4, 5)), np.ones((5, 4), dtype=bool))

    ids = predictor.add_masks(
        state,
        np.ones((1, 5, 4), dtype=bool),
        [9],
    )

    assert ids == [7, 9]
    assert model.calls[-2][:3] == ("add_masks", 0, [9])
    assert model.calls[-1] == ("preflight", True)


def test_video_predictor_removes_objects():
    model = FakeVideoModel()
    predictor = VideoPredictor(model, device="cpu")
    state = predictor.start(Image.new("RGB", (4, 5)), np.ones((5, 4), dtype=bool))

    ids = predictor.remove_objects(state, [7])

    assert ids == []
    assert model.calls[-1] == ("remove_objects", [7], True)
```

- [ ] **Step 4: Run the focused predictor tests and witness RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_video_predict.py -q
```

Expected: the new tests fail because `VideoPredictor` has no `add_masks` or `remove_objects` method.

---

### Task 2: Add Explicit State and Multiplex Invariants

**Files:**
- Create: `tests/test_video_state.py`
- Create: `tests/test_video_multiplex_math.py`
- Create: `src/ml/model/video/state.py`

**Interfaces:**
- Produces: `create_state`, `add_object`, `cached_frame`, and `forward_frames`.
- Consumes: `MultiplexController` and plain inference dictionaries.

- [ ] **Step 1: Write failing state tests**

```python
import pytest
import torch

from src.ml.model.video.state import add_object, cached_frame, create_state, forward_frames


def test_state_registers_unique_ordered_ids():
    state = create_state(
        num_frames=3,
        video_height=5,
        video_width=7,
        cached_features={0: ("image", "features")},
        device="cpu",
    )
    assert add_object(state, 8) == 0
    assert add_object(state, 3) == 1
    assert add_object(state, 8) == 0
    assert state["obj_ids"] == [8, 3]


def test_cached_frame_rejects_missing_index():
    state = create_state(
        num_frames=2,
        video_height=5,
        video_width=7,
        cached_features={0: ("image", "features")},
        device="cpu",
    )
    with pytest.raises(KeyError, match="frame 1 is not cached"):
        cached_frame(state, 1)


def test_forward_frames_never_returns_future_or_reverse_indices():
    assert list(forward_frames(start=2, count=3, num_frames=7)) == [2, 3, 4]
    assert list(forward_frames(start=5, count=4, num_frames=7)) == [5, 6]
```

- [ ] **Step 2: Run state tests and witness RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_video_state.py -q
```

Expected: import fails because `src.ml.model.video.state` does not exist.

- [ ] **Step 3: Implement the smallest state API**

`src/ml/model/video/state.py` starts with:

```python
from collections import OrderedDict

import torch


OUTPUT_KEYS = ("cond_frame_outputs", "non_cond_frame_outputs")


def output_store(factory=dict):
    return {key: factory() for key in OUTPUT_KEYS}


def create_state(
    *,
    num_frames,
    video_height,
    video_width,
    cached_features=None,
    device="cuda",
    offload_state_to_cpu=False,
):
    if min(num_frames, video_height, video_width) <= 0:
        raise ValueError("frame count and video size must be positive")
    device = torch.device(device)
    return {
        "num_frames": num_frames,
        "video_height": video_height,
        "video_width": video_width,
        "device": device,
        "storage_device": torch.device("cpu") if offload_state_to_cpu else device,
        "cached_features": {} if cached_features is None else cached_features,
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "output_dict": output_store(),
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "consolidated_frame_inds": output_store(set),
        "frames_already_tracked": {},
        "tracking_has_started": False,
        "first_ann_frame_idx": None,
        "constants": {},
        "multiplex_state": None,
    }


def add_object(state, obj_id):
    current = state["obj_id_to_idx"].get(obj_id)
    if current is not None:
        return current
    index = len(state["obj_ids"])
    state["obj_id_to_idx"][obj_id] = index
    state["obj_idx_to_id"][index] = obj_id
    state["obj_ids"].append(obj_id)
    state["point_inputs_per_obj"][index] = {}
    state["mask_inputs_per_obj"][index] = {}
    state["output_dict_per_obj"][index] = output_store()
    state["temp_output_dict_per_obj"][index] = output_store()
    return index


def cached_frame(state, frame_idx):
    try:
        return state["cached_features"][frame_idx]
    except KeyError:
        raise KeyError(f"frame {frame_idx} is not cached") from None


def forward_frames(start, count, num_frames):
    if start < 0 or count <= 0:
        raise ValueError("start must be non-negative and count must be positive")
    return range(start, min(start + count, num_frames))
```

- [ ] **Step 4: Add multiplex characterization tests**

```python
import torch

from src.ml.components.video.tracker.multiplex.state import MultiplexController


def make_state(count=3):
    controller = MultiplexController(4, eval_multiplex_count=4).eval()
    return controller.get_state(
        count,
        torch.device("cpu"),
        torch.float32,
        random=False,
        object_ids=list(range(10, 10 + count)),
    )


def test_demux_is_left_inverse_of_mux():
    state = make_state()
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.testing.assert_close(state.demux(state.mux(values)), values)


def test_removal_preserves_remaining_row_order():
    state = make_state()
    values = torch.tensor([[1.0], [2.0], [3.0]])
    multiplexed = state.mux(values)
    state.remove_objects([1])
    remaining = multiplexed[:, :, :]
    remaining = remaining.reshape(-1, 1)[state.get_valid_object_mask().reshape(-1)]
    assert state.object_ids == [10, 12]
    assert remaining.flatten().tolist() == [1.0, 3.0]
```

- [ ] **Step 5: Run the state and multiplex tests GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_video_state.py tests/test_video_multiplex_math.py -q
```

Expected: all tests pass.

---

### Task 3: Build One Fixed Inference Runtime

**Files:**
- Create: `src/ml/model/video/runtime.py`
- Create: `src/ml/model/video/model.py`
- Create: `src/ml/model/video/__init__.py`
- Modify: `src/ml/model/__init__.py`
- Modify: `src/ml/components/video/tracker/runtime/init.py`
- Modify: `src/ml/components/video/tracker/runtime/params.py`
- Delete: `src/ml/model/video.py`
- Delete: `src/ml/components/video/tracking_model.py`
- Delete: `src/ml/components/video/tracker/tracking.py`
- Delete: `src/ml/components/video/tracker/model.py`

**Interfaces:**
- Produces: `VideoRuntime(backbone, transformer, maskmem_backbone, multiplex_controller)`.
- Preserves: runtime registered attributes and the video checkpoint state dictionary.

- [ ] **Step 1: Confirm the structural test is RED before production edits**

Run the Task 1 structural test again and confirm the package import failure.

- [ ] **Step 2: Implement fixed runtime construction**

The constructor accepts four dependencies and calls an inference-only initializer:

```python
class VideoRuntime(nn.Module):
    def __init__(self, backbone, transformer, maskmem_backbone, multiplex_controller):
        super().__init__()
        init_runtime(
            self,
            backbone=backbone,
            transformer=transformer,
            maskmem_backbone=maskmem_backbone,
            multiplex_controller=multiplex_controller,
        )
```

`init_runtime` uses the current SAM3.1 values exactly:

```python
IMAGE_SIZE = 1008
BACKBONE_STRIDE = 14
NUM_MASKMEM = 7
MAX_OBJECT_POINTERS = 16
MAX_CONDITION_FRAMES = 4
MULTIMASK_OUTPUTS = 3
```

It keeps the current active values for sigmoid memory encoding, object pointers,
output suppression embeddings, mask conditioning, temporal position encoding,
and memory selection. It does not initialize training probabilities, random
prompt generators, compile flags, transition sampling, or reverse-only flags.

- [ ] **Step 3: Replace injected methods with direct methods**

Use explicit calls such as:

```python
def forward_image(self, image, **kwargs):
    return frame_features.forward_image(self, image, **kwargs)

def prepare_memory(self, **kwargs):
    return memory_conditioning.prepare_memory_conditioned_features(self, **kwargs)

def decode_masks(self, **kwargs):
    return decoder_heads.forward_sam_heads(self, **kwargs)
```

Do not use class-body assignments whose right-hand side is an imported function.

- [ ] **Step 4: Assemble the model package**

`src/ml/model/video/model.py` keeps the existing block attributes:

```python
class Sam3VideoModel(nn.Module):
    def __init__(self, path=None):
        super().__init__()
        self.video_feat = VideoFeat()
        self.video_mem = VideoMem()
        self.video_track = VideoTrack()
        self.runtime = VideoRuntime(
            self.video_feat,
            self.video_track.transformer,
            self.video_mem.encoder,
            MultiplexController(16, eval_multiplex_count=16),
        )
        self.runtime.image_pe_layer = self.video_track.image_pe
        self.runtime.sam_mask_decoder = self.video_track.mask_decoder
        self.runtime.output_valid_embed = self.video_track.output_valid_embed
        self.runtime.output_invalid_embed = self.video_track.output_invalid_embed
        if path is not None:
            self.load_weights(Checkpoint.load(path))
```

Keep `load_weights` calling `ckpt.load_block("video", self.runtime)`.

- [ ] **Step 5: Run focused structure and builder tests GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_video_runtime_structure.py tests/test_builder.py tests/test_checkpoint.py -q
```

Expected: the new runtime imports, model construction succeeds, and strict-load mocks retain their existing calls.

---

### Task 4: Move State Mutation and Make Propagation Forward-Only

**Files:**
- Create: `src/ml/model/video/masks.py`
- Create: `src/ml/model/video/objects.py`
- Create: `src/ml/model/video/propagate.py`
- Modify: `src/ml/model/video/runtime.py`
- Modify: `src/ml/components/video/tracker/frame/inference.py`
- Modify: `src/ml/components/video/tracker/runtime/step.py`
- Modify: `src/ml/components/video/tracker/runtime/initial.py`
- Modify: `tests/test_video_state.py`

**Interfaces:**
- Produces: `add_masks`, `remove_objects`, `preflight`, and `propagate`.
- Removes: reverse processing and point-correction arguments from the supported call chain.

- [ ] **Step 1: Add failing forward-only tests**

```python
import pytest

from src.ml.model.video.state import forward_frames


def test_forward_frames_rejects_empty_count():
    with pytest.raises(ValueError, match="count must be positive"):
        forward_frames(0, 0, 3)


def test_runtime_has_no_reverse_parameter():
    from inspect import signature
    from src.ml.model.video.propagate import propagate

    assert "reverse" not in signature(propagate).parameters
```

Run the focused tests and confirm failure because `propagate` does not exist.

- [ ] **Step 2: Move mask orchestration**

Move the active behavior from `interaction/masks.py` to
`src/ml/model/video/masks.py`, rename `add_new_masks` to `add_masks`, and add
public validation before tensor work:

```python
obj_ids = list(obj_ids)
if not obj_ids:
    raise ValueError("obj_ids must not be empty")
if len(obj_ids) != len(set(obj_ids)):
    raise ValueError("obj_ids in one request must be unique")
if masks.ndim != 3 or masks.shape[0] != len(obj_ids):
    raise ValueError("masks must have shape N x H x W matching obj_ids")
cached_frame(state, frame_idx)
```

Keep the current bilinear resize, `align_corners=False`, antialiasing, `> 0.5`
prompt threshold, multiplex allocation, overlap suppression, and consolidation.

- [ ] **Step 3: Move removal orchestration**

Move the active behavior from `interaction/objects.py` to
`src/ml/model/video/objects.py`. Replace calls to point-clearing APIs with direct
removal of the selected entries from `mask_inputs_per_obj` and the retained
empty compatibility point store. Keep strict unknown-id rejection, multiplex
bucket slicing, memory tensor slicing, and contiguous reindexing.

- [ ] **Step 4: Implement forward-only preflight and propagation**

`propagate` has this boundary:

```python
@torch.inference_mode()
def propagate(
    model,
    state,
    start_frame_idx,
    max_frame_num_to_track,
    tqdm_disable=False,
    run_mem_encoder=True,
):
```

It uses `forward_frames`, never accepts `reverse`, always passes
`track_in_reverse=False` to lower tensor helpers until that parameter is removed,
and yields:

```python
frame_idx, list(state["obj_ids"]), low_res_masks, video_res_masks, object_logits
```

- [ ] **Step 5: Remove correction sampling from the inference step**

Replace `run_track_step_aux` with a direct sequence:

```text
prepare mode and features
-> run mask-as-output or propagation
-> write current outputs
-> optionally merge new object masks
-> encode memory
-> trim stored output
```

Delete the call to `apply_correction_points`. `frames_to_add_correction_pt`,
`gt_masks`, `prev_sam_mask_logits`, and point-only modes leave the supported
signature.

- [ ] **Step 6: Run focused state, builder, and reviewed-bug tests GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_video_state.py tests/test_builder.py tests/test_reviewed_bugs.py -q
```

Expected: all tests pass with forward-only propagation.

---

### Task 5: Expose Dynamic Add and Remove Through the Predictor

**Files:**
- Modify: `src/ml/model/video/model.py`
- Modify: `src/predict/video.py`
- Modify: `src/predict/video_ops/session.py`
- Modify: `tests/test_video_predict.py`

**Interfaces:**
- Produces: predictor `add_masks` and `remove_objects`, both returning ordered ids.

- [ ] **Step 1: Confirm Task 1 predictor tests are RED**

Run the two new predictor tests and confirm missing methods.

- [ ] **Step 2: Add model boundary methods**

```python
def add_masks(self, state, frame_idx, obj_ids, masks):
    return self.runtime.add_masks(state, frame_idx, obj_ids, masks)

def remove_objects(self, state, obj_ids, strict=True):
    return self.runtime.remove_objects(state, obj_ids, strict=strict)
```

No predictor imports model internals below `Sam3VideoModel`.

- [ ] **Step 3: Add session adapters**

```python
def add_masks(model, session_state, masks, obj_ids, device, frame_idx=None):
    tracker_state = session_state["state"]
    if frame_idx is None:
        frame_idx = session_state["next_frame"] - 1
    if frame_idx not in tracker_state["cached_features"]:
        raise KeyError(f"frame {frame_idx} is not cached")
    masks = mask_tensor(masks, device)
    _, ids, _, _ = model.add_masks(tracker_state, frame_idx, obj_ids, masks)
    model.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)
    return list(ids)


def remove_objects(model, session_state, obj_ids, strict=True):
    ids, _ = model.remove_objects(session_state["state"], obj_ids, strict=strict)
    return list(ids)
```

- [ ] **Step 4: Add thin predictor methods**

Each method uses the predictor autocast context and delegates to `session`.

- [ ] **Step 5: Run predictor tests GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_video_predict.py -q
```

Expected: all predictor tests pass.

---

### Task 6: Delete Unreachable Training and Correction Code

**Files:**
- Delete paths listed in the plan's delete section after reference checks.
- Modify: `src/ml/structures.py`
- Modify: `tests/test_architecture.py`
- Modify: `tests/test_video_runtime_structure.py`

**Interfaces:**
- Preserves: `NestedTensor` and active component imports.
- Removes: training datapoints, prompt sampling, reverse flow, and method injection.

- [ ] **Step 1: Add failing absence and import tests**

Extend the structural test with every deleted path and scan AST class bodies for
assignments whose value is a bare imported function name.

- [ ] **Step 2: Run the test and witness RED**

Expected: current training, prompt, correction, and compile files still exist.

- [ ] **Step 3: Prove each delete target has no active importer**

Run:

```powershell
rg -n "prompt\.(inputs|order|sampling|utils)|runtime\.(loop|compile|correction)|interaction\.points|BatchedDatapoint|FindStage" src tests scripts
```

Only documentation or deletion-contract tests may match before deletion.

- [ ] **Step 4: Delete inactive files and remove stale types**

Remove `FindStage` and `BatchedDatapoint` from `src/ml/structures.py` after the
video training import is gone. Keep `NestedTensor` and its pytree registration.

- [ ] **Step 5: Run architecture and package tests GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_architecture.py tests/test_package_structure.py tests/test_video_runtime_structure.py -q
```

Expected: all tests pass and no deleted module is imported.

---

### Task 7: Add Standard and Dynamic Runtime Parity

**Files:**
- Modify: `scripts/parity_video.py`
- Create: `scripts/parity_video_dynamic.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: local `weight/sam3.1_multiplex.pt` and existing media in `asset/`.
- Produces: exact source/upstream comparisons for standard and add/remove flows.

- [ ] **Step 1: Update standard parity to the forward-only call**

Remove `reverse=False` only from the source call. Keep it for the upstream call
when the upstream signature requires it.

- [ ] **Step 2: Implement dynamic parity sequence**

Use the existing frame cache and comparison helpers. For both source and
upstream:

```text
start with two deterministic masks on frame 0
preflight
propagate frame 1
add object id 3 with a deterministic mask on cached frame 1
preflight
propagate frame 2
remove object id 2
compare remaining ids and stored outputs
```

Print max and mean absolute error for low-resolution masks, video-resolution
masks, and presence logits, plus thresholded-mask XOR.

The reference removal path has a historical empty-row bug. Patch the upstream
instance in memory with the same `local_obj_id_to_idx` lookup used by the source
runtime before comparing removal. Do not edit `sam3-main`.

- [ ] **Step 3: Document the public API boundary**

README example:

```python
state = predictor.start(first_frame, first_mask, obj_id=1)
prediction = predictor.predict(next_frame, state)
ids = predictor.add_masks(state, new_masks, [2, 3])
ids = predictor.remove_objects(state, [2])
```

State that automatic detection requires a separate grounding-to-tracker layer.

- [ ] **Step 4: Run both parity scripts**

Run:

```powershell
.venv\Scripts\python.exe scripts/parity_video.py
.venv\Scripts\python.exe scripts/parity_video_dynamic.py
```

Expected: every floating comparison has max/mean error `0`, every object-id list
matches, and every mask XOR is `0`.

---

### Task 8: Final Verification

**Files:** all changed files

- [ ] **Step 1: Run focused video tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_video_runtime_structure.py tests/test_video_state.py tests/test_video_multiplex_math.py tests/test_video_predict.py tests/test_builder.py tests/test_checkpoint.py tests/test_reviewed_bugs.py -q
```

- [ ] **Step 2: Run the full suite**

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

- [ ] **Step 3: Check changed Python formatting and lint**

```powershell
$tracked = git diff --name-only --diff-filter=ACMR
$untracked = git ls-files --others --exclude-standard
$files = @($tracked; $untracked) | Where-Object { $_ -like '*.py' } | Sort-Object -Unique
.venv\Scripts\python.exe -m black --check -- $files
.venv\Scripts\python.exe -m ruff check -- $files
```

- [ ] **Step 4: Run strict checkpoint construction and both parity scripts**

Construct `Sam3VideoModel` on CPU with the official local checkpoint, then run
the standard and dynamic CUDA parity commands from Task 7.

- [ ] **Step 5: Check scope and protected paths**

```powershell
git diff --check
git status --short -- sam3-main weight asset
git status --short
```

Expected: no protected-path output and only planned source, test, script,
README, spec, and plan changes.
