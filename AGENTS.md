# Agent Instructions

These instructions apply to the whole `D:\code\sam3` workspace.

## Workspace Layout

- `sam3-main/` is the SAM 3 source tree and the directory to use for package,
  test, formatting, notebook, training, and evaluation commands.
- `src/` is the new rewrite target. Build only the minimal modules needed for
  point, box, and mask prompted segmentation there. Keep `sam3-main/` as
  reference material and do not modify it for the rewrite unless explicitly
  asked.
- `weight/` sits outside the source tree. Treat it as local large-artifact
  storage for checkpoints or model weights. Do not move, delete, rename, or
  regenerate files there unless the user explicitly asks.
- Git remote for this project: `https://github.com/phykn/sam3-finetune.git`.
  The workspace may not be initialized yet; if Git is initialized, use this
  remote.
- Never stage, commit, or push `sam3-main/` or `weight/`. They are local
  reference/artifact directories only. The root `.gitignore` excludes them, but
  still verify `git status --short` before any commit or push.

## Project Facts

- This is Meta's SAM 3 Python package for promptable image and video
  segmentation with concept prompts. The checkout also includes SAM 3.1 Object
  Multiplex video tracking code.
- Main package: `sam3-main/sam3/`.
- Main model builder entrypoint: `sam3-main/sam3/model_builder.py`.
- Main public imports currently come from `sam3/__init__.py`:
  `build_sam3_image_model` and `build_sam3_predictor`.
- Important subpackages:
  - `sam3/model/`: image/video model definitions, processors, predictors,
    tokenizers, IO helpers, and SAM 3.1 multiplex tracking.
  - `sam3/train/`: Hydra-based training entrypoint, configs, data loading,
    losses, transforms, optimizers, and distributed utilities.
  - `sam3/eval/`: COCO, cgF1, TETA, HOTA, YTVIS, SA-Co, and conversion helpers.
  - `sam3/agent/`: SAM 3 agent inference wrappers, visualizers, prompts, and
    helper geometry/mask utilities.
  - `sam3/perflib/`: performance kernels and Triton/CUDA-adjacent utilities.
  - `sam3/sam/`: SAM-style prompt encoder, transformer, and mask decoder pieces.
- Supporting material:
  - `examples/`: notebooks for image, video, agent, SA-Co, and SAM 3.1 usage.
  - `scripts/`: speed, qualitative, extraction, and dataset/eval scripts.
  - `assets/`: demo media and diagrams.
  - `test/`: current unit tests.

## Environment And Setup

- Use a workspace-local `.venv` for new `src/` work. Create or activate it from
  `D:\code\sam3`, then install root `requirements.txt` there.
- Run new `src/` commands from the workspace root. Run upstream/reference
  commands from `sam3-main/` unless a task explicitly targets the workspace
  root or `weight/`.
- Install PyTorch separately with the CUDA wheel/index that matches the machine;
  root `requirements.txt` intentionally does not install `torch` or
  `torchvision`.
- The README's documented full environment target is Python 3.12 or higher,
  PyTorch 2.7 or higher, and a CUDA-compatible GPU with CUDA 12.6 or higher.
  `pyproject.toml` lists `requires-python >=3.8`, but prefer the README target
  for full inference/training work unless compatibility is being tested.
- Basic install:

  ```bash
  pip install -e .
  ```

- Development and training install:

  ```bash
  pip install -e ".[dev,train]"
  ```

- Notebook install:

  ```bash
  pip install -e ".[notebooks]"
  ```

- Do not use Hugging Face for the new `src/` rewrite. Load weights only from
  explicit local paths, normally under `weight/`.
- For no-text image grounding in `src/`, do not load the VLM/text encoder at
  runtime. Use a precomputed cached `"visual"` language feature tensor under
  `weight/` and inject it into the grounding backbone output.

## Commands

- Format code:

  ```bash
  ufmt format .
  ```

- Focused unit test:

  ```bash
  python -m pytest test/test_io_utils.py
  ```

- Current full local test path:

  ```bash
  python -m pytest test
  ```

  Note: `pyproject.toml` currently sets `testpaths = ["tests"]`, while this
  checkout has `test/`. Use an explicit test path unless fixing that layout is
  part of the task.

- Training and eval use `sam3/train/train.py` with Hydra config names relative
  to the `sam3.train` config package. Examples from the repo docs:

  ```bash
  python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 1
  python sam3/train/train.py -c configs/odinw13/odinw_text_only_train.yaml --use-cluster 0 --num-gpus 1
  python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_eval.yaml
  ```

  Do not start multi-GPU, SLURM, large dataset, or long training/evaluation jobs
  unless the user explicitly asks for that run.

## Coding Guidelines

- Preserve upstream style and file headers in `sam3-main/` reference work. For
  new `src/` rewrite code, do not add Meta copyright headers or `# pyre-unsafe`.
- For new `src/` rewrite code, do not add `__all__`,
  `from __future__ import annotations`, or similar module-level boilerplate
  unless the user explicitly asks for it.
- Write comments only when they explain why a non-obvious choice exists or when
  they clarify tensor shapes around non-trivial reshape, permute, attention, or
  batching code. Do not add note-taking comments or comments that merely restate
  non-shape code. Delete such comments when touching nearby code, with no
  exception for large files or copied reference code in `src/`.
- Keep changes narrowly scoped. This is a large model repository with expensive
  inference and training paths; avoid incidental refactors across model,
  training, and eval boundaries.
- Follow the configured formatter stack: Black-style line length 88, `ufmt`,
  `usort`, and `ruff-api` as declared in `pyproject.toml`.
- Prefer existing utilities and config patterns over new abstractions:
  Hydra YAML configs under `sam3/train/configs/`, model builders in
  `sam3/model_builder.py`, and package-local helpers under the relevant
  `sam3/*` subpackage.
- Do not hardcode local absolute paths for datasets, checkpoints, logs, or
  outputs. Route them through config, CLI args, or explicit user-provided paths.
- Treat `sam3/model_builder.py`, predictor APIs, notebook examples, and README
  snippets as public-facing surfaces. If behavior or signatures change, update
  the relevant docs or examples in the same task.
- Be careful with GPU assumptions. Many code paths call `.cuda()` or use
  float16 tensors. For CPU-only checks, prefer focused tests or static review
  unless the code path is known to be CPU-safe.

## Verification Expectations

- For docs-only edits, read back the changed file and check that command paths,
  links, and directory names match the current workspace.
- For code edits, run the narrowest meaningful test first. Use
  `python -m pytest test/test_io_utils.py` for IO routing changes and expand to
  `python -m pytest test` when the affected surface is broader.
- For formatting-sensitive Python edits, run `ufmt format .` or explain why it
  was not run.
- For model-building, inference, training, or eval changes, state whether
  verification used local CPU-only tests, GPU execution, checkpoint download,
  config inspection, or static reasoning. Do not present static inspection as a
  successful full inference or training run.

## License And External Assets

- The repository contains `LICENSE` with the SAM License. Do not replace it,
  summarize it as MIT-only, or remove redistribution notices based solely on the
  `pyproject.toml` classifier.
- Do not commit or redistribute checkpoints, downloaded datasets, generated
  experiment logs, or large media artifacts unless the user explicitly requests
  packaging or publication and the license/data terms have been checked.
