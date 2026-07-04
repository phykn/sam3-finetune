# Video Memory Reference Design

## Goal

Add real SAM 3.1 tracker memory support to the `src/` rewrite so reference
images with masks, points, or boxes can condition later target-frame
predictions. The implementation must use the upstream multiplex tracker memory
path from `sam3-main` as the source of truth, not a handcrafted feature bank.

## Scope

- Keep `sam3-main/` unchanged and use it only as reference source.
- Do not use Hugging Face. Load `weight/sam3.1_multiplex.pt` or another
  explicit local checkpoint path only.
- Keep language/text/detector prompting out of the new `src` API.
- Support multiple reference frames for the same `obj_id`.
- Convert reference image plus mask into real tracker memory
  (`cond_frame_outputs`, `maskmem_features`, `maskmem_pos_enc`, `obj_ptr`).
- Allow point and box prompts by first producing or accepting a mask, then
  adding that mask into tracker memory.

## Architecture

The public API will expose a small image-list style memory predictor. Reference
examples are appended as pseudo-video frames, and target images are appended
after them. Each reference frame becomes a conditioning frame for an object id.
The target prediction uses upstream tracking propagation instead of manually
matching embeddings.

The implementation will port the upstream tracker-only stack into `src`:

- multiplex mask memory encoder
- multiplex controller/state helpers
- multiplex mask decoder
- multiplex video tracking state machine
- the decoupled memory attention classes used by the tracker

Existing `src` modules for ViT, neck, prompt encoder, SAM mask decoder,
position encoding, transforms, and common model utilities remain the local
building blocks. The new builder must map checkpoint keys from
`tracker.model.*` and `detector.backbone.vision_backbone.*` into the tracker
model, rather than using the existing image-only checkpoint filter.

## Windows And Optional Kernels

The upstream import path currently requires `triton` through
`sam3_tracker_utils.py`. This is not acceptable for the workspace-local Windows
`.venv`. The port will lazy-load or fall back around the EDT helper and keep
`fill_hole_area=0` by default so connected-components/Triton postprocessing is
not required for the first memory smoke test.

## Verification

Tests must prove these behaviors before implementation is considered complete:

- video memory modules import without `sam3-main` on `sys.path`
- tracker checkpoint filtering keeps memory/tracker keys that the image-only
  loader currently ignores
- a memory predictor can accept two or more reference frames for the same
  object id
- a smoke script can create a pseudo-video from `asset/sample.jpg`, add a
  reference mask, propagate to a target frame, and save a visual output
