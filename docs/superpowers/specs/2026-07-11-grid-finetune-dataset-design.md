# Grid Finetune Dataset Design

## Goal

Generate more, cleaner finetuning masks with `GridPredictor` while preserving the existing dataset and its class meanings:

- `0`: background
- `1`: frog
- `2`: leaf

## Output

Update `finetune_dataset/` in place with the same split, class-folder, JSON schema, and embedded image/ROI format. Keep only this dataset version.

Write one preview image per source image under `finetune_dataset/preview/`. The preview shows candidate IDs, class colors, masks, and scores.

## Generation

Run the original `sam3.1_multiplex.pt` through `GridPredictor` with the existing grid defaults:

- tiles: `(1, 2)`
- points per side: `(10, 10)`
- overlap: `0.25`
- stability threshold: `0.75`
- NMS threshold: `0.7`
- minimum area: `64`

`GridPredictor` already refines each surviving candidate once with its low-resolution logit. Do not add a second refinement pass.

## Class Assignment

Use the existing manually defined class boxes as basic SAM box prompts to build class reference masks. Do not use LoRA or a trained classifier.

For every grid candidate:

1. Compute the fraction of candidate mask pixels inside each class reference mask.
2. Assign the class with the largest fraction when the candidate point is inside that reference mask and at least half of the candidate mask overlaps it.
3. Discard candidates that do not satisfy both conditions.
4. Keep GridPredictor's stability, area, edge, and NMS filtering.
5. For frog, keep only the largest candidate covering at least 25% of the manual frog region; use the original box mask when no candidate qualifies.

Class `0` remains manually confirmed background data. Preserve its existing objects because their masks are excluded from mask loss by `mask_valid=0`; they are still needed to create loader samples and prompts.

## Safety

- Fail before writing if an input image or original class JSON is missing.
- Update only known class JSON and preview files under `finetune_dataset/`.
- Do not modify `asset/` or `weight/`.
- Store the source grid point, score, stability, and class-region overlap in object metadata/metrics.

## Verification

- Unit-test class assignment at region boundaries, overlap threshold, and ambiguous regions.
- Verify candidate JSON loads through the existing `Sample` loader.
- Verify folder classes and schema with dataset tests.
- Run the generator on all configured train/valid images.
- Review candidate counts and previews against the current dataset before any replacement.
