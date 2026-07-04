import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.image import Sam3Predictor
from src.transforms import save_mask_png, save_overlay_png


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    image_path = ROOT / "asset" / "sample.jpg"
    checkpoint_path = ROOT / "weight" / "sam3.1_multiplex.pt"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    box = np.array([132.0, 92.0, 264.0, 375.0], dtype=np.float32)
    point_coords = np.array([[195.0, 295.0]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int64)

    predictor = Sam3Predictor.from_checkpoint(checkpoint_path, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        predictor.set_image(image)
        masks, scores, low_res = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        refined_masks, refined_scores, refined_low_res = predictor.predict(
            mask_input=low_res[best_idx],
            multimask_output=False,
        )

    best_mask = refined_masks[0].astype(bool)
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
    print(f"refined_masks_shape: {refined_masks.shape}")
    print(f"refined_low_res_shape: {refined_low_res.shape}")
    print(f"refined_scores: {refined_scores.tolist()}")
    print(f"mask_path: {mask_path}")
    print(f"overlay_path: {overlay_path}")


if __name__ == "__main__":
    main()
