import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.model import Sam3GroundingModel  # noqa: E402
from src.predict.ground import GroundPredictor  # noqa: E402

WEIGHT = ROOT / "weight" / "sam3.1_multiplex.pt"
VISUAL = ROOT / "weight" / "visual_token.pt"
REF = ROOT / "asset" / "frog_ref.jpg"
TARGET = ROOT / "asset" / "frog_tgt.jpg"
BOXES = np.array(
    [
        [350, 500, 580, 720],
        [0, 1010, 75, 1130],
        [230, 625, 320, 720],
        [770, 1090, 870, 1190],
    ],
    dtype=np.float32,
)
CLASS_IDS = np.arange(len(BOXES), dtype=np.int64)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the grounding benchmark")

    device = torch.device("cuda")
    reference_image = Image.open(REF).convert("RGB")
    target_image = Image.open(TARGET).convert("RGB")
    model = Sam3GroundingModel(path=WEIGHT, visual_path=VISUAL).to(device).eval()
    predictor = make_predictor(model, device)
    reference = predictor.encode_reference(reference_image, BOXES, CLASS_IDS)

    predictor.predict(target_image, [reference])
    first, first_ms, first_mb = measure(predictor, target_image, reference)
    second, second_ms, second_mb = measure(predictor, target_image, reference)
    compare(first, second)

    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"objects: {len(first)}")
    print(f"run_1_ms: {first_ms:.3f}")
    print(f"run_2_ms: {second_ms:.3f}")
    print(f"run_1_peak_mb: {first_mb:.3f}")
    print(f"run_2_peak_mb: {second_mb:.3f}")


def make_predictor(model, device):
    return GroundPredictor(
        model,
        device=device,
        score_thr=0.5,
        sim_thr=0.0,
        top_k=5,
    )


def measure(predictor, image, reference):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    start = perf_counter()
    out = predictor.predict(image, [reference])
    torch.cuda.synchronize()
    elapsed = (perf_counter() - start) * 1000
    peak = (torch.cuda.max_memory_allocated() - before) / (1024**2)
    return out, elapsed, peak


def compare(first, second):
    if len(first) != len(second):
        raise AssertionError("repeated object counts differ")
    for left, right in zip(first, second):
        if left["class_id"] != right["class_id"] or left["box"] != right["box"]:
            raise AssertionError("repeated object labels differ")
        if not np.array_equal(left["mask"], right["mask"]):
            raise AssertionError("repeated masks differ")
        np.testing.assert_allclose(left["logit"], right["logit"], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            list(left["metrics"].values()),
            list(right["metrics"].values()),
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    main()
