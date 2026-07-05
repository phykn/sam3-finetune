import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SAM3_MAIN = ROOT / "sam3-main"
if str(SAM3_MAIN) not in sys.path:
    sys.path.insert(0, str(SAM3_MAIN))

from sam3 import build_sam3_image_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache the SAM3 no-text grounding 'visual' language features."
    )
    parser.add_argument("--checkpoint", default="weight/sam3.1_multiplex.pt")
    parser.add_argument(
        "--bpe",
        default="sam3-main/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    )
    parser.add_argument("--prompt", default="visual")
    parser.add_argument("--output", default="weight/visual_language_sam31.pt")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = ROOT / args.checkpoint
    bpe = ROOT / args.bpe
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(
        bpe_path=str(bpe),
        path=str(checkpoint),
        load_from_HF=False,
        enable_inst_interactivity=False,
        device=device,
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device == "cuda" and dtype != torch.float32
        else torch.no_grad()
    )
    with autocast_context, torch.inference_mode():
        outputs = model.backbone.forward_text([args.prompt], device=device)

    cache = {
        "format": "sam3_visual_language_cache_v1",
        "prompt": args.prompt,
        "checkpoint_name": checkpoint.name,
        "language_features": outputs["language_features"].detach().cpu(),
        "language_mask": outputs["language_mask"].detach().cpu(),
        "language_embeds": outputs["language_embeds"].detach().cpu(),
    }
    torch.save(cache, output)
    print(f"device: {device}")
    print(f"prompt: {args.prompt}")
    print(f"dtype: {args.dtype}")
    print(f"language_features_shape: {tuple(cache['language_features'].shape)}")
    print(f"language_mask_shape: {tuple(cache['language_mask'].shape)}")
    print(f"language_embeds_shape: {tuple(cache['language_embeds'].shape)}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
