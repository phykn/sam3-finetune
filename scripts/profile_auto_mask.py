import argparse
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict.masks.generator import AutomaticMaskGenerator
from src.predict.masks.geometry import (
    build_point_grid,
    crop_image,
    generate_crop_boxes,
    image_size,
)
from src.predict.masks.proposals import count_proposals_by_crop_grid


@dataclass
class TimingStat:
    total: float = 0.0
    count: int = 0
    max_sec: float = 0.0

    def add(self, elapsed: float) -> None:
        self.total += elapsed
        self.count += 1
        self.max_sec = max(self.max_sec, elapsed)

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class StageProfiler:
    def __init__(self, sync_cuda: bool = True) -> None:
        self.sync_cuda = sync_cuda
        self.stats: dict[str, TimingStat] = defaultdict(TimingStat)

    @contextmanager
    def stage(self, name: str):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        started_at = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.stats[name].add(time.perf_counter() - started_at)

    def print_summary(self, total_elapsed: float) -> None:
        print("\nprofile_timings:")
        print("stage,count,total_sec,avg_ms,max_ms,pct_of_generate")
        for name, stat in sorted(
            self.stats.items(),
            key=lambda item: item[1].total,
            reverse=True,
        ):
            pct = (stat.total / total_elapsed * 100.0) if total_elapsed else 0.0
            print(
                f"{name},{stat.count},{stat.total:.4f},"
                f"{stat.average * 1000.0:.2f},{stat.max_sec * 1000.0:.2f},"
                f"{pct:.1f}"
            )


class ProfilingAutomaticMaskGenerator(AutomaticMaskGenerator):
    def __init__(self, *args: Any, profiler: StageProfiler, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.profiler = profiler
        self._nms_scope = "crop"

    def generate(self, image: Image.Image):
        with self.profiler.stage("generate_total"):
            width, height = image_size(image)
            point_grid_cache = {}
            proposals = []
            for crop_grid, points_per_side in self._crop_grid_config():
                with self.profiler.stage("point_grid_setup"):
                    normalized_grid = point_grid_cache.setdefault(
                        points_per_side,
                        build_point_grid(points_per_side),
                    )
                    crop_boxes = generate_crop_boxes(
                        width,
                        height,
                        crop_grid,
                        self.crop_overlap_ratio,
                    )

                crop_jobs = []
                for crop_index, crop_box in enumerate(crop_boxes):
                    with self.profiler.stage("crop_image"):
                        crop_jobs.append(
                            (crop_index, crop_box, crop_image(image, crop_box))
                        )

                proposals.extend(
                    self._generate_for_crop_jobs(
                        crop_jobs,
                        crop_grid,
                        normalized_grid,
                        (width, height),
                    )
                )

            self._nms_scope = "global"
            try:
                proposals = self._remove_duplicates(proposals)
            finally:
                self._nms_scope = "crop"

            with self.profiler.stage("final_sort_truncate"):
                proposals.sort(
                    key=lambda proposal: (
                        proposal.predicted_iou,
                        proposal.stability_score,
                        proposal.area,
                    ),
                    reverse=True,
                )
                if self.max_masks is not None:
                    proposals = proposals[: self.max_masks]
            return proposals

    def _generate_for_crop_jobs(self, *args: Any, **kwargs: Any):
        crop_grid = args[1] if len(args) > 1 else kwargs.get("crop_grid", "unknown")
        with self.profiler.stage("crop_jobs_total"):
            with self.profiler.stage(f"crop_jobs_grid_{crop_grid}"):
                return super()._generate_for_crop_jobs(*args, **kwargs)

    def _generate_for_crop_jobs_single(self, *args: Any, **kwargs: Any):
        crop_grid = args[1] if len(args) > 1 else kwargs.get("crop_grid", "unknown")
        with self.profiler.stage("crop_jobs_single_total"):
            with self.profiler.stage(f"crop_jobs_single_grid_{crop_grid}"):
                return super()._generate_for_crop_jobs_single(*args, **kwargs)

    def _generate_for_crop(self, *args: Any, **kwargs: Any):
        crop_grid = args[2] if len(args) > 2 else kwargs.get("crop_grid", "unknown")
        with self.profiler.stage("crop_total"):
            with self.profiler.stage(f"crop_total_grid_{crop_grid}"):
                return super()._generate_for_crop(*args, **kwargs)

    def _generate_for_crop_embedding(self, *args: Any, **kwargs: Any):
        crop_grid = args[3] if len(args) > 3 else kwargs.get("crop_grid", "unknown")
        with self.profiler.stage("crop_total"):
            with self.profiler.stage(f"crop_total_grid_{crop_grid}"):
                return super()._generate_for_crop_embedding(*args, **kwargs)

    def _proposals_from_batch(self, *args: Any, **kwargs: Any):
        with self.profiler.stage("proposal_filtering"):
            return super()._proposals_from_batch(*args, **kwargs)

    def _remove_duplicates(self, *args: Any, **kwargs: Any):
        name = "global_nms" if self._nms_scope == "global" else "crop_nms"
        with self.profiler.stage(name):
            return super()._remove_duplicates(*args, **kwargs)


def wrap_timed(
    profiler: StageProfiler,
    obj: Any,
    method_name: str,
    stage_name: str,
) -> None:
    original = getattr(obj, method_name)

    def timed_method(*args: Any, **kwargs: Any):
        with profiler.stage(stage_name):
            return original(*args, **kwargs)

    setattr(obj, method_name, timed_method)


def install_timed_predict_masks(profiler: StageProfiler, mask_decoder: Any) -> None:
    def timed_predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: list[torch.Tensor] | None = None,
    ):
        with profiler.stage("mask_decoder.predict_masks"):
            with profiler.stage("mask_decoder.tokens"):
                s = 0
                if self.pred_obj_scores:
                    output_tokens = torch.cat(
                        [
                            self.obj_score_token.weight,
                            self.iou_token.weight,
                            self.mask_tokens.weight,
                        ],
                        dim=0,
                    )
                    s = 1
                else:
                    output_tokens = torch.cat(
                        [self.iou_token.weight, self.mask_tokens.weight], dim=0
                    )
                output_tokens = output_tokens.unsqueeze(0).expand(
                    sparse_prompt_embeddings.size(0), -1, -1
                )
                tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

            with profiler.stage("mask_decoder.repeat_image"):
                if repeat_image:
                    src = torch.repeat_interleave(
                        image_embeddings,
                        tokens.shape[0],
                        dim=0,
                    )
                else:
                    assert image_embeddings.shape[0] == tokens.shape[0]
                    src = image_embeddings
                src = src + dense_prompt_embeddings
                assert (
                    image_pe.size(0) == 1
                ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
                pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
                b, c, h, w = src.shape

            with profiler.stage("mask_decoder.transformer"):
                hs, src = self.transformer(src, pos_src, tokens)
            iou_token_out = hs[:, s, :]
            mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

            with profiler.stage("mask_decoder.reshape_src"):
                src = src.transpose(1, 2).view(b, c, h, w)

            with profiler.stage("mask_decoder.upscale_total"):
                if not self.use_high_res_features:
                    upscaled_embedding = self.output_upscaling(src)
                else:
                    dc1, ln1, act1, dc2, act2 = self.output_upscaling
                    feat_s0, feat_s1 = high_res_features
                    upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
                    upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

            with profiler.stage("mask_decoder.hyper_stack"):
                hyper_in_list = []
                for i in range(self.num_mask_tokens):
                    hyper_in_list.append(
                        self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                    )
                hyper_in = torch.stack(hyper_in_list, dim=1)

            with profiler.stage("mask_decoder.mask_matmul"):
                b, c, h, w = upscaled_embedding.shape
                masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
                    b,
                    -1,
                    h,
                    w,
                )

            with profiler.stage("mask_decoder.quality_heads"):
                iou_pred = self.iou_prediction_head(iou_token_out)
                if self.pred_obj_scores:
                    assert s == 1
                    object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
                else:
                    object_score_logits = 10.0 * iou_pred.new_ones(
                        iou_pred.shape[0],
                        1,
                    )

            return masks, iou_pred, mask_tokens_out, object_score_logits

    mask_decoder.predict_masks = MethodType(timed_predict_masks, mask_decoder)


def instrument_mask_decoder_internals(
    profiler: StageProfiler, mask_decoder: Any
) -> None:
    install_timed_predict_masks(profiler, mask_decoder)

    for layer in mask_decoder.transformer.layers:
        wrap_timed(profiler, layer, "forward", "mask_decoder.transformer.layer")
        wrap_timed(
            profiler,
            layer.self_attn,
            "forward",
            "mask_decoder.attn.self",
        )
        wrap_timed(
            profiler,
            layer.cross_attn_token_to_image,
            "forward",
            "mask_decoder.attn.token_to_image",
        )
        wrap_timed(profiler, layer.mlp, "forward", "mask_decoder.transformer.mlp")
        wrap_timed(
            profiler,
            layer.cross_attn_image_to_token,
            "forward",
            "mask_decoder.attn.image_to_token",
        )

    wrap_timed(
        profiler,
        mask_decoder.transformer.final_attn_token_to_image,
        "forward",
        "mask_decoder.attn.final_token_to_image",
    )

    output_upscaling = mask_decoder.output_upscaling
    wrap_timed(profiler, output_upscaling[0], "forward", "mask_decoder.upscale.dc1")
    wrap_timed(profiler, output_upscaling[1], "forward", "mask_decoder.upscale.ln1")
    wrap_timed(profiler, output_upscaling[3], "forward", "mask_decoder.upscale.dc2")
    for hyper_mlp in mask_decoder.output_hypernetworks_mlps:
        wrap_timed(profiler, hyper_mlp, "forward", "mask_decoder.hyper_mlp")
    wrap_timed(
        profiler,
        mask_decoder.iou_prediction_head,
        "forward",
        "mask_decoder.iou_head",
    )
    if getattr(mask_decoder, "pred_obj_scores", False):
        wrap_timed(
            profiler,
            mask_decoder.pred_obj_score_head,
            "forward",
            "mask_decoder.obj_score_head",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-grids", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--crop-points-per-side", nargs="*", type=int, default=[32, 32])
    parser.add_argument("--crop-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--max-masks", type=int, default=200)
    parser.add_argument("--max-masks-per-crop", type=int, default=None)
    parser.add_argument("--keep-crop-edge-masks", action="store_true")
    parser.add_argument("--crop-encode-batch-size", type=int, default=2)
    parser.add_argument("--prompt-decode-batch-size", type=int, default=1)
    parser.add_argument("--image-batch-size", type=int, default=None)
    parser.add_argument("--prompt-batch-size", type=int, default=None)
    parser.add_argument("--allow-cross-crop-prompt-decode", action="store_true")
    parser.add_argument("--skip-mask-decoder-internals", action="store_true")
    parser.add_argument("--no-sync-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling.")

    image_path = ROOT / "asset" / "sample.jpg"
    checkpoint_path = ROOT / "weight" / "sam3.1_multiplex.pt"

    profiler = StageProfiler(sync_cuda=not args.no_sync_cuda)
    image = Image.open(image_path).convert("RGB")
    image_batch_size = (
        args.crop_encode_batch_size
        if args.image_batch_size is None
        else args.image_batch_size
    )
    prompt_batch_size = (
        args.prompt_decode_batch_size
        if args.prompt_batch_size is None
        else args.prompt_batch_size
    )
    generator = ProfilingAutomaticMaskGenerator.from_checkpoint(
        checkpoint_path,
        device="cuda",
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.75,
        box_nms_thresh=0.7,
        max_masks=args.max_masks,
        crop_grids=args.crop_grids,
        crop_points_per_side=args.crop_points_per_side,
        crop_overlap_ratio=args.crop_overlap_ratio,
        max_masks_per_crop=args.max_masks_per_crop,
        filter_crop_edge_masks=not args.keep_crop_edge_masks,
        image_batch_size=image_batch_size,
        prompt_batch_size=prompt_batch_size,
        allow_cross_crop_prompt_decode=args.allow_cross_crop_prompt_decode,
        profiler=profiler,
    )

    predictor = generator.predictor
    wrap_timed(profiler, predictor, "set_image", "predictor_set_image")
    wrap_timed(profiler, predictor, "encode_image_batch", "encode_image_batch")
    wrap_timed(profiler, predictor, "encode_image_tensor_batch", "image_encoder")
    wrap_timed(profiler, predictor, "predict", "predict")
    wrap_timed(profiler, predictor, "predict_from_embedding", "predict_from_embedding")
    wrap_timed(profiler, predictor, "_format_outputs", "format_outputs_cpu_copy")
    wrap_timed(profiler, predictor.model.prompt_encoder, "forward", "prompt_encoder")
    wrap_timed(profiler, predictor.model.mask_decoder, "forward", "mask_decoder")
    wrap_timed(profiler, predictor.transforms, "transform_coords", "transform_coords")
    wrap_timed(profiler, predictor.transforms, "postprocess_masks", "postprocess_masks")
    if not args.skip_mask_decoder_internals:
        instrument_mask_decoder_internals(profiler, predictor.model.mask_decoder)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started_at = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        proposals = generator.generate(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at

    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"crop_grids: {args.crop_grids}")
    print(f"crop_points_per_side: {args.crop_points_per_side}")
    print(f"points_per_batch: {args.points_per_batch}")
    print(f"image_batch_size: {image_batch_size}")
    print(f"prompt_batch_size: {prompt_batch_size}")
    print(f"allow_cross_crop_prompt_decode: {args.allow_cross_crop_prompt_decode}")
    print(f"mask_decoder_internals: {not args.skip_mask_decoder_internals}")
    mask_decoder = predictor.model.mask_decoder
    print(f"mask_decoder_transformer_depth: {mask_decoder.transformer.depth}")
    print(f"mask_decoder_num_mask_tokens: {mask_decoder.num_mask_tokens}")
    print(f"mask_decoder_use_high_res_features: {mask_decoder.use_high_res_features}")
    print(f"mask_decoder_pred_obj_scores: {mask_decoder.pred_obj_scores}")
    print(f"sync_cuda: {not args.no_sync_cuda}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"proposal_count: {len(proposals)}")
    print(f"proposal_count_by_crop_grid: {count_proposals_by_crop_grid(proposals)}")
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        print(f"cuda_peak_allocated_mb: {peak_mb:.1f}")
    profiler.print_summary(elapsed)


if __name__ == "__main__":
    main()
