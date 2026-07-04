from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auto_mask_generator import (
    Sam3AutomaticMaskGenerator,
    _crop_image,
    _image_size,
    build_point_grid,
    count_proposals_by_crop_grid,
    generate_crop_boxes,
)


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


class ProfilingAutomaticMaskGenerator(Sam3AutomaticMaskGenerator):
    def __init__(self, *args: Any, profiler: StageProfiler, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.profiler = profiler
        self._nms_scope = "crop"

    def generate(self, image: Image.Image):
        with self.profiler.stage("generate_total"):
            width, height = _image_size(image)
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
                            (crop_index, crop_box, _crop_image(image, crop_box))
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
        crop_encode_batch_size=args.crop_encode_batch_size,
        profiler=profiler,
    )

    predictor = generator.predictor
    wrap_timed(profiler, predictor, "set_image", "predictor_set_image")
    wrap_timed(profiler, predictor, "encode_image_batch", "encode_image_batch")
    wrap_timed(profiler, predictor, "encode_image_tensor_batch", "image_encoder")
    wrap_timed(profiler, predictor, "predict", "predict")
    wrap_timed(profiler, predictor, "predict_from_embedding", "predict_from_embedding")
    wrap_timed(profiler, predictor.model.prompt_encoder, "forward", "prompt_encoder")
    wrap_timed(profiler, predictor.model.mask_decoder, "forward", "mask_decoder")
    wrap_timed(profiler, predictor.transforms, "transform_coords", "transform_coords")
    wrap_timed(profiler, predictor.transforms, "postprocess_masks", "postprocess_masks")

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
    print(f"crop_encode_batch_size: {args.crop_encode_batch_size}")
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
