from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bas.app import RuntimePipeline
from bas.config import AppConfig
from bas.runtime_env import prepare_runtime_environment, preload_torch_for_backend
from bas.user_settings import UserSettings


@dataclass
class BenchmarkResult:
    frames_requested: int
    frames_processed: int
    fps: float
    planner_plan_avg_ms: float
    target_shot_avg_ms: float
    detect_calls_per_frame: float
    max_detect_calls_per_frame: int


def _active_config() -> AppConfig:
    return UserSettings.load().apply_to_config(AppConfig.load()).resolve_paths()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def benchmark_frames(frame_limit: int) -> BenchmarkResult:
    cfg = _active_config()
    prepare_runtime_environment()
    preload_torch_for_backend(cfg.detector.backend)
    pipeline = RuntimePipeline(cfg)
    planner = pipeline.planner

    plan_times_ms: list[float] = []
    target_times_ms: list[float] = []
    frame_detect_counts: list[int] = []
    current_frame_detects = 0

    original_plan = planner.plan
    original_target_update = planner.target_shot_mode.update
    original_detect = planner.aim_detector.detect

    def wrapped_detect(*args: Any, **kwargs: Any):
        nonlocal current_frame_detects
        current_frame_detects += 1
        return original_detect(*args, **kwargs)

    def wrapped_target_update(*args: Any, **kwargs: Any):
        start = time.perf_counter()
        try:
            return original_target_update(*args, **kwargs)
        finally:
            target_times_ms.append((time.perf_counter() - start) * 1000.0)

    def wrapped_plan(*args: Any, **kwargs: Any):
        nonlocal current_frame_detects
        current_frame_detects = 0
        start = time.perf_counter()
        try:
            return original_plan(*args, **kwargs)
        finally:
            plan_times_ms.append((time.perf_counter() - start) * 1000.0)
            frame_detect_counts.append(int(current_frame_detects))

    planner.aim_detector.detect = wrapped_detect
    planner.target_shot_mode.update = wrapped_target_update
    planner.plan = wrapped_plan

    frames_processed = 0
    start = time.perf_counter()
    try:
        while frames_processed < frame_limit:
            output = pipeline.step()
            if output is None:
                break
            frames_processed += 1
    finally:
        pipeline.close()

    elapsed_s = max(1.0e-9, time.perf_counter() - start)
    return BenchmarkResult(
        frames_requested=int(frame_limit),
        frames_processed=int(frames_processed),
        fps=float(frames_processed / elapsed_s),
        planner_plan_avg_ms=_mean(plan_times_ms),
        target_shot_avg_ms=_mean(target_times_ms),
        detect_calls_per_frame=_mean([float(count) for count in frame_detect_counts]),
        max_detect_calls_per_frame=max(frame_detect_counts) if frame_detect_counts else 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark shared cue-aim planner behavior.")
    parser.add_argument("--frames", type=int, nargs="+", default=[40, 90, 120], help="Frame counts to benchmark.")
    args = parser.parse_args()

    results = [benchmark_frames(max(1, int(frame_limit))) for frame_limit in args.frames]
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
