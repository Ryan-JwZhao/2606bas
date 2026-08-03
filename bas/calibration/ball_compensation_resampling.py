from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .ball_compensation_sampling import BallCompensationSample


@dataclass(frozen=True)
class LocationRepeatability:
    location_index: int
    repeat_count: int
    median_observed_table_mm: tuple[float, float]
    maximum_deviation_mm: float
    passed: bool


@dataclass(frozen=True)
class HoldoutRepeatabilityAssessment:
    locations: tuple[LocationRepeatability, ...]
    unstable_location_indices: tuple[int, ...]
    maximum_deviation_mm: float

    @property
    def passed(self) -> bool:
        return not self.unstable_location_indices


@dataclass(frozen=True)
class FailedHoldoutRecoveryPlan:
    failed_holdout_location_indices: tuple[int, ...]
    failed_targets_table_mm: tuple[tuple[float, float], ...]
    training_sample_indices: tuple[int, ...]


def aggregate_training_repeats(
    samples: Sequence[BallCompensationSample],
    *,
    minimum_repeats: int = 3,
) -> BallCompensationSample:
    """Median-combine repeated stable measurements for one training target."""

    items = list(samples)
    required = max(1, int(minimum_repeats))
    if len(items) < required:
        raise ValueError(f"training target has {len(items)} repeats; at least {required} are required")
    sample_indices = {int(sample.sample_index) for sample in items}
    if len(sample_indices) != 1:
        raise ValueError("training repeats mix different sample indices")
    targets = np.asarray([sample.target_table_mm for sample in items], dtype=np.float64)
    target = np.median(targets, axis=0)
    if float(np.max(np.linalg.norm(targets - target, axis=1))) > 0.5:
        raise ValueError("training repeats mix different target positions")

    observed = np.asarray([sample.observed_table_mm for sample in items], dtype=np.float64)
    median_observed = np.median(observed, axis=0)
    methods = [str(sample.geometry_method) for sample in items]
    versions = [str(sample.detector_version) for sample in items]
    return BallCompensationSample(
        sample_index=int(items[0].sample_index),
        target_table_mm=(float(target[0]), float(target[1])),
        detected_camera_px=_median_point(items, "detected_camera_px"),
        projected_target_px=_median_point(items, "projected_target_px"),
        expected_camera_px=_median_point(items, "expected_camera_px"),
        observed_table_mm=(float(median_observed[0]), float(median_observed[1])),
        delta_table_mm=(float(target[0] - median_observed[0]), float(target[1] - median_observed[1])),
        detected_radius_px=float(np.median([sample.detected_radius_px for sample in items])),
        detection_confidence=float(np.median([sample.detection_confidence for sample in items])),
        stability_spread_px=float(np.median([sample.stability_spread_px for sample in items])),
        geometry_quality=float(np.median([sample.geometry_quality for sample in items])),
        geometry_method=max(dict.fromkeys(methods), key=methods.count),
        detector_version=max(dict.fromkeys(versions), key=versions.count),
    )


def assess_holdout_repeatability(
    repeated_samples: Sequence[Sequence[BallCompensationSample]],
    *,
    maximum_deviation_mm: float,
    minimum_repeats: int = 3,
) -> HoldoutRepeatabilityAssessment:
    """Find locations whose independent placements disagree, ignoring target offset."""

    limit = max(0.0, float(maximum_deviation_mm))
    required = max(1, int(minimum_repeats))
    locations: list[LocationRepeatability] = []
    unstable: list[int] = []
    for location_index, group in enumerate(repeated_samples):
        items = list(group)
        if len(items) < required:
            deviation = float("inf")
            median = np.asarray([float("nan"), float("nan")], dtype=np.float64)
        else:
            observed = np.asarray([sample.observed_table_mm for sample in items], dtype=np.float64)
            median = np.median(observed, axis=0)
            deviation = float(np.max(np.linalg.norm(observed - median, axis=1)))
        passed = bool(np.isfinite(deviation) and deviation <= limit)
        if not passed:
            unstable.append(location_index)
        locations.append(
            LocationRepeatability(
                location_index=location_index,
                repeat_count=len(items),
                median_observed_table_mm=(float(median[0]), float(median[1])),
                maximum_deviation_mm=deviation,
                passed=passed,
            )
        )
    return HoldoutRepeatabilityAssessment(
        locations=tuple(locations),
        unstable_location_indices=tuple(unstable),
        maximum_deviation_mm=limit,
    )


def plan_failed_holdout_recovery(
    training_samples: Sequence[BallCompensationSample],
    holdout_report: Mapping[str, Any],
    *,
    failure_error_mm: float = 2.0,
    surrounding_neighbor_count: int = 8,
) -> FailedHoldoutRecoveryPlan:
    """Map failed Holdout targets to existing training samples plus one nearby ring."""

    training = list(training_samples)
    report_samples = list(holdout_report.get("samples", []))
    failed_locations: list[int] = []
    failed_targets: list[tuple[float, float]] = []
    for location_index, item in enumerate(report_samples):
        error_mm = float(dict(item).get("error_mm", float("inf")))
        if not np.isfinite(error_mm) or error_mm >= float(failure_error_mm):
            target = np.asarray(dict(item).get("target_table_mm", []), dtype=np.float64).reshape((-1,))
            if target.shape[0] < 2:
                continue
            failed_locations.append(location_index)
            failed_targets.append((float(target[0]), float(target[1])))

    if not training or not failed_targets:
        return FailedHoldoutRecoveryPlan(tuple(failed_locations), tuple(failed_targets), ())

    points = np.asarray([sample.target_table_mm for sample in training], dtype=np.float64).reshape((-1, 2))
    scale = np.maximum(np.ptp(points, axis=0), 1.0)
    normalized = points / scale
    select_count = min(len(training), max(1, int(surrounding_neighbor_count) + 1))
    selected: set[int] = set()
    for target in failed_targets:
        target_normalized = np.asarray(target, dtype=np.float64) / scale
        distances = np.linalg.norm(normalized - target_normalized, axis=1)
        nearest = np.argsort(distances, kind="stable")[:select_count]
        selected.update(int(training[int(index)].sample_index) for index in nearest)
    return FailedHoldoutRecoveryPlan(
        failed_holdout_location_indices=tuple(failed_locations),
        failed_targets_table_mm=tuple(failed_targets),
        training_sample_indices=tuple(sorted(selected)),
    )


def _median_point(samples: Sequence[BallCompensationSample], field: str) -> tuple[float, float]:
    values = np.asarray([getattr(sample, field) for sample in samples], dtype=np.float64).reshape((-1, 2))
    median = np.median(values, axis=0)
    return float(median[0]), float(median[1])
