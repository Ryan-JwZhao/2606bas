"""Shared acceptance thresholds for calibration quality gates.

Keep automatic activation and the standalone verification report on the same
physical accuracy scale so that a model cannot be activated with errors that
the formal verifier would reject.
"""

from __future__ import annotations

import math

IMAGE_ERROR_MEAN_PX = 0.20
IMAGE_ERROR_P95_PX = 0.35

MVP_TABLE_ERROR_MEDIAN_MM = 1.5
MVP_TABLE_ERROR_P95_MM = 3.0
FORMAL_TABLE_ERROR_MEDIAN_MM = 1.0
FORMAL_TABLE_ERROR_P95_MM = 2.0

BALL_COMPENSATION_MIN_TRAINING_SAMPLES = 20
BALL_COMPENSATION_MIN_HOLDOUT_SAMPLES = 8
BALL_COMPENSATION_MIN_ACCEPTED_SAMPLES = (
    BALL_COMPENSATION_MIN_TRAINING_SAMPLES + BALL_COMPENSATION_MIN_HOLDOUT_SAMPLES
)

POCKET_ZONE_ERROR_P95_MM = 2.5
DISTANCE_ERROR_SLOPE_ABS_MM_PER_CM = 0.03


def ball_holdout_quality_errors(
    *,
    sample_count: int,
    median_mm: float,
    p95_mm: float,
    mean_bias_mm: float,
) -> list[str]:
    """Return reasons why independent ball-center validation is not formal-grade."""

    errors: list[str] = []
    if int(sample_count) < BALL_COMPENSATION_MIN_HOLDOUT_SAMPLES:
        errors.append(
            f"holdout has {int(sample_count)} samples; "
            f"at least {BALL_COMPENSATION_MIN_HOLDOUT_SAMPLES} are required"
        )
    if not math.isfinite(float(median_mm)) or float(median_mm) >= FORMAL_TABLE_ERROR_MEDIAN_MM:
        errors.append(
            f"holdout median {float(median_mm):.2f} mm is not below "
            f"{FORMAL_TABLE_ERROR_MEDIAN_MM:.2f} mm"
        )
    if not math.isfinite(float(p95_mm)) or float(p95_mm) >= FORMAL_TABLE_ERROR_P95_MM:
        errors.append(
            f"holdout P95 {float(p95_mm):.2f} mm is not below "
            f"{FORMAL_TABLE_ERROR_P95_MM:.2f} mm"
        )
    if not math.isfinite(float(mean_bias_mm)) or float(mean_bias_mm) >= FORMAL_TABLE_ERROR_MEDIAN_MM:
        errors.append(
            f"holdout mean bias {float(mean_bias_mm):.2f} mm is not below "
            f"{FORMAL_TABLE_ERROR_MEDIAN_MM:.2f} mm"
        )
    return errors
