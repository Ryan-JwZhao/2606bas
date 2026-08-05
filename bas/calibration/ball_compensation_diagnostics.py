from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..geometry_contract import context_compatibility_errors
from .ball_compensation import BallCompensationModel
from .ball_compensation_sampling import (
    BallCompensationSample,
    ball_compensation_sample_from_dict,
    ball_compensation_sample_weights,
    build_ball_compensation_model,
    evaluate_ball_compensation_holdout,
)
from .geometry import ball_center_map_training_residuals


@dataclass(frozen=True)
class ReusableHoldoutSource:
    source_path: Path
    samples: tuple[BallCompensationSample, ...]
    repeatability: dict[str, Any]
    observation_count: int


@dataclass(frozen=True)
class ReusedHoldoutDiagnostic:
    model: BallCompensationModel
    training_report: dict[str, Any]
    holdout_report: dict[str, Any]


def load_reusable_holdout_source(
    path: str | Path | None,
    *,
    expected_sampling_grid_table_mm: Sequence[Sequence[float]],
    expected_calibration_context: Mapping[str, Any] | None = None,
) -> ReusableHoldoutSource | None:
    """Load the median samples representing an earlier 10x3 Holdout pass.

    A reused Holdout is diagnostic only.  Spatial grid and calibration context
    checks prevent accidentally applying observations from a different table,
    camera domain, projection, or detector model.
    """

    if path is None or not str(path).strip():
        return None
    source_path = Path(path)
    if not source_path.exists() or not source_path.is_file():
        return None
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    stored_grid = np.asarray(payload.get("sampling_grid_table_mm", []), dtype=np.float64).reshape((-1, 2))
    expected_grid = np.asarray(expected_sampling_grid_table_mm, dtype=np.float64).reshape((-1, 2))
    if stored_grid.shape != expected_grid.shape or not np.allclose(stored_grid, expected_grid, atol=0.5):
        return None
    if expected_calibration_context is not None:
        stored_context = dict(payload.get("calibration_context", {}))
        if context_compatibility_errors(stored_context, dict(expected_calibration_context)):
            return None

    raw_samples = payload.get("holdout_samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raw_samples = payload.get("diagnostic_reused_holdout_samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        return None
    samples = tuple(ball_compensation_sample_from_dict(item) for item in raw_samples)

    repeatability = _persisted_repeatability(payload)
    observation_count = int(repeatability.get("observation_count", len(samples)))
    if len(samples) < 8 or observation_count < len(samples):
        return None
    return ReusableHoldoutSource(
        source_path=source_path.resolve(),
        samples=samples,
        repeatability=repeatability,
        observation_count=observation_count,
    )


def fit_with_reused_holdout_diagnostic(
    training_samples: Sequence[BallCompensationSample],
    calibration,
    source: ReusableHoldoutSource,
    *,
    calibration_context: Mapping[str, Any] | None = None,
) -> ReusedHoldoutDiagnostic:
    """Fit new training data and score it against an earlier Holdout pass.

    The calibration service is restored before returning.  The caller decides
    whether to activate/save the resulting temporary model.
    """

    model = build_ball_compensation_model(
        training_samples,
        ball_diameter_mm=float(calibration.table.ball_diameter_mm),
        table_width_mm=float(calibration.table.width_mm),
        table_height_mm=float(calibration.table.height_mm),
    )
    model.calibration_context = dict(calibration_context or {})
    previous_model = calibration.ball_compensation_model
    try:
        calibration.ball_compensation_model = model
        calibration._rebuild_geometry()
        training_report = evaluate_ball_compensation_training_residuals(training_samples, calibration)
        holdout_report = evaluate_ball_compensation_holdout(
            source.samples,
            calibration,
            require_formal_geometry=True,
        )
    finally:
        calibration.ball_compensation_model = previous_model
        calibration._rebuild_geometry()

    holdout_report = {
        **holdout_report,
        "repeatability": dict(source.repeatability),
        "diagnostic_only": True,
        "formal_holdout": False,
        "source_file": str(source.source_path),
        "observation_count": int(source.observation_count),
        "diagnostic_reason": "Holdout observations predate the resampled training points.",
    }
    return ReusedHoldoutDiagnostic(
        model=model,
        training_report=training_report,
        holdout_report=holdout_report,
    )


def evaluate_ball_compensation_training_residuals(
    samples: Sequence[BallCompensationSample],
    calibration,
) -> dict[str, Any]:
    """Return all 56 in-sample runtime-map residual vectors and summary stats."""

    items = list(samples)
    camera = np.asarray([sample.detected_camera_px for sample in items], dtype=np.float32).reshape((-1, 2))
    target = np.asarray([sample.target_table_mm for sample in items], dtype=np.float64).reshape((-1, 2))
    predicted = calibration.ball_camera_px_to_table_mm(camera).astype(np.float64)
    vectors = predicted - target
    errors = np.linalg.norm(vectors, axis=1)
    # This scalar call is intentionally retained as a consistency guard: the
    # report must use the exact weighted runtime map family used by fitting.
    expected_errors = ball_center_map_training_residuals(
        camera,
        target,
        sample_weights=ball_compensation_sample_weights(items),
    )
    if not np.allclose(errors, expected_errors, atol=1e-4):
        raise RuntimeError("training residual evaluation does not match the runtime ball map")
    return {
        "sample_count": len(items),
        "error_mm": _stats(errors),
        "count_ge_1mm": int(np.sum(errors >= 1.0)),
        "count_ge_2mm": int(np.sum(errors >= 2.0)),
        "count_ge_5mm": int(np.sum(errors >= 5.0)),
        "samples": [
            {
                "sample_index": int(sample.sample_index),
                "target_table_mm": target[index].tolist(),
                "predicted_table_mm": predicted[index].tolist(),
                "error_vector_mm": vectors[index].tolist(),
                "error_mm": float(errors[index]),
            }
            for index, sample in enumerate(items)
        ],
    }


def _persisted_repeatability(payload: Mapping[str, Any]) -> dict[str, Any]:
    for key in (
        "holdout_repeatability",
        "diagnostic_holdout_repeatability",
        "diagnostic_reused_holdout_repeatability",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            return dict(value)
    for key in (
        "holdout_quality_report",
        "diagnostic_failed_holdout_quality_report",
        "diagnostic_reused_holdout_quality_report",
    ):
        report = payload.get(key)
        if isinstance(report, dict) and isinstance(report.get("repeatability"), dict):
            return dict(report["repeatability"])
    return {}


def _stats(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape((-1,))
    if data.size == 0:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": float(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
    }
