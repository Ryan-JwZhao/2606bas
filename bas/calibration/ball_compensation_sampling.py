from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from .ball_compensation import BallCompensationModel


@dataclass
class BallCompensationSample:
    sample_index: int
    target_table_mm: tuple[float, float]
    detected_camera_px: tuple[float, float]
    projected_target_px: tuple[float, float]
    expected_camera_px: tuple[float, float]
    observed_table_mm: tuple[float, float]
    delta_table_mm: tuple[float, float]
    detected_radius_px: float = 0.0
    detection_confidence: float = 0.0
    stability_spread_px: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_index": int(self.sample_index),
            "target_table_mm": [float(self.target_table_mm[0]), float(self.target_table_mm[1])],
            "detected_camera_px": [float(self.detected_camera_px[0]), float(self.detected_camera_px[1])],
            "projected_target_px": [float(self.projected_target_px[0]), float(self.projected_target_px[1])],
            "expected_camera_px": [float(self.expected_camera_px[0]), float(self.expected_camera_px[1])],
            "observed_table_mm": [float(self.observed_table_mm[0]), float(self.observed_table_mm[1])],
            "delta_table_mm": [float(self.delta_table_mm[0]), float(self.delta_table_mm[1])],
            "detected_radius_px": float(self.detected_radius_px),
            "detection_confidence": float(self.detection_confidence),
            "stability_spread_px": float(self.stability_spread_px),
        }


def build_engineered_ball_sampling_grid(
    table_width_mm: float,
    table_height_mm: float,
    ball_diameter_mm: float,
    cols: int = 5,
    rows: int = 4,
) -> np.ndarray:
    width = max(1.0, float(table_width_mm))
    height = max(1.0, float(table_height_mm))
    diameter = max(1.0, float(ball_diameter_mm))
    cols = max(2, int(cols))
    rows = max(2, int(rows))

    margin_x = min(width * 0.16, max(135.0, diameter * 2.4))
    margin_y = min(height * 0.16, max(110.0, diameter * 2.0))
    x0 = min(margin_x, width * 0.48)
    y0 = min(margin_y, height * 0.48)
    x1 = max(x0, width - x0)
    y1 = max(y0, height - y0)
    xs = np.linspace(x0, x1, cols, dtype=np.float64)
    ys = np.linspace(y0, y1, rows, dtype=np.float64)
    grid = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float64)
    return grid.reshape((-1, 2))


def build_ball_compensation_model(
    samples: Sequence[BallCompensationSample],
    ball_diameter_mm: float,
    max_neighbors: int = 8,
    mode: str = "engineered_ball_comp_v1",
) -> BallCompensationModel:
    if not samples:
        raise ValueError("At least one sampling point is required to build a ball compensation model.")
    controls = np.asarray([sample.detected_camera_px for sample in samples], dtype=np.float64).reshape((-1, 2))
    deltas = np.asarray([sample.delta_table_mm for sample in samples], dtype=np.float64).reshape((-1, 2))
    targets = np.asarray([sample.target_table_mm for sample in samples], dtype=np.float64).reshape((-1, 2))
    radii = np.asarray([float(sample.detected_radius_px) for sample in samples], dtype=np.float64)
    spreads = np.asarray([float(sample.stability_spread_px) for sample in samples], dtype=np.float64)
    confidences = np.asarray([float(sample.detection_confidence) for sample in samples], dtype=np.float64)
    delta_norms = np.linalg.norm(deltas, axis=1)

    quality_report = {
        "sample_count": int(len(samples)),
        "ball_diameter_mm": float(ball_diameter_mm),
        "camera_span_px": _span_report(controls),
        "table_span_mm": _span_report(targets),
        "delta_norm_mm": _stats_report(delta_norms),
        "stability_spread_px": _stats_report(spreads),
        "detected_radius_px": _stats_report(radii),
        "detection_confidence": _stats_report(confidences),
    }
    return BallCompensationModel(
        mode=str(mode or "engineered_ball_comp_v1"),
        control_points_camera_px=controls,
        delta_table_mm=deltas,
        max_neighbors=max(1, min(int(max_neighbors), len(samples))),
        quality_report=quality_report,
    )


def _stats_report(values: np.ndarray) -> Dict[str, float]:
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


def _span_report(points: np.ndarray) -> Dict[str, float]:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    if pts.size == 0:
        return {"width": 0.0, "height": 0.0}
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    return {
        "width": float(maxs[0] - mins[0]),
        "height": float(maxs[1] - mins[1]),
    }
