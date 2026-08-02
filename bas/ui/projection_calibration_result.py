from __future__ import annotations

from typing import Tuple

import numpy as np

from ..schemas import OverlayCircle, OverlayLine, OverlayText, ProjectionOverlay


GOOD_ERROR_COLOR = (80, 255, 120)
WARNING_ERROR_COLOR = (0, 220, 255)
BAD_ERROR_COLOR = (60, 60, 255)
UNKNOWN_ERROR_COLOR = (220, 220, 220)


def build_projection_calibration_result_overlay(
    calibration,
    projector_size: Tuple[int, int],
) -> ProjectionOverlay:
    """Build an all-point, error-coded linked-calibration acceptance view."""

    size = (int(projector_size[0]), int(projector_size[1]))
    overlay = ProjectionOverlay(
        overlay_id="projector_calibration_result",
        frame_id=0,
        projector_size=size,
        suppress_star_formula=True,
    )
    projection = calibration.projection
    polygon = _points(getattr(projection, "table_polygon_proj", None))
    if polygon.shape[0] >= 3:
        closed = [(float(x), float(y)) for x, y in np.vstack([polygon, polygon[0]])]
        overlay.lines.append(OverlayLine(points=closed, color=(255, 255, 255), width=3, label="table"))

    targets = _points(getattr(projection, "proj_points", None))
    camera = _points(getattr(projection, "cam_points", None))
    if targets.shape[0] == 0 and polygon.shape[0] >= 3:
        targets = polygon
    errors = _base_reprojection_errors(projection, camera, targets)
    for index, point in enumerate(targets):
        error = float(errors[index]) if index < errors.size else float("nan")
        color = _error_color(error)
        x, y = float(point[0]), float(point[1])
        overlay.circles.append(OverlayCircle(center=(x, y), radius=6.0, color=color, width=2))
        overlay.labels.append(((x + 7.0, y - 6.0), f"{index + 1:03d}", color))

    report = dict(getattr(projection, "quality_report", {}) or {})
    stats = dict(projection.calibration_error_stats() or {})
    overlay.texts.append(
        OverlayText(
            position=(0.5 * size[0], 74.0),
            text=_acceptance_summary(report, stats, int(targets.shape[0])),
            color=(255, 255, 255),
            font_size_px=23.0,
            max_width_ratio=0.96,
            outline_width_px=2.0,
            background_alpha=175,
        )
    )
    return overlay


def _points(value) -> np.ndarray:
    if value is None:
        return np.zeros((0, 2), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape((-1, 2))
    return array[np.all(np.isfinite(array), axis=1)]


def _base_reprojection_errors(projection, camera: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if camera.shape != targets.shape or camera.shape[0] == 0:
        return np.full((targets.shape[0],), np.nan, dtype=np.float64)
    try:
        predicted = np.asarray(
            projection.camera_to_projector_points(camera, refined=False),
            dtype=np.float64,
        ).reshape((-1, 2))
    except (AttributeError, TypeError, ValueError):
        return np.full((targets.shape[0],), np.nan, dtype=np.float64)
    if predicted.shape != targets.shape:
        return np.full((targets.shape[0],), np.nan, dtype=np.float64)
    return np.linalg.norm(predicted - targets, axis=1)


def _error_color(error_px: float) -> tuple[int, int, int]:
    if not np.isfinite(error_px):
        return UNKNOWN_ERROR_COLOR
    if error_px <= 1.0:
        return GOOD_ERROR_COLOR
    if error_px <= 3.0:
        return WARNING_ERROR_COLOR
    return BAD_ERROR_COLOR


def _acceptance_summary(report: dict, stats: dict, point_count: int) -> str:
    passed = bool(report.get("quality_gate_passed", False))
    coverage = report.get("spatial_coverage", {})
    if not isinstance(coverage, dict):
        coverage = {}
    mean_px = float(report.get("ransac_inlier_mean_px", stats.get("mean_px", 0.0)))
    p95_px = float(
        report.get(
            "ransac_inlier_p95_px",
            stats.get("p95_px", stats.get("max_px", 0.0)),
        )
    )
    maximum_px = float(report.get("ransac_inlier_max_px", stats.get("max_px", 0.0)))
    return (
        f"{'PASS' if passed else 'FAIL'} | patterns "
        f"{int(report.get('patterns_used', 0))}/{int(report.get('patterns_total', 0))} | "
        f"{point_count} points | inlier mean {mean_px:.2f}px  "
        f"P95 {p95_px:.2f}px  max {maximum_px:.2f}px\n"
        f"coverage W{float(coverage.get('width_ratio', 0.0)):.0%} "
        f"H{float(coverage.get('height_ratio', 0.0)):.0%} "
        f"hull{float(coverage.get('hull_area_ratio', 0.0)):.0%} | "
        f"pockets {int(report.get('pocket_zones_used', 0))} | "
        f"CV P95 {float(report.get('pattern_cv_p95_px', 0.0)):.2f}px\n"
        "point error: green <=1px, yellow <=3px, red >3px"
    )
