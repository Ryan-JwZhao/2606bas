from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

from .ball_compensation import BallCompensationModel
from ..table_boundaries import EdgeInsets, derive_table_boundaries, inset_polygon_uniform


MAX_BALL_MAP_CV_P95_BALL_DIAMETER_RATIO = 0.18
MIN_BALL_MAP_CV_P95_MM = 8.0
MIN_BALL_TARGET_WIDTH_RATIO = 0.60
MIN_BALL_TARGET_HEIGHT_RATIO = 0.55
MIN_BALL_TARGET_HULL_AREA_RATIO = 0.35
MAX_BALL_HOLDOUT_P95_BALL_DIAMETER_RATIO = 0.12
MIN_BALL_HOLDOUT_P95_MM = 5.0
MAX_BALL_HOLDOUT_BIAS_BALL_DIAMETER_RATIO = 0.05
MIN_BALL_HOLDOUT_BIAS_MM = 2.0


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
    geometry_quality: float = 0.0
    geometry_method: str = "unknown"
    detector_version: str = "unknown"

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
            "geometry_quality": float(self.geometry_quality),
            "geometry_method": str(self.geometry_method),
            "detector_version": str(self.detector_version),
        }


@dataclass(frozen=True)
class ValidatedBallCompensation:
    model: BallCompensationModel
    training_samples: tuple[BallCompensationSample, ...]
    holdout_samples: tuple[BallCompensationSample, ...]
    holdout_report: Dict[str, object]


class BallCompensationValidationError(ValueError):
    def __init__(
        self,
        report: Dict[str, object],
        *,
        training_count: int,
        holdout_count: int,
    ) -> None:
        self.report = dict(report)
        self.training_count = int(training_count)
        self.holdout_count = int(holdout_count)
        reasons = "; ".join(str(item) for item in report.get("quality_gate_errors", []))
        super().__init__(f"ball compensation holdout validation failed: {reasons}")


def build_engineered_ball_sampling_grid(
    table_width_mm: float,
    table_height_mm: float,
    ball_diameter_mm: float,
    cols: int = 5,
    rows: int = 4,
    preferred_polygon_mm: np.ndarray | None = None,
    extra_safe_inset_mm: float | None = None,
    priority_points_mm: np.ndarray | None = None,
) -> np.ndarray:
    width = max(1.0, float(table_width_mm))
    height = max(1.0, float(table_height_mm))
    diameter = max(1.0, float(ball_diameter_mm))
    cols = max(2, int(cols))
    rows = max(2, int(rows))
    target_count = cols * rows

    polygon = np.asarray(preferred_polygon_mm, dtype=np.float32).reshape((-1, 2)) if preferred_polygon_mm is not None else np.zeros((0, 2), dtype=np.float32)
    if polygon.shape[0] >= 3:
        safe_inset = max(0.0, float(extra_safe_inset_mm if extra_safe_inset_mm is not None else 0.5 * diameter))
        safe_polygon = inset_polygon_uniform(polygon, safe_inset, width, height) if safe_inset > 1e-6 else polygon
        polygon_grid = _sampling_grid_inside_polygon(safe_polygon, target_count=target_count, cols=cols, rows=rows)
        if polygon_grid.shape[0] >= 4:
            edge_target_count = min(max(12, int(round(target_count * 0.55))), max(0, target_count - 4))
            edge_points = _polygon_edge_priority_points(
                safe_polygon,
                target_count=edge_target_count,
                priority_points_mm=priority_points_mm,
            )
            if edge_points.shape[0] > 0:
                merged = _merge_sampling_sets(
                    edge_points,
                    polygon_grid,
                    target_count=target_count,
                    min_spacing=_estimated_sampling_spacing(safe_polygon, cols=cols, rows=rows),
                )
                if merged.shape[0] >= 4:
                    return _efficient_sampling_order(merged)
            return _efficient_sampling_order(polygon_grid)

    margin_x = min(width * 0.16, max(135.0, diameter * 2.4))
    margin_y = min(height * 0.16, max(110.0, diameter * 2.0))
    x0 = min(margin_x, width * 0.48)
    y0 = min(margin_y, height * 0.48)
    x1 = max(x0, width - x0)
    y1 = max(y0, height - y0)
    xs = np.linspace(x0, x1, cols, dtype=np.float64)
    ys = np.linspace(y0, y1, rows, dtype=np.float64)
    grid = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float64)
    return _efficient_sampling_order(grid)


def _efficient_sampling_order(points: np.ndarray) -> np.ndarray:
    """Order spread points by short moves without changing their coverage."""

    pts = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    if pts.shape[0] <= 2:
        return pts.copy()
    remaining = set(range(pts.shape[0]))
    current = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
    order = [current]
    remaining.remove(current)
    while remaining:
        candidates = np.asarray(sorted(remaining), dtype=np.int32)
        distances = np.linalg.norm(pts[candidates] - pts[current], axis=1)
        current = int(candidates[int(np.argmin(distances))])
        order.append(current)
        remaining.remove(current)
    return pts[np.asarray(order, dtype=np.int32)].reshape((-1, 2))


def update_calibration_table_boundaries_from_geometry_frame(
    calibration,
    geometry,
    frame_shape,
    *,
    projection_visible_insets: EdgeInsets,
    physical_rail_insets: EdgeInsets,
    physical_middle_pocket_relief_top_mm: float,
    physical_middle_pocket_relief_bottom_mm: float,
    center_reachable_extra_margin_mm: float,
) -> bool:
    if geometry is None or getattr(geometry, "is_empty", True):
        return False
    if frame_shape is None or len(frame_shape) < 2:
        return False
    height = int(frame_shape[0])
    width = int(frame_shape[1])
    if width <= 0 or height <= 0:
        return False
    _, inner_px, pockets_px = geometry.scaled(width, height)
    inner_px = np.asarray(inner_px, dtype=np.float32).reshape((-1, 2))
    if inner_px.shape[0] < 3:
        return False
    visible_mm = calibration.camera_px_to_table_mm(inner_px)
    pocket_curves_mm = [
        calibration.camera_px_to_table_mm(np.asarray(pocket, dtype=np.float32).reshape((-1, 2)))
        for pocket in pockets_px
        if np.asarray(pocket, dtype=np.float32).reshape((-1, 2)).shape[0] >= 2
    ]
    boundaries = derive_table_boundaries(
        visible_mm,
        pocket_curves_mm,
        table_width_mm=float(calibration.table.width_mm),
        table_height_mm=float(calibration.table.height_mm),
        ball_diameter_mm=float(calibration.table.ball_diameter_mm),
        projection_visible_insets=projection_visible_insets,
        physical_rail_insets=physical_rail_insets,
        physical_middle_pocket_relief_top_mm=float(physical_middle_pocket_relief_top_mm),
        physical_middle_pocket_relief_bottom_mm=float(physical_middle_pocket_relief_bottom_mm),
        center_reachable_extra_margin_mm=float(center_reachable_extra_margin_mm),
    )
    calibration.table.projection_visible_polygon_mm = _points_to_tuples(boundaries.projection_visible_polygon_mm)
    calibration.table.inner_polygon_mm = _points_to_tuples(boundaries.physical_rail_polygon_mm)
    calibration.table.center_playable_polygon_mm = _points_to_tuples(boundaries.center_playable_polygon_mm)
    if boundaries.projection_visible_pocket_points_mm:
        calibration.table.projection_visible_pockets_mm = list(boundaries.projection_visible_pocket_points_mm)
    if boundaries.physical_pocket_points_mm:
        calibration.table.pockets_mm = list(boundaries.physical_pocket_points_mm)
    return True


def build_ball_compensation_model(
    samples: Sequence[BallCompensationSample],
    ball_diameter_mm: float,
    max_neighbors: int = 8,
    mode: str = "engineered_ball_comp_v3",
    minimum_samples: int = 20,
    table_width_mm: float | None = None,
    table_height_mm: float | None = None,
) -> BallCompensationModel:
    required = max(4, int(minimum_samples))
    if len(samples) < required:
        raise ValueError(f"At least {required} sampling points are required to build a ball compensation model.")
    controls = np.asarray([sample.detected_camera_px for sample in samples], dtype=np.float64).reshape((-1, 2))
    deltas = np.asarray([sample.delta_table_mm for sample in samples], dtype=np.float64).reshape((-1, 2))
    targets = np.asarray([sample.target_table_mm for sample in samples], dtype=np.float64).reshape((-1, 2))
    radii = np.asarray([float(sample.detected_radius_px) for sample in samples], dtype=np.float64)
    spreads = np.asarray([float(sample.stability_spread_px) for sample in samples], dtype=np.float64)
    confidences = np.asarray([float(sample.detection_confidence) for sample in samples], dtype=np.float64)
    geometry_qualities = np.asarray([float(sample.geometry_quality) for sample in samples], dtype=np.float64)
    sample_weights = np.asarray(
        [
            np.clip(
                float(sample.detection_confidence)
                * float(sample.geometry_quality) ** 2
                * _geometry_method_weight(sample.geometry_method)
                / (1.0 + float(sample.stability_spread_px) ** 2),
                0.01,
                1.0,
            )
            for sample in samples
        ],
        dtype=np.float64,
    )
    delta_norms = np.linalg.norm(deltas, axis=1)

    quality_gate = evaluate_ball_compensation_quality(
        controls,
        targets,
        sample_weights=sample_weights,
        ball_diameter_mm=ball_diameter_mm,
        table_width_mm=table_width_mm,
        table_height_mm=table_height_mm,
    )

    quality_report = {
        "sample_count": int(len(samples)),
        "ball_diameter_mm": float(ball_diameter_mm),
        "camera_span_px": _span_report(controls),
        "table_span_mm": _span_report(targets),
        "delta_norm_mm": _stats_report(delta_norms),
        "stability_spread_px": _stats_report(spreads),
        "detected_radius_px": _stats_report(radii),
        "detection_confidence": _stats_report(confidences),
        "geometry_quality": _stats_report(geometry_qualities),
        "geometry_methods": sorted({str(sample.geometry_method) for sample in samples}),
        **quality_gate,
    }
    if not bool(quality_gate["quality_gate_passed"]):
        raise ValueError(
            "Ball compensation quality gate failed: "
            + "; ".join(str(error) for error in quality_gate["quality_gate_errors"])
        )
    return BallCompensationModel(
        mode=str(mode or "engineered_ball_comp_v3"),
        control_points_camera_px=controls,
        delta_table_mm=deltas,
        target_table_mm=targets,
        sample_weights=sample_weights,
        max_neighbors=max(1, min(int(max_neighbors), len(samples))),
        quality_report=quality_report,
    )


def split_ball_compensation_samples(
    samples: Sequence[BallCompensationSample],
    *,
    holdout_ratio: float = 0.18,
    minimum_holdout: int = 8,
) -> tuple[list[BallCompensationSample], list[BallCompensationSample]]:
    """Reserve spatially spread samples for honest end-to-end validation."""

    items = list(samples)
    if len(items) < 24:
        return items, []
    count = min(len(items) - 20, max(int(minimum_holdout), int(round(len(items) * float(holdout_ratio)))))
    points = np.asarray([sample.target_table_mm for sample in items], dtype=np.float64)
    scale = np.maximum(np.ptp(points, axis=0), 1.0)
    normalized = (points - np.min(points, axis=0)) / scale
    center = np.asarray([0.5, 0.5], dtype=np.float64)
    selected = [int(np.argmax(np.linalg.norm(normalized - center, axis=1)))]
    while len(selected) < count:
        distance = np.min(
            np.linalg.norm(normalized[:, None, :] - normalized[np.asarray(selected)][None, :, :], axis=2),
            axis=1,
        )
        distance[np.asarray(selected, dtype=np.int32)] = -1.0
        selected.append(int(np.argmax(distance)))
    held = set(selected)
    return [sample for index, sample in enumerate(items) if index not in held], [items[index] for index in selected]


def evaluate_ball_compensation_holdout(
    samples: Sequence[BallCompensationSample],
    calibration,
) -> Dict[str, object]:
    if not samples:
        return {
            "sample_count": 0,
            "error_mm": _stats_report(np.zeros((0,), dtype=np.float64)),
            "mean_vector_mm": [0.0, 0.0],
            "quality_gate_passed": False,
            "quality_gate_errors": ["no holdout samples were reserved"],
        }
    camera = np.asarray([sample.detected_camera_px for sample in samples], dtype=np.float32)
    target = np.asarray([sample.target_table_mm for sample in samples], dtype=np.float64)
    predicted = calibration.ball_camera_px_to_table_mm(camera).astype(np.float64)
    vectors = predicted - target
    error_report = _stats_report(np.linalg.norm(vectors, axis=1))
    mean_vector = np.mean(vectors, axis=0)
    ball_diameter_mm = float(calibration.table.ball_diameter_mm)
    maximum_p95_mm = max(
        MIN_BALL_HOLDOUT_P95_MM,
        ball_diameter_mm * MAX_BALL_HOLDOUT_P95_BALL_DIAMETER_RATIO,
    )
    maximum_bias_mm = max(
        MIN_BALL_HOLDOUT_BIAS_MM,
        ball_diameter_mm * MAX_BALL_HOLDOUT_BIAS_BALL_DIAMETER_RATIO,
    )
    quality_errors: list[str] = []
    if float(error_report["p95"]) > maximum_p95_mm:
        quality_errors.append(
            f"holdout P95 {float(error_report['p95']):.2f} mm exceeds {maximum_p95_mm:.2f} mm"
        )
    bias_mm = float(np.linalg.norm(mean_vector))
    if bias_mm > maximum_bias_mm:
        quality_errors.append(
            f"holdout mean bias {bias_mm:.2f} mm exceeds {maximum_bias_mm:.2f} mm"
        )
    return {
        "sample_count": int(len(samples)),
        "error_mm": error_report,
        "mean_vector_mm": mean_vector.tolist(),
        "maximum_p95_mm": float(maximum_p95_mm),
        "maximum_mean_bias_mm": float(maximum_bias_mm),
        "quality_gate_passed": not quality_errors,
        "quality_gate_errors": quality_errors,
        "samples": [
            {
                "sample_index": int(sample.sample_index),
                "target_table_mm": list(sample.target_table_mm),
                "predicted_table_mm": predicted[index].tolist(),
                "error_vector_mm": vectors[index].tolist(),
                "error_mm": float(np.linalg.norm(vectors[index])),
            }
            for index, sample in enumerate(samples)
        ],
    }


def fit_and_validate_ball_compensation(
    samples: Sequence[BallCompensationSample],
    calibration,
    *,
    calibration_context: Optional[Dict[str, Any]] = None,
) -> ValidatedBallCompensation:
    """Fit the runtime model and validate it through the same calibration seam used by the wizard."""

    training_samples, holdout_samples = split_ball_compensation_samples(samples)
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
        holdout_report = evaluate_ball_compensation_holdout(holdout_samples, calibration)
        if not bool(holdout_report.get("quality_gate_passed", False)):
            raise BallCompensationValidationError(
                holdout_report,
                training_count=len(training_samples),
                holdout_count=len(holdout_samples),
            )
    except Exception:
        calibration.ball_compensation_model = previous_model
        calibration._rebuild_geometry()
        raise
    return ValidatedBallCompensation(
        model=model,
        training_samples=tuple(training_samples),
        holdout_samples=tuple(holdout_samples),
        holdout_report=holdout_report,
    )


def evaluate_ball_compensation_quality(
    control_points_camera_px: np.ndarray,
    target_table_mm: np.ndarray,
    *,
    sample_weights: np.ndarray | None,
    ball_diameter_mm: float,
    table_width_mm: float | None = None,
    table_height_mm: float | None = None,
) -> Dict[str, object]:
    """Evaluate a persisted or newly sampled model with the runtime map family."""

    controls = np.asarray(control_points_camera_px, dtype=np.float64).reshape((-1, 2))
    targets = np.asarray(target_table_mm, dtype=np.float64).reshape((-1, 2))
    weights = None if sample_weights is None else np.asarray(sample_weights, dtype=np.float64).reshape((-1,))

    # Import lazily so the persistence module remains independent while using
    # the exact regularized mapping family selected by IndependentGeometry.
    from .geometry import ball_center_map_quality

    mapping_quality = ball_center_map_quality(
        controls,
        targets,
        sample_weights=weights,
    )
    maximum_cv_p95_mm = max(
        MIN_BALL_MAP_CV_P95_MM,
        float(ball_diameter_mm) * MAX_BALL_MAP_CV_P95_BALL_DIAMETER_RATIO,
    )
    quality_errors: list[str] = []
    cv_p95_mm = float(mapping_quality.get("cv_p95", float("inf")))
    if not bool(mapping_quality.get("model_available", False)):
        quality_errors.append("cross-validation model is unavailable")
    elif not np.isfinite(cv_p95_mm) or cv_p95_mm > maximum_cv_p95_mm:
        quality_errors.append(
            f"cross-validation P95 {cv_p95_mm:.2f} mm exceeds {maximum_cv_p95_mm:.2f} mm"
        )

    coverage_report = _target_coverage_report(
        targets,
        table_width_mm=table_width_mm,
        table_height_mm=table_height_mm,
    )
    if coverage_report.get("evaluated"):
        width_ratio = float(coverage_report["width_ratio"])
        height_ratio = float(coverage_report["height_ratio"])
        hull_area_ratio = float(coverage_report["hull_area_ratio"])
        if width_ratio < MIN_BALL_TARGET_WIDTH_RATIO:
            quality_errors.append(
                f"target width coverage {width_ratio:.1%} is below {MIN_BALL_TARGET_WIDTH_RATIO:.0%}"
            )
        if height_ratio < MIN_BALL_TARGET_HEIGHT_RATIO:
            quality_errors.append(
                f"target height coverage {height_ratio:.1%} is below {MIN_BALL_TARGET_HEIGHT_RATIO:.0%}"
            )
        if hull_area_ratio < MIN_BALL_TARGET_HULL_AREA_RATIO:
            quality_errors.append(
                f"target hull coverage {hull_area_ratio:.1%} is below {MIN_BALL_TARGET_HULL_AREA_RATIO:.0%}"
            )

    return {
        "mapping_cross_validation": {
            "model_kind": str(mapping_quality.get("model_kind", "unavailable")),
            "degree": int(mapping_quality.get("degree", 0)),
            "p95_mm": cv_p95_mm,
            "maximum_p95_mm": float(maximum_cv_p95_mm),
        },
        "target_coverage": coverage_report,
        "quality_gate_passed": not quality_errors,
        "quality_gate_errors": quality_errors,
    }


def _geometry_method_weight(method: str) -> float:
    normalized = str(method or "unknown").strip().lower()
    if normalized.startswith("segmentation_ellipse"):
        return 1.0
    if normalized.startswith("appearance_ellipse"):
        return 0.85
    if normalized.startswith("bbox"):
        return 0.25
    return 0.50


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


def _target_coverage_report(
    points: np.ndarray,
    *,
    table_width_mm: float | None,
    table_height_mm: float | None,
) -> Dict[str, float | bool]:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    width = float(table_width_mm or 0.0)
    height = float(table_height_mm or 0.0)
    if pts.shape[0] < 3 or width <= 0.0 or height <= 0.0:
        return {"evaluated": False}
    span = np.ptp(pts, axis=0)
    hull = cv2.convexHull(pts.astype(np.float32)).reshape((-1, 2))
    hull_area = abs(float(cv2.contourArea(hull.reshape((-1, 1, 2))))) if hull.shape[0] >= 3 else 0.0
    return {
        "evaluated": True,
        "width_ratio": float(span[0] / width),
        "height_ratio": float(span[1] / height),
        "hull_area_ratio": float(hull_area / (width * height)),
        "minimum_width_ratio": float(MIN_BALL_TARGET_WIDTH_RATIO),
        "minimum_height_ratio": float(MIN_BALL_TARGET_HEIGHT_RATIO),
        "minimum_hull_area_ratio": float(MIN_BALL_TARGET_HULL_AREA_RATIO),
    }


def _sampling_grid_inside_polygon(
    polygon_mm: np.ndarray,
    *,
    target_count: int,
    cols: int,
    rows: int,
) -> np.ndarray:
    poly = np.asarray(polygon_mm, dtype=np.float32).reshape((-1, 2))
    if poly.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64)
    x_min = float(np.min(poly[:, 0]))
    x_max = float(np.max(poly[:, 0]))
    y_min = float(np.min(poly[:, 1]))
    y_max = float(np.max(poly[:, 1]))
    dense_cols = max(cols * 4, cols + 4)
    dense_rows = max(rows * 4, rows + 4)
    xs = np.linspace(x_min, x_max, dense_cols, dtype=np.float64)
    ys = np.linspace(y_min, y_max, dense_rows, dtype=np.float64)
    candidates: list[list[float]] = []
    boundary_scores: list[float] = []
    contour = poly.reshape((-1, 1, 2)).astype(np.float32)
    for y in ys:
        for x in xs:
            signed_distance = float(cv2.pointPolygonTest(contour, (float(x), float(y)), True))
            if signed_distance < 0.0:
                continue
            candidates.append([float(x), float(y)])
            boundary_scores.append(signed_distance)
    if not candidates:
        return np.zeros((0, 2), dtype=np.float64)
    points = np.asarray(candidates, dtype=np.float64).reshape((-1, 2))
    scores = np.asarray(boundary_scores, dtype=np.float64).reshape((-1,))
    selected = _select_spread_points(points, scores, target_count=min(max(4, target_count), points.shape[0]))
    order = np.lexsort((selected[:, 0], selected[:, 1]))
    return selected[order].reshape((-1, 2))


def _polygon_edge_priority_points(
    polygon_mm: np.ndarray,
    *,
    target_count: int,
    priority_points_mm: np.ndarray | None = None,
) -> np.ndarray:
    poly = np.asarray(polygon_mm, dtype=np.float32).reshape((-1, 2))
    if poly.shape[0] < 3 or target_count <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    perimeter_candidates = _sample_polygon_perimeter(
        poly,
        sample_count=max(target_count * 24, poly.shape[0] * 2),
    )
    perimeter_points = _select_spaced_points(
        perimeter_candidates,
        min(max(target_count * 2, target_count), perimeter_candidates.shape[0]),
    )
    anchors = _nearest_polygon_boundary_points(poly, priority_points_mm)
    merged = _merge_sampling_sets(
        anchors,
        perimeter_points,
        target_count=target_count,
        min_spacing=max(
            2.0,
            0.42 * _estimated_sampling_spacing(poly, cols=max(2, target_count // 2), rows=2),
        ),
    )
    return merged if merged.shape[0] > 0 else perimeter_points[:target_count].reshape((-1, 2))


def _nearest_polygon_boundary_points(
    polygon_mm: np.ndarray,
    points_mm: np.ndarray | None,
) -> np.ndarray:
    poly = np.asarray(polygon_mm, dtype=np.float64).reshape((-1, 2))
    points = (
        np.asarray(points_mm, dtype=np.float64).reshape((-1, 2))
        if points_mm is not None
        else np.zeros((0, 2), dtype=np.float64)
    )
    if poly.shape[0] < 2 or points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    closed = np.vstack([poly, poly[0:1]])
    starts = closed[:-1]
    segments = closed[1:] - starts
    segment_norm_sq = np.sum(segments * segments, axis=1)
    out: list[list[float]] = []
    for point in points:
        rel = point.reshape((1, 2)) - starts
        t = np.divide(
            np.sum(rel * segments, axis=1),
            segment_norm_sq,
            out=np.zeros_like(segment_norm_sq),
            where=segment_norm_sq > 1e-9,
        )
        t = np.clip(t, 0.0, 1.0)
        projected = starts + t[:, None] * segments
        nearest = projected[int(np.argmin(np.sum((projected - point.reshape((1, 2))) ** 2, axis=1)))]
        out.append([float(nearest[0]), float(nearest[1])])
    return _unique_points(np.asarray(out, dtype=np.float64).reshape((-1, 2)))


def _sample_polygon_perimeter(polygon_mm: np.ndarray, *, sample_count: int) -> np.ndarray:
    poly = np.asarray(polygon_mm, dtype=np.float64).reshape((-1, 2))
    if poly.shape[0] < 2 or sample_count <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    closed = np.vstack([poly, poly[0:1]])
    segments = closed[1:] - closed[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    perimeter = float(np.sum(lengths))
    if perimeter <= 1e-6:
        return poly.copy()
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    positions = np.linspace(0.0, perimeter, int(sample_count), endpoint=False, dtype=np.float64)
    sampled: list[list[float]] = []
    for position in positions:
        edge_idx = int(np.searchsorted(cumulative, position, side="right") - 1)
        edge_idx = max(0, min(edge_idx, len(lengths) - 1))
        seg_len = float(lengths[edge_idx])
        t = 0.0 if seg_len <= 1e-6 else float((position - cumulative[edge_idx]) / seg_len)
        point = closed[edge_idx] + t * segments[edge_idx]
        sampled.append([float(point[0]), float(point[1])])
    return np.asarray(sampled, dtype=np.float64).reshape((-1, 2))


def _estimated_sampling_spacing(polygon_mm: np.ndarray, *, cols: int, rows: int) -> float:
    poly = np.asarray(polygon_mm, dtype=np.float64).reshape((-1, 2))
    if poly.shape[0] < 2:
        return 0.0
    spans = np.ptp(poly, axis=0)
    step_x = float(spans[0]) / max(1, int(cols) - 1)
    step_y = float(spans[1]) / max(1, int(rows) - 1)
    return max(8.0, 0.42 * min(step_x, step_y))


def _merge_sampling_sets(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    target_count: int,
    min_spacing: float,
) -> np.ndarray:
    pri = np.asarray(primary, dtype=np.float64).reshape((-1, 2))
    sec = np.asarray(secondary, dtype=np.float64).reshape((-1, 2))
    target = max(1, int(target_count))
    for spacing in (max(0.0, float(min_spacing)), max(0.0, float(min_spacing) * 0.55), 0.0):
        selected: list[list[float]] = []
        _append_points_with_spacing(selected, pri, target_count=target, min_spacing=spacing)
        _append_spread_points_with_spacing(selected, sec, target_count=target, min_spacing=spacing)
        if len(selected) >= target or spacing <= 0.0:
            return np.asarray(selected[:target], dtype=np.float64).reshape((-1, 2))
    return np.zeros((0, 2), dtype=np.float64)


def _append_points_with_spacing(
    selected: list[list[float]],
    points: np.ndarray,
    *,
    target_count: int,
    min_spacing: float,
) -> None:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    for point in pts:
        if len(selected) >= target_count:
            return
        if not selected:
            selected.append([float(point[0]), float(point[1])])
            continue
        picked = np.asarray(selected, dtype=np.float64).reshape((-1, 2))
        distances = np.linalg.norm(picked - point.reshape((1, 2)), axis=1)
        if float(np.min(distances)) < max(1e-6, float(min_spacing)):
            continue
        selected.append([float(point[0]), float(point[1])])


def _append_spread_points_with_spacing(
    selected: list[list[float]],
    points: np.ndarray,
    *,
    target_count: int,
    min_spacing: float,
) -> None:
    remaining = _unique_points(np.asarray(points, dtype=np.float64).reshape((-1, 2)))
    while len(selected) < target_count and remaining.shape[0] > 0:
        if selected:
            picked = np.asarray(selected, dtype=np.float64).reshape((-1, 2))
            spacing = np.min(
                np.linalg.norm(remaining[:, None, :] - picked[None, :, :], axis=2),
                axis=1,
            )
        else:
            center = np.mean(remaining, axis=0)
            spacing = np.linalg.norm(remaining - center.reshape((1, 2)), axis=1)
        next_index = int(np.argmax(spacing))
        if selected and float(spacing[next_index]) < max(1e-6, float(min_spacing)):
            return
        point = remaining[next_index]
        selected.append([float(point[0]), float(point[1])])
        remaining = np.delete(remaining, next_index, axis=0)


def _select_spread_points(points: np.ndarray, boundary_scores: np.ndarray, target_count: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    scores = np.asarray(boundary_scores, dtype=np.float64).reshape((-1,))
    if pts.shape[0] <= target_count:
        return pts
    centroid = np.mean(pts, axis=0)
    center_penalty = np.linalg.norm(pts - centroid.reshape((1, 2)), axis=1)
    first_idx = int(np.argmax(scores - 0.12 * center_penalty))
    selected_indices = [first_idx]
    remaining = np.ones((pts.shape[0],), dtype=bool)
    remaining[first_idx] = False
    while len(selected_indices) < target_count and bool(np.any(remaining)):
        remain_pts = pts[remaining]
        remain_scores = scores[remaining]
        picked_pts = pts[np.asarray(selected_indices, dtype=np.int32)]
        spacing = np.min(np.linalg.norm(remain_pts[:, None, :] - picked_pts[None, :, :], axis=2), axis=1)
        combined = spacing + 0.35 * remain_scores
        remain_indices = np.flatnonzero(remaining)
        next_idx = int(remain_indices[int(np.argmax(combined))])
        selected_indices.append(next_idx)
        remaining[next_idx] = False
    return pts[np.asarray(selected_indices, dtype=np.int32)].reshape((-1, 2))


def _select_spaced_points(points: np.ndarray, target_count: int) -> np.ndarray:
    pts = _unique_points(np.asarray(points, dtype=np.float64).reshape((-1, 2)))
    if pts.shape[0] <= target_count:
        return pts
    centroid = np.mean(pts, axis=0)
    radial = np.linalg.norm(pts - centroid.reshape((1, 2)), axis=1)
    first_idx = int(np.argmax(radial))
    selected_indices = [first_idx]
    remaining = np.ones((pts.shape[0],), dtype=bool)
    remaining[first_idx] = False
    while len(selected_indices) < target_count and bool(np.any(remaining)):
        remain_pts = pts[remaining]
        remain_radial = radial[remaining]
        picked_pts = pts[np.asarray(selected_indices, dtype=np.int32)]
        spacing = np.min(np.linalg.norm(remain_pts[:, None, :] - picked_pts[None, :, :], axis=2), axis=1)
        combined = spacing + 0.08 * remain_radial
        remain_indices = np.flatnonzero(remaining)
        next_idx = int(remain_indices[int(np.argmax(combined))])
        selected_indices.append(next_idx)
        remaining[next_idx] = False
    return pts[np.asarray(selected_indices, dtype=np.int32)].reshape((-1, 2))


def _unique_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    if pts.shape[0] <= 1:
        return pts
    rounded = np.round(pts, decimals=4)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return pts[np.sort(indices)].reshape((-1, 2))


def _points_to_tuples(points: np.ndarray) -> list[tuple[float, float]]:
    pts = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    return [(float(point[0]), float(point[1])) for point in pts]
