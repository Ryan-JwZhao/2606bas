from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import cv2
import numpy as np

from .ball_compensation import BallCompensationModel
from ..table_boundaries import EdgeInsets, derive_table_boundaries, inset_polygon_uniform


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
    preferred_polygon_mm: np.ndarray | None = None,
    extra_safe_inset_mm: float | None = None,
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
            return polygon_grid

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


def _points_to_tuples(points: np.ndarray) -> list[tuple[float, float]]:
    pts = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    return [(float(point[0]), float(point[1])) for point in pts]
