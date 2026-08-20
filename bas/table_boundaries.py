from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .geometry import pocket_arc_center
from .utils import ensure_numpy_points


PointArray = np.ndarray


@dataclass(frozen=True)
class EdgeInsets:
    top_mm: float = 0.0
    right_mm: float = 0.0
    bottom_mm: float = 0.0
    left_mm: float = 0.0

    @classmethod
    def uniform(cls, value_mm: float) -> "EdgeInsets":
        value = max(0.0, float(value_mm))
        return cls(top_mm=value, right_mm=value, bottom_mm=value, left_mm=value)


@dataclass
class TableBoundarySet:
    projection_visible_polygon_mm: PointArray
    physical_rail_polygon_mm: PointArray
    center_playable_polygon_mm: PointArray
    projection_visible_pocket_points_mm: List[Tuple[float, float]]
    physical_pocket_points_mm: List[Tuple[float, float]]
    planning_pocket_points_mm: List[Tuple[float, float]]
    planning_pocket_mouths_mm: List[Tuple[Tuple[float, float], Tuple[float, float]]]


def derive_table_boundaries(
    visible_polygon_mm: np.ndarray,
    pocket_curves_mm: Sequence[np.ndarray],
    *,
    table_width_mm: float,
    table_height_mm: float,
    ball_diameter_mm: float,
    projection_visible_insets: EdgeInsets,
    physical_rail_insets: EdgeInsets,
    physical_middle_pocket_relief_top_mm: float,
    physical_middle_pocket_relief_bottom_mm: float,
    center_reachable_extra_margin_mm: float,
) -> TableBoundarySet:
    width = max(1.0, float(table_width_mm))
    height = max(1.0, float(table_height_mm))
    visible_base = _fallback_polygon(visible_polygon_mm, width, height)
    projection_visible = apply_edge_insets(visible_base, width, height, projection_visible_insets)
    physical_rail = apply_edge_insets(visible_base, width, height, physical_rail_insets)
    physical_rail = apply_middle_pocket_relief(
        physical_rail,
        visible_base,
        pocket_curves_mm,
        width_mm=width,
        height_mm=height,
        top_relief_mm=float(physical_middle_pocket_relief_top_mm),
        bottom_relief_mm=float(physical_middle_pocket_relief_bottom_mm),
        insets=physical_rail_insets,
    )
    center_margin = 0.5 * max(0.0, float(ball_diameter_mm)) + max(0.0, float(center_reachable_extra_margin_mm))
    center_playable = inset_polygon_uniform(physical_rail, center_margin, width, height)
    visible_pockets = _pocket_centers_mm(
        pocket_curves_mm,
        width,
        height,
        projection_visible_insets,
        ball_diameter_mm=ball_diameter_mm,
    )
    physical_pockets = _pocket_centers_mm(
        pocket_curves_mm,
        width,
        height,
        physical_rail_insets,
        ball_diameter_mm=ball_diameter_mm,
        middle_relief_top_mm=float(physical_middle_pocket_relief_top_mm),
        middle_relief_bottom_mm=float(physical_middle_pocket_relief_bottom_mm),
    )
    planning_pockets = fit_pocket_centers_mm(
        pocket_curves_mm,
        ball_diameter_mm=ball_diameter_mm,
    )
    planning_mouths = fit_pocket_mouths_mm(pocket_curves_mm)
    return TableBoundarySet(
        projection_visible_polygon_mm=projection_visible,
        physical_rail_polygon_mm=physical_rail,
        center_playable_polygon_mm=center_playable,
        projection_visible_pocket_points_mm=visible_pockets,
        physical_pocket_points_mm=physical_pockets,
        planning_pocket_points_mm=planning_pockets,
        planning_pocket_mouths_mm=planning_mouths,
    )


def apply_edge_insets(points_mm: np.ndarray, width_mm: float, height_mm: float, insets: EdgeInsets) -> PointArray:
    pts = ensure_numpy_points(points_mm).astype(np.float32)
    if pts.shape[0] == 0:
        return pts
    width = max(1.0, float(width_mm))
    height = max(1.0, float(height_mm))
    band = _edge_band(width, height, insets)

    left_dist = np.clip(pts[:, 0], 0.0, width)
    right_dist = np.clip(width - pts[:, 0], 0.0, width)
    top_dist = np.clip(pts[:, 1], 0.0, height)
    bottom_dist = np.clip(height - pts[:, 1], 0.0, height)
    distances = np.stack([top_dist, right_dist, bottom_dist, left_dist], axis=1)
    nearest = np.argmin(distances, axis=1)

    near_top = top_dist <= band
    near_right = right_dist <= band
    near_bottom = bottom_dist <= band
    near_left = left_dist <= band
    has_assignment = near_top | near_right | near_bottom | near_left
    near_top = near_top | (~has_assignment & (nearest == 0))
    near_right = near_right | (~has_assignment & (nearest == 1))
    near_bottom = near_bottom | (~has_assignment & (nearest == 2))
    near_left = near_left | (~has_assignment & (nearest == 3))

    out = pts.copy()
    out[:, 0] += near_left.astype(np.float32) * float(max(0.0, insets.left_mm))
    out[:, 0] -= near_right.astype(np.float32) * float(max(0.0, insets.right_mm))
    out[:, 1] += near_top.astype(np.float32) * float(max(0.0, insets.top_mm))
    out[:, 1] -= near_bottom.astype(np.float32) * float(max(0.0, insets.bottom_mm))
    out[:, 0] = np.clip(out[:, 0], 0.0, width)
    out[:, 1] = np.clip(out[:, 1], 0.0, height)
    return out.astype(np.float32)


def _pocket_centers_mm(
    pocket_curves_mm: Sequence[np.ndarray],
    width_mm: float,
    height_mm: float,
    insets: EdgeInsets,
    *,
    ball_diameter_mm: float,
    middle_relief_top_mm: float = 0.0,
    middle_relief_bottom_mm: float = 0.0,
) -> List[Tuple[float, float]]:
    centers: List[Tuple[float, float]] = []
    top_middle_idx, bottom_middle_idx = _middle_pocket_indices(pocket_curves_mm, width_mm, height_mm)
    for curve in pocket_curves_mm:
        pts = ensure_numpy_points(curve)
        if pts.shape[0] < 2:
            continue
        adjusted = apply_edge_insets(pts, width_mm, height_mm, insets)
        center = pocket_arc_center(adjusted, ball_diameter_mm)
        centers.append((float(center[0]), float(center[1])))
    out = list(centers)
    if top_middle_idx is not None and 0 <= top_middle_idx < len(out):
        x, y = out[top_middle_idx]
        out[top_middle_idx] = (x, y - min(max(0.0, float(middle_relief_top_mm)), max(0.0, float(insets.top_mm))))
    if bottom_middle_idx is not None and 0 <= bottom_middle_idx < len(out):
        x, y = out[bottom_middle_idx]
        out[bottom_middle_idx] = (x, y + min(max(0.0, float(middle_relief_bottom_mm)), max(0.0, float(insets.bottom_mm))))
    return out


def fit_pocket_centers_mm(
    pocket_curves_mm: Sequence[np.ndarray],
    *,
    ball_diameter_mm: float,
) -> List[Tuple[float, float]]:
    """Fit shot targets from untouched jaw arcs.

    Pocket arcs may legitimately extend outside the nominal playing rectangle.
    Rail insets and rectangle clipping therefore must not participate in this
    fit; they describe playable boundaries, not the physical hole centre.
    """

    centers: List[Tuple[float, float]] = []
    for curve in pocket_curves_mm:
        points = ensure_numpy_points(curve)
        if points.shape[0] < 2:
            continue
        center = pocket_arc_center(points, ball_diameter_mm)
        centers.append((float(center[0]), float(center[1])))
    return centers


def fit_pocket_mouths_mm(
    pocket_curves_mm: Sequence[np.ndarray],
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return the two physical jaw tips from every untouched pocket curve.

    Annotation point order is not part of the geometry contract, so selecting
    the farthest pair is more robust than assuming the first and last points
    are the mouth endpoints.
    """

    mouths: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for curve in pocket_curves_mm:
        points = ensure_numpy_points(curve).astype(np.float64)
        if points.shape[0] < 2:
            continue
        differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        squared_distances = np.sum(differences * differences, axis=2)
        jaw_a_index, jaw_b_index = np.unravel_index(int(np.argmax(squared_distances)), squared_distances.shape)
        jaw_a = points[jaw_a_index]
        jaw_b = points[jaw_b_index]
        mouths.append(
            (
                (float(jaw_a[0]), float(jaw_a[1])),
                (float(jaw_b[0]), float(jaw_b[1])),
            )
        )
    return mouths


def _fallback_polygon(points_mm: np.ndarray, width_mm: float, height_mm: float) -> PointArray:
    pts = ensure_numpy_points(points_mm)
    if pts.shape[0] >= 3:
        return pts.astype(np.float32)
    return np.asarray([(0.0, 0.0), (width_mm, 0.0), (width_mm, height_mm), (0.0, height_mm)], dtype=np.float32)


def _edge_band(width_mm: float, height_mm: float, insets: EdgeInsets) -> float:
    max_inset = max(float(insets.top_mm), float(insets.right_mm), float(insets.bottom_mm), float(insets.left_mm), 0.0)
    return float(max(24.0, 0.08 * min(width_mm, height_mm), 2.0 * max_inset))


def apply_middle_pocket_relief(
    adjusted_points_mm: np.ndarray,
    base_points_mm: np.ndarray,
    pocket_curves_mm: Sequence[np.ndarray],
    *,
    width_mm: float,
    height_mm: float,
    top_relief_mm: float,
    bottom_relief_mm: float,
    insets: EdgeInsets,
) -> PointArray:
    pts = ensure_numpy_points(adjusted_points_mm).astype(np.float32)
    base = ensure_numpy_points(base_points_mm).astype(np.float32)
    if pts.shape[0] == 0 or base.shape[0] != pts.shape[0]:
        return pts
    top_middle_idx, bottom_middle_idx = _middle_pocket_indices(pocket_curves_mm, width_mm, height_mm)
    out = pts.copy()
    if top_middle_idx is not None and top_middle_idx < len(pocket_curves_mm) and top_relief_mm > 0.0 and float(insets.top_mm) > 0.0:
        curve = ensure_numpy_points(pocket_curves_mm[top_middle_idx]).astype(np.float32)
        out[:, 1] -= _pocket_relief_weights(base, curve, width_mm) * min(float(top_relief_mm), float(insets.top_mm))
    if bottom_middle_idx is not None and bottom_middle_idx < len(pocket_curves_mm) and bottom_relief_mm > 0.0 and float(insets.bottom_mm) > 0.0:
        curve = ensure_numpy_points(pocket_curves_mm[bottom_middle_idx]).astype(np.float32)
        out[:, 1] += _pocket_relief_weights(base, curve, width_mm) * min(float(bottom_relief_mm), float(insets.bottom_mm))
    out[:, 0] = np.clip(out[:, 0], 0.0, float(width_mm))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(height_mm))
    return out.astype(np.float32)


def inset_polygon_uniform(points_mm: np.ndarray, inset_mm: float, width_mm: float, height_mm: float) -> PointArray:
    pts = ensure_numpy_points(points_mm).astype(np.float32)
    inset = max(0.0, float(inset_mm))
    if pts.shape[0] < 3 or inset <= 1e-6:
        return pts
    scale = float(min(1.0, max(0.35, 900.0 / max(width_mm, height_mm, 1.0))))
    inset_px = max(1, int(round(inset * scale)))
    pad = inset_px + 8
    canvas_w = max(32, int(round(width_mm * scale)) + 2 * pad + 4)
    canvas_h = max(32, int(round(height_mm * scale)) + 2 * pad + 4)
    poly_px = np.round(pts * scale).astype(np.int32) + np.asarray([pad, pad], dtype=np.int32)
    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_px.reshape((-1, 1, 2))], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset_px + 1, 2 * inset_px + 1))
    eroded = cv2.erode(mask, kernel)
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return pts
    contour = max(contours, key=cv2.contourArea).reshape((-1, 2)).astype(np.float32)
    contour -= np.asarray([pad, pad], dtype=np.float32)
    contour /= scale
    contour[:, 0] = np.clip(contour[:, 0], 0.0, float(width_mm))
    contour[:, 1] = np.clip(contour[:, 1], 0.0, float(height_mm))
    return contour.astype(np.float32)


def _middle_pocket_indices(
    pocket_curves_mm: Sequence[np.ndarray],
    width_mm: float,
    height_mm: float,
) -> tuple[int | None, int | None]:
    top_idx: int | None = None
    bottom_idx: int | None = None
    top_score = float("inf")
    bottom_score = -float("inf")
    x_lo = 0.25 * float(width_mm)
    x_hi = 0.75 * float(width_mm)
    mid_y = 0.5 * float(height_mm)
    for idx, curve in enumerate(pocket_curves_mm):
        pts = ensure_numpy_points(curve)
        if pts.shape[0] < 2:
            continue
        center = np.mean(pts, axis=0)
        if not (x_lo <= float(center[0]) <= x_hi):
            continue
        if float(center[1]) <= mid_y and float(center[1]) < top_score:
            top_score = float(center[1])
            top_idx = idx
        if float(center[1]) >= mid_y and float(center[1]) > bottom_score:
            bottom_score = float(center[1])
            bottom_idx = idx
    return top_idx, bottom_idx


def _pocket_relief_weights(points_mm: np.ndarray, curve_mm: np.ndarray, width_mm: float) -> np.ndarray:
    pts = ensure_numpy_points(points_mm).astype(np.float32)
    curve = ensure_numpy_points(curve_mm).astype(np.float32)
    if pts.shape[0] == 0 or curve.shape[0] < 2:
        return np.zeros((pts.shape[0],), dtype=np.float32)
    span_x = float(np.max(curve[:, 0]) - np.min(curve[:, 0]))
    radius = max(80.0, 0.11 * float(width_mm), 1.2 * span_x)
    weights = np.zeros((pts.shape[0],), dtype=np.float32)
    for idx, point in enumerate(pts):
        dist = _point_to_polyline_distance(point, curve)
        if dist >= radius:
            continue
        weights[idx] = float((1.0 - dist / radius) ** 2)
    return weights


def _point_to_polyline_distance(point: np.ndarray, curve: np.ndarray) -> float:
    best = float("inf")
    for idx in range(curve.shape[0] - 1):
        best = min(best, _point_to_segment_distance(point, curve[idx], curve[idx + 1]))
    return best if np.isfinite(best) else 0.0


def _point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    seg = b - a
    denom = float(np.dot(seg, seg))
    if denom <= 1e-6:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, seg) / denom)
    t = min(1.0, max(0.0, t))
    proj = a + seg * t
    return float(np.linalg.norm(point - proj))
