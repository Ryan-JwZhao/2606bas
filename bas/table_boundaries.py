from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

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


def derive_table_boundaries(
    visible_polygon_mm: np.ndarray,
    pocket_curves_mm: Sequence[np.ndarray],
    *,
    table_width_mm: float,
    table_height_mm: float,
    ball_diameter_mm: float,
    projection_visible_insets: EdgeInsets,
    physical_rail_insets: EdgeInsets,
    center_reachable_extra_margin_mm: float,
) -> TableBoundarySet:
    width = max(1.0, float(table_width_mm))
    height = max(1.0, float(table_height_mm))
    visible_base = _fallback_polygon(visible_polygon_mm, width, height)
    projection_visible = apply_edge_insets(visible_base, width, height, projection_visible_insets)
    physical_rail = apply_edge_insets(visible_base, width, height, physical_rail_insets)
    center_margin = 0.5 * max(0.0, float(ball_diameter_mm)) + max(0.0, float(center_reachable_extra_margin_mm))
    center_playable = apply_edge_insets(physical_rail, width, height, EdgeInsets.uniform(center_margin))
    visible_pockets = _pocket_centers_mm(pocket_curves_mm, width, height, projection_visible_insets)
    physical_pockets = _pocket_centers_mm(pocket_curves_mm, width, height, physical_rail_insets)
    return TableBoundarySet(
        projection_visible_polygon_mm=projection_visible,
        physical_rail_polygon_mm=physical_rail,
        center_playable_polygon_mm=center_playable,
        projection_visible_pocket_points_mm=visible_pockets,
        physical_pocket_points_mm=physical_pockets,
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
) -> List[Tuple[float, float]]:
    centers: List[Tuple[float, float]] = []
    for curve in pocket_curves_mm:
        pts = ensure_numpy_points(curve)
        if pts.shape[0] < 2:
            continue
        adjusted = apply_edge_insets(pts, width_mm, height_mm, insets)
        center = np.mean(adjusted, axis=0)
        centers.append((float(center[0]), float(center[1])))
    return centers


def _fallback_polygon(points_mm: np.ndarray, width_mm: float, height_mm: float) -> PointArray:
    pts = ensure_numpy_points(points_mm)
    if pts.shape[0] >= 3:
        return pts.astype(np.float32)
    return np.asarray([(0.0, 0.0), (width_mm, 0.0), (width_mm, height_mm), (0.0, height_mm)], dtype=np.float32)


def _edge_band(width_mm: float, height_mm: float, insets: EdgeInsets) -> float:
    max_inset = max(float(insets.top_mm), float(insets.right_mm), float(insets.bottom_mm), float(insets.left_mm), 0.0)
    return float(max(24.0, 0.08 * min(width_mm, height_mm), 2.0 * max_inset))
