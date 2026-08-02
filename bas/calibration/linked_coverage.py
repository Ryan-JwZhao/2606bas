from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import cv2
import numpy as np

from ..utils import ensure_numpy_points
from .projector import polygon_quad


LINKED_COVERAGE_GRID_COLS = 5
LINKED_COVERAGE_GRID_ROWS = 4
LINKED_EDGE_HORIZONTAL_SEGMENTS = 8
LINKED_EDGE_VERTICAL_SEGMENTS = 4
LINKED_EDGE_BAND_RATIO = 0.14
MIN_LINKED_TOTAL_MATCHED_POINTS = 80
MIN_LINKED_COVERAGE_WIDTH_RATIO = 0.85
MIN_LINKED_COVERAGE_HEIGHT_RATIO = 0.85
MIN_LINKED_COVERAGE_HULL_AREA_RATIO = 0.72
MIN_LINKED_CORE_GRID_OCCUPIED_RATIO = 1.0
MIN_LINKED_EDGE_COVERAGE_RATIO = 1.0
MAX_LINKED_HOLE_DISTANCE_RATIO = 0.14


@dataclass(frozen=True)
class LinkedPatternPlacement:
    pattern_id: str
    title: str
    emphasis_zone: str
    center_x_ratio: float
    center_y_ratio: float
    width_ratio: float = 0.24
    height_ratio: float = 0.30


def dense_linked_pattern_placements() -> tuple[LinkedPatternPlacement, ...]:
    """Return overlapping tiles whose planned corners cover every core/edge bin."""

    x_centers = (0.04, 0.27, 0.50, 0.73, 0.96)
    y_centers = (0.05, 0.35, 0.65, 0.95)
    return tuple(
        LinkedPatternPlacement(
            pattern_id=f"coverage_r{row + 1}_c{col + 1}",
            title=f"全桌覆盖 {row + 1}-{col + 1}",
            emphasis_zone=f"coverage_r{row + 1}_c{col + 1}",
            center_x_ratio=float(x_ratio),
            center_y_ratio=float(y_ratio),
        )
        for row, y_ratio in enumerate(y_centers)
        for col, x_ratio in enumerate(x_centers)
    )


def evaluate_linked_point_coverage(
    points: np.ndarray,
    table_polygon: np.ndarray,
) -> Dict[str, object]:
    """Measure exterior span, internal holes and segmented four-edge coverage."""

    normalized = _normalize_to_table(points, table_polygon)
    if normalized.shape[0] < 3:
        return _empty_report()
    normalized = normalized[
        (normalized[:, 0] >= 0.0)
        & (normalized[:, 0] <= 1.0)
        & (normalized[:, 1] >= 0.0)
        & (normalized[:, 1] <= 1.0)
    ]
    if normalized.shape[0] < 3:
        return _empty_report()
    clipped = np.clip(normalized, 0.0, 1.0)
    width_ratio = float(np.ptp(clipped[:, 0]))
    height_ratio = float(np.ptp(clipped[:, 1]))
    hull = cv2.convexHull(clipped.astype(np.float32)).reshape((-1, 2))
    hull_area_ratio = float(np.clip(abs(cv2.contourArea(hull.reshape((-1, 1, 2)))), 0.0, 1.0))
    grid_counts = _grid_counts(clipped, LINKED_COVERAGE_GRID_ROWS, LINKED_COVERAGE_GRID_COLS)
    occupied_ratio = float(np.count_nonzero(grid_counts) / grid_counts.size)
    edge_coverage, edge_counts = _edge_coverage(clipped)
    maximum_hole = _maximum_hole_distance(clipped)
    return {
        "evaluated_point_count": int(clipped.shape[0]),
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "hull_area_ratio": hull_area_ratio,
        "core_grid_rows": LINKED_COVERAGE_GRID_ROWS,
        "core_grid_cols": LINKED_COVERAGE_GRID_COLS,
        "core_grid_counts": grid_counts.tolist(),
        "core_grid_occupied_ratio": occupied_ratio,
        "edge_coverage": edge_coverage,
        "edge_segment_counts": edge_counts,
        "maximum_hole_distance_ratio": maximum_hole,
    }


def linked_coverage_errors(report: Dict[str, object], *, matched_point_count: int) -> list[str]:
    errors: list[str] = []
    if int(matched_point_count) < MIN_LINKED_TOTAL_MATCHED_POINTS:
        errors.append(
            f"matched points {int(matched_point_count)}/{MIN_LINKED_TOTAL_MATCHED_POINTS}"
        )
    checks = (
        ("width", float(report.get("width_ratio", 0.0)), MIN_LINKED_COVERAGE_WIDTH_RATIO),
        ("height", float(report.get("height_ratio", 0.0)), MIN_LINKED_COVERAGE_HEIGHT_RATIO),
        ("hull", float(report.get("hull_area_ratio", 0.0)), MIN_LINKED_COVERAGE_HULL_AREA_RATIO),
        (
            "core grid",
            float(report.get("core_grid_occupied_ratio", 0.0)),
            MIN_LINKED_CORE_GRID_OCCUPIED_RATIO,
        ),
    )
    for name, actual, required in checks:
        if actual + 1e-9 < required:
            errors.append(f"{name} coverage {actual:.1%}/{required:.0%}")
    edge = report.get("edge_coverage", {})
    if not isinstance(edge, dict):
        edge = {}
    for name in ("top", "right", "bottom", "left"):
        actual = float(edge.get(name, 0.0))
        if actual + 1e-9 < MIN_LINKED_EDGE_COVERAGE_RATIO:
            errors.append(f"{name} edge coverage {actual:.1%}/{MIN_LINKED_EDGE_COVERAGE_RATIO:.0%}")
    hole = float(report.get("maximum_hole_distance_ratio", float("inf")))
    if not np.isfinite(hole) or hole > MAX_LINKED_HOLE_DISTANCE_RATIO:
        errors.append(f"maximum hole {hole:.3f}/{MAX_LINKED_HOLE_DISTANCE_RATIO:.2f}")
    return errors


def _normalize_to_table(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    pts = ensure_numpy_points(points).astype(np.float32)
    quad = polygon_quad(ensure_numpy_points(polygon)).astype(np.float32)
    if pts.shape[0] == 0 or quad.shape[0] != 4:
        return np.zeros((0, 2), dtype=np.float32)
    rect = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(quad, rect)
    return cv2.perspectiveTransform(pts.reshape((-1, 1, 2)), transform).reshape((-1, 2))


def _grid_counts(points: np.ndarray, rows: int, cols: int) -> np.ndarray:
    counts = np.zeros((rows, cols), dtype=np.int32)
    for x, y in points:
        col = min(cols - 1, max(0, int(float(x) * cols)))
        row = min(rows - 1, max(0, int(float(y) * rows)))
        counts[row, col] += 1
    return counts


def _edge_coverage(points: np.ndarray) -> tuple[dict[str, float], dict[str, list[int]]]:
    band = LINKED_EDGE_BAND_RATIO
    definitions: Sequence[tuple[str, np.ndarray, np.ndarray, int]] = (
        ("top", points[:, 1] <= band, points[:, 0], LINKED_EDGE_HORIZONTAL_SEGMENTS),
        ("right", points[:, 0] >= 1.0 - band, points[:, 1], LINKED_EDGE_VERTICAL_SEGMENTS),
        ("bottom", points[:, 1] >= 1.0 - band, points[:, 0], LINKED_EDGE_HORIZONTAL_SEGMENTS),
        ("left", points[:, 0] <= band, points[:, 1], LINKED_EDGE_VERTICAL_SEGMENTS),
    )
    coverage: dict[str, float] = {}
    counts_by_edge: dict[str, list[int]] = {}
    for name, mask, coordinate, segments in definitions:
        counts = np.zeros((segments,), dtype=np.int32)
        for value in coordinate[mask]:
            index = min(segments - 1, max(0, int(float(value) * segments)))
            counts[index] += 1
        coverage[name] = float(np.count_nonzero(counts) / segments)
        counts_by_edge[name] = counts.tolist()
    return coverage, counts_by_edge


def _maximum_hole_distance(points: np.ndarray) -> float:
    xs = np.linspace(0.02, 0.98, 25, dtype=np.float64)
    ys = np.linspace(0.02, 0.98, 17, dtype=np.float64)
    probes = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float64)
    distances = np.min(np.linalg.norm(probes[:, None, :] - points[None, :, :], axis=2), axis=1)
    return float(np.max(distances))


def _empty_report() -> Dict[str, object]:
    return {
        "width_ratio": 0.0,
        "evaluated_point_count": 0,
        "height_ratio": 0.0,
        "hull_area_ratio": 0.0,
        "core_grid_rows": LINKED_COVERAGE_GRID_ROWS,
        "core_grid_cols": LINKED_COVERAGE_GRID_COLS,
        "core_grid_counts": np.zeros((LINKED_COVERAGE_GRID_ROWS, LINKED_COVERAGE_GRID_COLS), dtype=np.int32).tolist(),
        "core_grid_occupied_ratio": 0.0,
        "edge_coverage": {name: 0.0 for name in ("top", "right", "bottom", "left")},
        "edge_segment_counts": {name: [] for name in ("top", "right", "bottom", "left")},
        "maximum_hole_distance_ratio": float("inf"),
    }
