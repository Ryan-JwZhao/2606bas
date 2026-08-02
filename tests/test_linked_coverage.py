from __future__ import annotations

import numpy as np

from bas.calibration.linked import LinkedCalibrationObservation, build_linked_patterns, solve_linked_projection_calibration
from bas.calibration.linked_coverage import (
    MIN_LINKED_TOTAL_MATCHED_POINTS,
    evaluate_linked_point_coverage,
    linked_coverage_errors,
)
from bas.geometry import TableGeometry


def _geometry() -> TableGeometry:
    return TableGeometry(
        outer_norm=np.asarray([[0.02, 0.03], [0.98, 0.03], [0.98, 0.97], [0.02, 0.97]], dtype=np.float32),
        inner_norm=np.asarray([[0.08, 0.08], [0.92, 0.08], [0.92, 0.92], [0.08, 0.92]], dtype=np.float32),
        pockets_norm=[],
    )


def test_dense_linked_layout_covers_every_core_cell_and_edge_segment() -> None:
    patterns = [
        pattern
        for pattern in build_linked_patterns(_geometry(), (1280, 800))
        if pattern.collect_for_solver
    ]
    points = np.vstack([pattern.projector_points for pattern in patterns])
    table_quad = np.asarray([[13.0, 8.0], [1267.0, 8.0], [1267.0, 792.0], [13.0, 792.0]], dtype=np.float32)

    report = evaluate_linked_point_coverage(points, table_quad)

    assert len(patterns) == 28
    assert sum(pattern.pattern_id.startswith("coverage_") for pattern in patterns) == 20
    assert report["core_grid_occupied_ratio"] == 1.0
    assert report["edge_coverage"]["top"] == 1.0
    assert report["edge_coverage"]["right"] == 1.0
    assert report["edge_coverage"]["bottom"] == 1.0
    assert report["edge_coverage"]["left"] == 1.0
    assert report["maximum_hole_distance_ratio"] <= 0.13
    assert linked_coverage_errors(report, matched_point_count=len(points)) == []


def test_linked_coverage_rejects_internal_holes_even_with_large_outer_hull() -> None:
    top = np.column_stack([np.linspace(0.02, 0.98, 40), np.full(40, 0.04)])
    right = np.column_stack([np.full(20, 0.96), np.linspace(0.04, 0.96, 20)])
    bottom = np.column_stack([np.linspace(0.98, 0.02, 40), np.full(40, 0.96)])
    left = np.column_stack([np.full(20, 0.04), np.linspace(0.96, 0.04, 20)])
    points = np.vstack([top, right, bottom, left]).astype(np.float32)
    quad = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    report = evaluate_linked_point_coverage(points, quad)
    errors = linked_coverage_errors(
        report,
        matched_point_count=max(MIN_LINKED_TOTAL_MATCHED_POINTS, len(points)),
    )

    assert report["width_ratio"] >= 0.9
    assert report["height_ratio"] >= 0.9
    assert report["hull_area_ratio"] >= 0.8
    assert report["core_grid_occupied_ratio"] < 1.0
    assert any("core grid" in error or "hole" in error for error in errors)


def test_linked_coverage_accepts_four_points_per_pattern_when_aggregate_is_dense() -> None:
    xs = np.linspace(0.04, 0.96, 10)
    ys = np.linspace(0.04, 0.96, 8)
    points = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float32)
    quad = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    report = evaluate_linked_point_coverage(points, quad)

    assert len(points) == MIN_LINKED_TOTAL_MATCHED_POINTS
    assert linked_coverage_errors(report, matched_point_count=len(points)) == []

    observations = []
    projector = points * np.asarray([1100.0, 650.0], dtype=np.float32) + np.asarray([30.0, 25.0], dtype=np.float32)
    camera = points * np.asarray([1500.0, 900.0], dtype=np.float32) + np.asarray([100.0, 60.0], dtype=np.float32)
    for index in range(0, len(points), 4):
        observations.append(
            LinkedCalibrationObservation(
                pattern_id=f"tile_{index // 4}",
                title=f"tile_{index // 4}",
                emphasis_zone=f"coverage_{index // 4}",
                camera_points=camera[index : index + 4],
                projector_points=projector[index : index + 4],
                ids=np.arange(4, dtype=np.int32),
                detected_count=4,
                matched_count=4,
            )
        )

    result = solve_linked_projection_calibration(
        observations,
        (1280, 720),
        table_polygon_cam=np.asarray(
            [[100.0, 60.0], [1600.0, 60.0], [1600.0, 960.0], [100.0, 960.0]],
            dtype=np.float32,
        ),
    )

    assert result.summary["patterns_used"] == 20
    assert result.summary["matched_points_total"] == MIN_LINKED_TOTAL_MATCHED_POINTS
