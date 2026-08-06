from __future__ import annotations

import numpy as np

from bas.geometry import pocket_arc_center
from bas.table_boundaries import EdgeInsets, derive_table_boundaries
from bas.table_boundaries import apply_edge_insets


def test_derive_table_boundaries_applies_separate_visible_physical_and_center_offsets() -> None:
    visible_polygon = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float32)
    pocket_curve = np.array([[470, 0], [500, 0], [530, 0]], dtype=np.float32)

    boundaries = derive_table_boundaries(
        visible_polygon,
        [pocket_curve],
        table_width_mm=1000.0,
        table_height_mm=500.0,
        ball_diameter_mm=57.15,
        projection_visible_insets=EdgeInsets(bottom_mm=12.0),
        physical_rail_insets=EdgeInsets.uniform(10.0),
        physical_middle_pocket_relief_top_mm=0.0,
        physical_middle_pocket_relief_bottom_mm=0.0,
        center_reachable_extra_margin_mm=2.0,
    )

    visible_bbox = (
        float(np.min(boundaries.projection_visible_polygon_mm[:, 0])),
        float(np.min(boundaries.projection_visible_polygon_mm[:, 1])),
        float(np.max(boundaries.projection_visible_polygon_mm[:, 0])),
        float(np.max(boundaries.projection_visible_polygon_mm[:, 1])),
    )
    physical_bbox = (
        float(np.min(boundaries.physical_rail_polygon_mm[:, 0])),
        float(np.min(boundaries.physical_rail_polygon_mm[:, 1])),
        float(np.max(boundaries.physical_rail_polygon_mm[:, 0])),
        float(np.max(boundaries.physical_rail_polygon_mm[:, 1])),
    )
    center_bbox = (
        float(np.min(boundaries.center_playable_polygon_mm[:, 0])),
        float(np.min(boundaries.center_playable_polygon_mm[:, 1])),
        float(np.max(boundaries.center_playable_polygon_mm[:, 0])),
        float(np.max(boundaries.center_playable_polygon_mm[:, 1])),
    )

    assert visible_bbox[1] <= 1.0
    assert visible_bbox[3] < 500.0
    assert physical_bbox[0] > 0.0
    assert physical_bbox[1] > 0.0
    assert physical_bbox[2] < 1000.0
    assert physical_bbox[3] < 500.0
    assert center_bbox[0] > physical_bbox[0]
    assert center_bbox[1] > physical_bbox[1]
    assert center_bbox[2] < physical_bbox[2]
    assert center_bbox[3] < physical_bbox[3]
    assert boundaries.physical_pocket_points_mm[0][1] > boundaries.projection_visible_pocket_points_mm[0][1]


def test_apply_edge_insets_keeps_straight_rail_segments_straight() -> None:
    top_rail = np.array([[120, 0], [320, 0], [520, 0], [720, 0], [920, 0]], dtype=np.float32)

    adjusted = apply_edge_insets(
        top_rail,
        width_mm=1000.0,
        height_mm=500.0,
        insets=EdgeInsets(top_mm=14.0, right_mm=8.0, bottom_mm=12.0, left_mm=8.0),
    )

    np.testing.assert_allclose(adjusted[:, 0], top_rail[:, 0], atol=1e-5)
    np.testing.assert_allclose(adjusted[:, 1], np.full((top_rail.shape[0],), 14.0, dtype=np.float32), atol=1e-5)


def test_right_top_pocket_center_uses_annotated_jaw_arc_center() -> None:
    # 0804 right-top pocket annotation: the point mean is on the table side,
    # while the fitted circle center is the physical pocket center.
    pocket_curve = np.array(
        [
            [1847.7, 57.6],
            [1849.6, 63.7],
            [1853.2, 70.2],
            [1856.3, 75.2],
            [1860.4, 79.6],
            [1865.5, 85.1],
            [1871.6418, 90.4592],
            [1878.5304, 94.4776],
            [1886.3375, 97.5775],
            [1894.8335, 98.8404],
            [1900.9185, 98.7256],
        ],
        dtype=np.float32,
    )

    boundaries = derive_table_boundaries(
        np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.float32),
        [pocket_curve],
        table_width_mm=1920.0,
        table_height_mm=1080.0,
        ball_diameter_mm=48.2,
        projection_visible_insets=EdgeInsets(),
        physical_rail_insets=EdgeInsets(),
        physical_middle_pocket_relief_top_mm=0.0,
        physical_middle_pocket_relief_bottom_mm=0.0,
        center_reachable_extra_margin_mm=0.0,
    )

    np.testing.assert_allclose(boundaries.physical_pocket_points_mm[0], (1902.2, 43.1), atol=1.0)


def test_planning_pocket_center_is_fitted_before_rail_inset_or_table_clipping() -> None:
    # A corner-pocket jaw arc legitimately extends outside the nominal playing
    # rectangle.  Rail insets may move rail polygons, but must never deform the
    # arc that supplies the shot target.
    angles = np.deg2rad(np.linspace(120.0, 210.0, 13))
    expected_center = np.asarray([1070.0, -20.0], dtype=np.float32)
    pocket_curve = expected_center + 80.0 * np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    raw_fitted_center = pocket_arc_center(pocket_curve, expected_diameter=57.15)

    boundaries = derive_table_boundaries(
        np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float32),
        [pocket_curve],
        table_width_mm=1000.0,
        table_height_mm=500.0,
        ball_diameter_mm=57.15,
        projection_visible_insets=EdgeInsets(),
        physical_rail_insets=EdgeInsets.uniform(10.0),
        physical_middle_pocket_relief_top_mm=0.0,
        physical_middle_pocket_relief_bottom_mm=0.0,
        center_reachable_extra_margin_mm=0.0,
    )

    np.testing.assert_allclose(raw_fitted_center, expected_center, atol=0.1)
    np.testing.assert_allclose(boundaries.planning_pocket_points_mm[0], raw_fitted_center, atol=0.1)


def test_middle_pocket_relief_brings_physical_boundary_closer_to_visible() -> None:
    visible_polygon = np.array(
        [[0, 0], [430, 0], [470, 20], [500, 28], [530, 20], [570, 0], [1000, 0], [1000, 500], [0, 500]],
        dtype=np.float32,
    )
    top_middle_pocket = np.array([[470, 20], [500, 28], [530, 20]], dtype=np.float32)

    without_relief = derive_table_boundaries(
        visible_polygon,
        [top_middle_pocket],
        table_width_mm=1000.0,
        table_height_mm=500.0,
        ball_diameter_mm=57.15,
        projection_visible_insets=EdgeInsets(),
        physical_rail_insets=EdgeInsets(top_mm=10.0),
        physical_middle_pocket_relief_top_mm=0.0,
        physical_middle_pocket_relief_bottom_mm=0.0,
        center_reachable_extra_margin_mm=2.0,
    )
    with_relief = derive_table_boundaries(
        visible_polygon,
        [top_middle_pocket],
        table_width_mm=1000.0,
        table_height_mm=500.0,
        ball_diameter_mm=57.15,
        projection_visible_insets=EdgeInsets(),
        physical_rail_insets=EdgeInsets(top_mm=10.0),
        physical_middle_pocket_relief_top_mm=8.0,
        physical_middle_pocket_relief_bottom_mm=0.0,
        center_reachable_extra_margin_mm=2.0,
    )

    mid_idx = int(np.argmin(np.abs(visible_polygon[:, 0] - 500.0)))
    assert with_relief.physical_rail_polygon_mm[mid_idx, 1] < without_relief.physical_rail_polygon_mm[mid_idx, 1]
