from __future__ import annotations

import numpy as np

from bas.table_boundaries import EdgeInsets, derive_table_boundaries


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
