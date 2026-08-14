from __future__ import annotations

import numpy as np

from bas.route_geometry import (
    cue_alignment_start,
    estimate_route_end,
    rule_cue_separation_end,
    segment_inside_polygon_to_pocket,
)


TABLE_POLYGON = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]], dtype=np.float32)


def test_cue_alignment_start_stays_inside_table() -> None:
    start = cue_alignment_start(
        np.array([120.0, 250.0], dtype=np.float32),
        np.array([465.0, 236.0], dtype=np.float32),
        TABLE_POLYGON,
        table_width_mm=1000.0,
        table_height_mm=500.0,
    )
    assert start[0] >= 0.0
    assert 0.0 <= start[1] <= 500.0
    assert start[0] < 120.0


def test_estimate_route_end_hits_table_boundary() -> None:
    end = estimate_route_end(
        np.array([500.0, 250.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        TABLE_POLYGON,
        fallback_length_mm=1200.0,
    )
    assert np.isclose(end[0], 1000.0, atol=1.0)
    assert np.isclose(end[1], 250.0, atol=1.0)


def test_rule_cue_separation_end_returns_projected_exit() -> None:
    end = rule_cue_separation_end(
        np.array([120.0, 250.0], dtype=np.float32),
        np.array([465.0, 236.0], dtype=np.float32),
        np.array([520.0, 220.0], dtype=np.float32),
        TABLE_POLYGON,
        fallback_length_mm=1200.0,
    )
    assert end is not None
    assert 0.0 <= float(end[0]) <= 1000.0
    assert 0.0 <= float(end[1]) <= 500.0


def test_all_six_pocket_routes_allow_external_fitted_centres() -> None:
    playable = np.asarray(
        [[31.0, 31.0], [2509.0, 31.0], [2509.0, 1239.0], [31.0, 1239.0]],
        dtype=np.float32,
    )
    table_center = np.asarray([1270.0, 635.0], dtype=np.float32)
    outside = 115.6
    corner_offset = outside / np.sqrt(2.0)
    pocket_centres = [
        np.asarray([31.0 - corner_offset, 31.0 - corner_offset], dtype=np.float32),
        np.asarray([1270.0, 31.0 - outside], dtype=np.float32),
        np.asarray([2509.0 + corner_offset, 31.0 - corner_offset], dtype=np.float32),
        np.asarray([2509.0 + corner_offset, 1239.0 + corner_offset], dtype=np.float32),
        np.asarray([1270.0, 1239.0 + outside], dtype=np.float32),
        np.asarray([31.0 - corner_offset, 1239.0 + corner_offset], dtype=np.float32),
    ]

    for pocket in pocket_centres:
        toward_table = table_center - pocket
        toward_table /= np.linalg.norm(toward_table)
        object_ball = pocket + toward_table * 288.0

        assert segment_inside_polygon_to_pocket(
            playable,
            object_ball,
            pocket,
            margin_mm=5.0,
            pocket_relief_mm=114.3,
        )
