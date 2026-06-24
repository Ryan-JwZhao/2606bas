from __future__ import annotations

import numpy as np

from bas.route_geometry import cue_alignment_start, estimate_route_end, rule_cue_separation_end


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
