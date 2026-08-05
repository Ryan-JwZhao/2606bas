from __future__ import annotations

import numpy as np

from bas.geometry import TableGeometry
from bas.perception.pocket_observer import _point_in_entry_gate
from bas.perception.regions import DetectionRegionPolicy, build_detection_region_policy, filter_detections_by_region
from bas.schemas import Detection


def _square(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _detection(cls_name: str, cx: float, cy: float, radius: float = 1.0) -> Detection:
    return Detection(
        bbox=(cx - radius, cy - radius, cx + radius, cy + radius),
        conf=0.9,
        cls_id=0,
        cls_name=cls_name,
    )


def test_build_detection_region_policy_uses_inner_for_balls_and_outer_for_cue_sticks() -> None:
    geometry = TableGeometry(
        outer_norm=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        inner_norm=np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], dtype=np.float32),
    )

    policy = build_detection_region_policy((100, 200, 3), geometry)

    np.testing.assert_allclose(policy.global_polygon, _square(0.0, 0.0, 200.0, 100.0))
    np.testing.assert_allclose(policy.ball_polygon, _square(40.0, 20.0, 160.0, 80.0))
    np.testing.assert_allclose(policy.cue_stick_polygon, _square(0.0, 0.0, 200.0, 100.0))


def test_filter_detections_by_region_keeps_balls_inside_inner_and_cue_sticks_inside_outline() -> None:
    outer = _square(0.0, 0.0, 10.0, 10.0)
    inner = _square(2.0, 2.0, 8.0, 8.0)
    policy = DetectionRegionPolicy(
        global_polygon=outer,
        ball_polygon=inner,
        cue_stick_polygon=outer,
    )

    detections = [
        _detection("cue", 1.0, 1.0),
        _detection("solid", 5.0, 5.0),
        _detection("cue_stick", 1.0, 1.0),
        _detection("cue_stick", 12.0, 12.0),
    ]

    filtered = filter_detections_by_region(detections, policy)

    assert [det.cls_name for det in filtered] == ["solid", "cue_stick"]


def test_pocket_guard_band_is_observation_only_and_does_not_admit_balls_outside_inner() -> None:
    geometry = TableGeometry(
        outer_norm=_square(0.0, 0.0, 1.0, 1.0),
        inner_norm=_square(0.2, 0.2, 0.8, 0.8),
        pockets_norm=[np.array([[0.44, 0.20], [0.50, 0.12], [0.56, 0.20]], dtype=np.float32)],
    )

    policy = build_detection_region_policy(
        (100, 200, 3),
        geometry,
        ball_diameter_px_by_pocket=[12.0],
    )
    detections = [
        _detection("solid", 100.0, 15.0),
        _detection("cue", 10.0, 10.0),
    ]

    filtered = filter_detections_by_region(detections, policy)

    assert len(policy.ball_guard_regions) == 1
    assert filtered == []


def test_corner_pocket_guard_uses_arc_center_and_rejects_recorded_table_side_false_point() -> None:
    # 0804 right-top pocket jaw curve.  Its point mean lies roughly one ball
    # radius inside the table, while the fitted arc center is the physical
    # pocket center.  The recorded 2026-08-05 false WB point must remain on the
    # table side of the entry gate.
    curve_px = np.array(
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
    geometry = TableGeometry(
        pockets_norm=[curve_px / np.array([1920.0, 1080.0], dtype=np.float32)],
    )

    policy = build_detection_region_policy(
        (1080, 1920, 3),
        geometry,
        ball_diameter_px_by_pocket=[48.2],
    )

    guard = policy.ball_guard_regions[0]
    np.testing.assert_allclose(guard.center_px, (1902.2, 43.1), atol=1.0)
    assert not _point_in_entry_gate((1859.4, 161.4), guard, (960.0, 540.0))


def test_center_playable_polygon_does_not_discard_ball_inside_inline_boundary() -> None:
    geometry = TableGeometry(
        outer_norm=_square(0.0, 0.0, 1.0, 1.0),
        inner_norm=_square(0.1, 0.1, 0.9, 0.9),
    )
    policy = build_detection_region_policy(
        (100, 100, 3),
        geometry,
    )
    detections = [
        _detection("solid", 15.0, 50.0),
        _detection("stripe", 50.0, 50.0),
    ]

    filtered = filter_detections_by_region(detections, policy)

    assert [(det.cls_name, det.center) for det in filtered] == [
        ("solid", (15.0, 50.0)),
        ("stripe", (50.0, 50.0)),
    ]


def test_disabled_region_policy_rejects_every_detection() -> None:
    detections = [
        _detection("solid", 50.0, 50.0),
        _detection("cue_stick", 50.0, 50.0),
    ]

    filtered = filter_detections_by_region(
        detections,
        DetectionRegionPolicy(detection_enabled=False),
    )

    assert filtered == []
