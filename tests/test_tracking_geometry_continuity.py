from __future__ import annotations

import numpy as np

from bas.config import TrackerConfig
from bas.schemas import Detection, DetectionsFrame
from bas.tracking import TemporalTracker
from bas.training.numbered_tracker import NumberedBallTracker


def _frame(frame_id: int, detection: Detection) -> DetectionsFrame:
    return DetectionsFrame(
        frame_id=frame_id,
        ts_cam_ns=frame_id * 33_000_000,
        detections=[detection],
    )


def _formal_detection(*, bbox, center) -> Detection:
    return Detection(
        bbox=bbox,
        conf=0.9,
        cls_id=2,
        cls_name="stb",
        refined_center_px=center,
        refined_radius_px=10.0,
        geometry_quality=0.85,
        geometry_method="appearance_ellipse",
    )


def _bbox_detection(*, bbox) -> Detection:
    return Detection(
        bbox=bbox,
        conf=0.9,
        cls_id=2,
        cls_name="stb",
        refined_center_px=(0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])),
        refined_radius_px=10.0,
        geometry_quality=0.45,
        geometry_method="bbox",
    )


def test_tracker_keeps_geometric_center_continuous_during_brief_bbox_fallback() -> None:
    tracker = TemporalTracker(TrackerConfig(match_distance_px=50))
    first = tracker.update(
        _frame(
            1,
            _formal_detection(bbox=(94.0, 90.0, 114.0, 110.0), center=(100.0, 100.0)),
        )
    ).tracks[0]
    fallback = tracker.update(
        _frame(2, _bbox_detection(bbox=(94.2, 90.0, 114.2, 110.0)))
    ).tracks[0]

    assert fallback.geometry_method == "bbox"
    assert np.linalg.norm(np.asarray(fallback.center_px) - np.asarray(first.center_px)) < 0.35


def test_bbox_fallback_still_follows_real_ball_motion_in_the_same_frame() -> None:
    tracker = TemporalTracker(TrackerConfig(match_distance_px=50))
    tracker.update(
        _frame(
            1,
            _formal_detection(bbox=(94.0, 90.0, 114.0, 110.0), center=(100.0, 100.0)),
        )
    )
    moved = tracker.update(
        _frame(2, _bbox_detection(bbox=(104.0, 90.0, 124.0, 110.0)))
    ).tracks[0]

    np.testing.assert_allclose(moved.center_px, (110.0, 100.0), atol=1e-5)


def test_bbox_only_track_uses_bbox_center_without_inventing_an_offset() -> None:
    tracker = TemporalTracker(TrackerConfig())
    track = tracker.update(
        _frame(1, _bbox_detection(bbox=(90.0, 80.0, 110.0, 100.0)))
    ).tracks[0]

    assert track.center_px == (100.0, 90.0)


def test_tracker_dampens_alternating_subpixel_formal_center_noise() -> None:
    tracker = TemporalTracker(TrackerConfig(match_distance_px=50))
    outputs = []
    for frame_id, center_x in enumerate((100.0, 100.7, 99.3), start=1):
        outputs.append(
            tracker.update(
                _frame(
                    frame_id,
                    _formal_detection(
                        bbox=(94.0, 90.0, 114.0, 110.0),
                        center=(center_x, 100.0),
                    ),
                )
            ).tracks[0]
        )

    steps = [
        np.linalg.norm(
            np.asarray(outputs[index].center_px) - np.asarray(outputs[index - 1].center_px)
        )
        for index in range(1, len(outputs))
    ]
    assert max(steps) < 0.35


def test_numbered_tracker_uses_the_same_center_continuity_contract() -> None:
    tracker = NumberedBallTracker(TrackerConfig(high_conf=0.4, low_conf=0.1))
    first = tracker.update(
        _frame(
            1,
            _formal_detection(bbox=(94.0, 90.0, 114.0, 110.0), center=(100.0, 100.0)),
        )
    ).tracks[0]
    fallback = tracker.update(
        _frame(2, _bbox_detection(bbox=(94.2, 90.0, 114.2, 110.0)))
    ).tracks[0]

    assert fallback.track_id == first.track_id == 2
    assert np.linalg.norm(np.asarray(fallback.center_px) - np.asarray(first.center_px)) < 0.35
