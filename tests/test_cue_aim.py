from __future__ import annotations

import numpy as np

from bas.planning.cue_aim import CueStickAimDetector
from bas.schemas import TrackObservation


def _stick(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(x1, y1, x2, y2),
        center_px=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        radius_px=0.25 * ((x2 - x1) + (y2 - y1)),
        cls_name="cue_stick",
        group="cue_stick",
        confidence=0.95,
        quality=0.95,
    )


def _detect(track: TrackObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detector = CueStickAimDetector()
    aim = detector.detect(
        frame_bgr=None,
        tracks=[track],
        cue_center_px=np.asarray([500.0, 250.0], dtype=np.float32),
        cue_radius_px=15.0,
        inner_polygon_px=None,
        min_stick_quality=0.0,
    )
    assert aim is not None
    return aim.tip_px, aim.tail_px, aim.direction_px


def test_track_fallback_points_forward_away_from_body_on_left_side() -> None:
    tip, tail, direction = _detect(_stick(1, 360.0, 245.0, 430.0, 255.0))

    assert direction[0] > 0.99
    assert abs(float(direction[1])) < 1.0e-3
    assert float(tip[0]) > float(tail[0])


def test_track_fallback_points_forward_away_from_body_on_right_side() -> None:
    tip, tail, direction = _detect(_stick(1, 560.0, 245.0, 630.0, 255.0))

    assert direction[0] < -0.99
    assert abs(float(direction[1])) < 1.0e-3
    assert float(tip[0]) < float(tail[0])


def test_track_fallback_uses_body_majority_when_segment_crosses_cue_ball() -> None:
    tip, _tail, direction = _detect(_stick(1, 450.0, 245.0, 620.0, 255.0))

    assert direction[0] < -0.99
    assert 500.0 <= float(tip[0]) <= 520.0
