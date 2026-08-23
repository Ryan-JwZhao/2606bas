from __future__ import annotations

import cv2
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


def test_prefer_tracks_skips_edge_detection_when_track_is_available(monkeypatch) -> None:
    detector = CueStickAimDetector()

    def _fail_edges(*_args, **_kwargs):
        raise AssertionError("edge detector should not run when prefer_tracks succeeds")

    monkeypatch.setattr(detector, "_detect_from_edges", _fail_edges)

    aim = detector.detect(
        frame_bgr=np.zeros((600, 800, 3), dtype=np.uint8),
        tracks=[_stick(1, 360.0, 245.0, 430.0, 255.0)],
        cue_center_px=np.asarray([500.0, 250.0], dtype=np.float32),
        cue_radius_px=15.0,
        inner_polygon_px=None,
        min_stick_quality=0.0,
        prefer_tracks=True,
    )

    assert aim is not None


def test_frame_edges_stay_associated_with_detected_cue_stick() -> None:
    frame = np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.line(frame, (70, 300), (275, 300), (255, 255, 255), 6)
    cv2.line(frame, (300, 40), (300, 280), (255, 255, 255), 6)

    aim = CueStickAimDetector().detect(
        frame_bgr=frame,
        tracks=[_stick(1, 70.0, 294.0, 275.0, 306.0)],
        cue_center_px=np.asarray([300.0, 300.0], dtype=np.float32),
        cue_radius_px=15.0,
        min_stick_quality=0.0,
    )

    assert aim is not None
    assert float(aim.direction_px[0]) > 0.95
    assert abs(float(aim.direction_px[1])) < 0.10


def test_frame_edges_reject_line_that_misses_cue_ball_center() -> None:
    frame = np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.line(frame, (80, 340), (280, 340), (255, 255, 255), 6)

    aim = CueStickAimDetector().detect(
        frame_bgr=frame,
        tracks=[],
        cue_center_px=np.asarray([300.0, 300.0], dtype=np.float32),
        cue_radius_px=15.0,
        min_stick_quality=0.0,
    )

    assert aim is None


def test_track_fallback_uses_oriented_axis_when_available() -> None:
    track = _stick(1, 400.0, 400.0, 460.0, 460.0)
    track.axis_endpoints_px = ((400.0, 400.0), (460.0, 460.0))
    track.axis_quality = 0.95

    aim = CueStickAimDetector().detect(
        frame_bgr=None,
        tracks=[track],
        cue_center_px=np.asarray([500.0, 500.0], dtype=np.float32),
        cue_radius_px=15.0,
        min_stick_quality=0.0,
    )

    assert aim is not None
    expected = np.asarray([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
    np.testing.assert_allclose(aim.direction_px, expected, atol=1.0e-3)
