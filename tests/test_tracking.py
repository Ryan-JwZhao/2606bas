from __future__ import annotations

from bas.config import TrackerConfig
from bas.schemas import Detection, DetectionsFrame
from bas.tracking import TemporalTracker


def test_tracker_propagates_ball_geometry_provenance() -> None:
    tracker = TemporalTracker(TrackerConfig())
    frame = DetectionsFrame(
        frame_id=1,
        ts_cam_ns=1_000_000_000,
        detections=[
            Detection(
                bbox=(10, 10, 30, 30),
                conf=0.9,
                cls_id=0,
                cls_name="cue",
                geometry_quality=0.42,
                geometry_method="bbox",
            )
        ],
    )

    track = tracker.update(frame).tracks[0]

    assert track.geometry_quality == 0.42
    assert track.geometry_method == "bbox"


def test_tracker_keeps_id_across_small_motion() -> None:
    tracker = TemporalTracker(TrackerConfig(match_distance_px=50))
    f1 = DetectionsFrame(
        frame_id=1,
        ts_cam_ns=1_000_000_000,
        detections=[Detection(bbox=(10, 10, 30, 30), conf=0.9, cls_id=0, cls_name="cue")],
    )
    f2 = DetectionsFrame(
        frame_id=2,
        ts_cam_ns=1_033_000_000,
        detections=[Detection(bbox=(14, 12, 34, 32), conf=0.85, cls_id=0, cls_name="cue")],
    )
    t1 = tracker.update(f1)
    t2 = tracker.update(f2)
    assert len(t1.tracks) == 1
    assert len(t2.tracks) == 1
    assert t1.tracks[0].track_id == t2.tracks[0].track_id
    assert t2.tracks[0].velocity_px_s[0] > 0


def test_tracker_keeps_occluded_track_briefly() -> None:
    tracker = TemporalTracker(TrackerConfig(max_lost_frames=3))
    f1 = DetectionsFrame(
        frame_id=1,
        ts_cam_ns=1_000_000_000,
        detections=[Detection(bbox=(10, 10, 30, 30), conf=0.9, cls_id=0, cls_name="cue")],
    )
    f2 = DetectionsFrame(frame_id=2, ts_cam_ns=1_033_000_000, detections=[])
    tracker.update(f1)
    out = tracker.update(f2)
    assert len(out.tracks) == 1
    assert out.tracks[0].visibility == "occluded"


def test_tracker_keeps_id_for_fast_motion_with_adaptive_gate() -> None:
    tracker = TemporalTracker(TrackerConfig(match_distance_px=20))
    f1 = DetectionsFrame(
        frame_id=1,
        ts_cam_ns=1_000_000_000,
        detections=[Detection(bbox=(10, 10, 30, 30), conf=0.9, cls_id=0, cls_name="cue")],
    )
    f2 = DetectionsFrame(
        frame_id=2,
        ts_cam_ns=1_100_000_000,
        detections=[Detection(bbox=(30, 10, 50, 30), conf=0.9, cls_id=0, cls_name="cue")],
    )
    f3 = DetectionsFrame(
        frame_id=3,
        ts_cam_ns=1_200_000_000,
        detections=[Detection(bbox=(80, 10, 100, 30), conf=0.9, cls_id=0, cls_name="cue")],
    )

    t1 = tracker.update(f1)
    t2 = tracker.update(f2)
    t3 = tracker.update(f3)

    assert len(t1.tracks) == 1
    assert len(t2.tracks) == 1
    assert len(t3.tracks) == 1
    assert t1.tracks[0].track_id == t2.tracks[0].track_id == t3.tracks[0].track_id


def test_tracker_prunes_overlapping_duplicate_ball_detections() -> None:
    tracker = TemporalTracker(TrackerConfig())
    frame = DetectionsFrame(
        frame_id=1,
        ts_cam_ns=1_000_000_000,
        detections=[
            Detection(bbox=(10, 10, 30, 30), conf=0.92, cls_id=0, cls_name="cue"),
            Detection(bbox=(11, 11, 31, 31), conf=0.91, cls_id=0, cls_name="cue"),
        ],
    )

    out = tracker.update(frame)

    assert len(out.tracks) == 1


def test_tracker_dampens_small_stationary_center_jitter() -> None:
    tracker = TemporalTracker(TrackerConfig(match_distance_px=50))
    tracker.update(
        DetectionsFrame(
            frame_id=1,
            ts_cam_ns=1_000_000_000,
            detections=[
                Detection(
                    bbox=(10, 10, 30, 30),
                    conf=0.9,
                    cls_id=0,
                    cls_name="cue",
                    refined_center_px=(20.0, 20.0),
                    refined_radius_px=10.0,
                    geometry_quality=0.9,
                )
            ],
        )
    )
    out = tracker.update(
        DetectionsFrame(
            frame_id=2,
            ts_cam_ns=1_033_000_000,
            detections=[
                Detection(
                    bbox=(12, 10, 32, 30),
                    conf=0.9,
                    cls_id=0,
                    cls_name="cue",
                    refined_center_px=(22.0, 20.0),
                    refined_radius_px=10.0,
                    geometry_quality=0.9,
                )
            ],
        )
    )

    assert 20.0 < out.tracks[0].center_px[0] < 22.0


def test_tracker_confirms_a_new_track_only_after_two_real_hits() -> None:
    tracker = TemporalTracker(TrackerConfig(min_confirmed_hits=2))

    first = tracker.update(
        DetectionsFrame(
            frame_id=1,
            ts_cam_ns=1_000_000_000,
            detections=[Detection(bbox=(10, 10, 30, 30), conf=0.73, cls_id=1, cls_name="solid")],
        )
    )
    second = tracker.update(
        DetectionsFrame(
            frame_id=2,
            ts_cam_ns=1_066_000_000,
            detections=[Detection(bbox=(10, 10, 30, 30), conf=0.73, cls_id=1, cls_name="solid")],
        )
    )

    assert first.tracks[0].confirmed is False
    assert second.tracks[0].confirmed is True


def test_tracker_does_not_confirm_two_hits_separated_by_a_real_miss() -> None:
    tracker = TemporalTracker(TrackerConfig(min_confirmed_hits=2, max_lost_frames=3))
    detection = Detection(bbox=(10, 10, 30, 30), conf=0.73, cls_id=1, cls_name="solid")

    tracker.update(DetectionsFrame(frame_id=1, ts_cam_ns=1_000_000_000, detections=[detection]))
    tracker.update(DetectionsFrame(frame_id=2, ts_cam_ns=1_066_000_000, detections=[]))
    reacquired = tracker.update(
        DetectionsFrame(frame_id=3, ts_cam_ns=1_132_000_000, detections=[detection])
    )

    assert reacquired.tracks[0].confirmed is False
