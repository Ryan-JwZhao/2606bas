from __future__ import annotations

from bas.config import TrackerConfig
from bas.schemas import Detection, DetectionsFrame
from bas.tracking import TemporalTracker


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

