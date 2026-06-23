from __future__ import annotations

from bas.config import StateConfig
from bas.schemas import TrackObservation, TracksFrame
from bas.state import MatchStateMachine


def _track(speed: float, frame: int) -> TrackObservation:
    return TrackObservation(
        track_id=1,
        bbox=(0, 0, 10, 10),
        center_px=(10 + frame * 2, 10),
        radius_px=5,
        cls_name="cue",
        group="cue",
        confidence=0.9,
        velocity_px_s=(speed, 0.0),
        quality=0.9,
    )


def test_state_enters_shot_active_on_motion() -> None:
    sm = MatchStateMachine(StateConfig(stable_frames=2, settle_frames=2))
    out = sm.update(TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[_track(100.0, 1)]))
    assert out.phase == "SHOT_ACTIVE"
    assert any(e.name == "SHOT_STARTED" for e in out.events)


def test_state_settles_after_stable_frames() -> None:
    sm = MatchStateMachine(StateConfig(stable_frames=2, settle_frames=2))
    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[_track(100.0, 1)]))
    sm.update(TracksFrame(frame_id=2, ts_cam_ns=2, tracks=[_track(0.0, 2)]))
    out = sm.update(TracksFrame(frame_id=3, ts_cam_ns=3, tracks=[_track(0.0, 3)]))
    assert out.phase in {"SETTLING", "TURN_RESOLVE"}

