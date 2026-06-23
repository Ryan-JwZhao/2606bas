from __future__ import annotations

from bas.config import StateConfig
from bas.schemas import MatchPhase, TrackObservation, TracksFrame
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


def test_operator_can_force_and_hold_state() -> None:
    sm = MatchStateMachine(StateConfig(stable_frames=2, settle_frames=2))
    first = TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[_track(0.0, 1)])
    sm.update(first)
    sm.snapshot_layout(first)
    sm.force_phase(MatchPhase.TURN_RESOLVE)
    sm.set_operator_hold(True)

    out = sm.update(TracksFrame(frame_id=2, ts_cam_ns=2, tracks=[_track(100.0, 2)]))

    assert out.phase == "TURN_RESOLVE"
    assert out.state_version.endswith("+operator_hold")
    assert [event.name for event in out.events] == [
        "OPERATOR_SNAPSHOT_LAYOUT",
        "OPERATOR_FORCE_PHASE",
        "OPERATOR_HOLD_ENABLED",
    ]
    assert out.layout[0].center_px == first.tracks[0].center_px


def test_operator_force_phase_survives_next_update() -> None:
    sm = MatchStateMachine(StateConfig(stable_frames=2, settle_frames=2))
    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[_track(0.0, 1)]))
    sm.force_phase(MatchPhase.TURN_RESOLVE)

    out = sm.update(TracksFrame(frame_id=2, ts_cam_ns=2, tracks=[_track(0.0, 2)]))

    assert out.phase == "TURN_RESOLVE"
    assert out.state_version.endswith("+operator_override")
    assert any(event.name == "OPERATOR_FORCE_PHASE" for event in out.events)
