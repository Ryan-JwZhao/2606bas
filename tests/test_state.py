from __future__ import annotations

from bas.config import StateConfig
from bas.schemas import Event, MatchPhase, TrackObservation, TracksFrame
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


def _ball(track_id: int, group: str, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(x - 5, y - 5, x + 5, y + 5),
        center_px=(x, y),
        radius_px=5,
        cls_name=group,
        group=group,
        confidence=0.9,
        velocity_px_s=(vx, vy),
        center_mm=(x, y),
        velocity_mm_s=(vx, vy),
        radius_mm=28.0,
        quality=0.9,
    )


def _cue_stick(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(x1, y1, x2, y2),
        center_px=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        radius_px=max(abs(x2 - x1), abs(y2 - y1)) * 0.5,
        cls_name="cue_stick",
        group="cue_stick",
        confidence=0.9,
        velocity_px_s=(0.0, 0.0),
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


def test_state_emits_ball_collision_candidate() -> None:
    sm = MatchStateMachine(StateConfig(stable_frames=2, settle_frames=2))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[], ball_diameter_mm=56)

    out = sm.update(
        TracksFrame(
            frame_id=1,
            ts_cam_ns=1,
            tracks=[_ball(1, "cue", 100, 100, 35, 0), _ball(2, "solid", 158, 100, 0, 0)],
        )
    )

    assert any(event.name == "BALL_COLLISION_CANDIDATE" for event in out.events)


def test_state_emits_rail_collision_candidate() -> None:
    sm = MatchStateMachine(StateConfig(stable_frames=2, settle_frames=2))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[], ball_diameter_mm=56)

    out = sm.update(TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[_ball(1, "cue", 25, 250, 40, 0)]))

    assert any(event.name == "RAIL_COLLISION_CANDIDATE" for event in out.events)


def test_state_emits_probable_pot_on_disappearance_near_pocket() -> None:
    sm = MatchStateMachine(StateConfig(stable_frames=2, settle_frames=2))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[_ball(1, "solid", 35, 35, -20, -20)]))
    sm.phase = MatchPhase.SHOT_ACTIVE

    out = sm.update(TracksFrame(frame_id=6, ts_cam_ns=6, tracks=[]))

    assert any(event.name == "POT_PROBABLE" for event in out.events)


def test_state_emits_shot_start_vote_with_two_conditions() -> None:
    sm = MatchStateMachine(StateConfig(stable_frames=2, settle_frames=2))
    out = sm.update(
        TracksFrame(
            frame_id=1,
            ts_cam_ns=1_000_000_000,
            tracks=[_ball(1, "cue", 100, 100, 40, 0), _cue_stick(2, 40, 95, 90, 105)],
        )
    )

    assert any(event.name == "SHOT_START_VOTED" for event in out.events)
    assert out.phase == "SHOT_ACTIVE"


def test_low_quality_duplicate_cue_does_not_trigger_anomaly() -> None:
    sm = MatchStateMachine(StateConfig(anomaly_frames=1, stable_frames=2, settle_frames=2))
    out = sm.update(
        TracksFrame(
            frame_id=1,
            ts_cam_ns=1,
            tracks=[
                _ball(1, "cue", 100, 100),
                TrackObservation(
                    track_id=2,
                    bbox=(100, 100, 110, 110),
                    center_px=(105, 105),
                    radius_px=5,
                    cls_name="cue",
                    group="cue",
                    confidence=0.4,
                    velocity_px_s=(0.0, 0.0),
                    center_mm=(105, 105),
                    velocity_mm_s=(0.0, 0.0),
                    radius_mm=28.0,
                    quality=0.1,
                ),
            ],
        )
    )

    assert out.phase != "ANOMALY_RECOVERY"


def test_turn_target_group_becomes_black_after_last_solid_is_potted() -> None:
    sm = MatchStateMachine(StateConfig())
    sm._turn_target_group = "solid"
    frame = TracksFrame(
        frame_id=10,
        ts_cam_ns=10,
        tracks=[
            _ball(1, "cue", 100, 100),
            _ball(2, "stripe", 300, 100),
            _ball(3, "black", 500, 100),
        ],
    )

    sm._resolve_turn_target_group(
        frame,
        [Event("POT_PROBABLE", 10, 10, payload={"group": "solid"}), Event("TURN_RESOLVE", 10, 10)],
    )

    assert sm._turn_target_group == "black"


def test_turn_target_group_stays_on_stripe_when_solids_are_already_cleared() -> None:
    sm = MatchStateMachine(StateConfig())
    sm._turn_target_group = "stripe"
    frame = TracksFrame(
        frame_id=10,
        ts_cam_ns=10,
        tracks=[
            _ball(1, "cue", 100, 100),
            _ball(2, "stripe", 300, 100),
            _ball(3, "black", 500, 100),
        ],
    )

    sm._resolve_turn_target_group(
        frame,
        [Event("POT_PROBABLE", 10, 10, payload={"group": "stripe"}), Event("TURN_RESOLVE", 10, 10)],
    )

    assert sm._turn_target_group == "stripe"


def test_turn_target_group_stays_black_after_both_object_groups_clear() -> None:
    sm = MatchStateMachine(StateConfig())
    sm._turn_target_group = "stripe"
    frame = TracksFrame(
        frame_id=10,
        ts_cam_ns=10,
        tracks=[
            _ball(1, "cue", 100, 100),
            _ball(2, "black", 500, 100),
        ],
    )

    sm._resolve_turn_target_group(
        frame,
        [Event("POT_PROBABLE", 10, 10, payload={"group": "stripe"}), Event("TURN_RESOLVE", 10, 10)],
    )

    assert sm._turn_target_group == "black"
