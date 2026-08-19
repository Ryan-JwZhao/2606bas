from __future__ import annotations

from bas.config import StateConfig
from bas.schemas import (
    MatchPhase,
    PocketVisualObservation,
    PocketVisualObservationFrame,
    TrackObservation,
    TracksFrame,
)
from bas.state import ModernMatchStateMachine
from bas.state.pocket import PerBallPocketFSM


def _ball(track_id: int, group: str, x: float, y: float) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(x - 5, y - 5, x + 5, y + 5),
        center_px=(x, y),
        radius_px=5,
        cls_name=group,
        group=group,
        confidence=0.9,
        center_mm=(x, y),
        velocity_mm_s=(0.0, -300.0),
        radius_mm=28.0,
        quality=0.9,
    )


def _visual(
    frame_id: int,
    ts_ns: int,
    *,
    group: str = "solid",
    inward: bool = False,
    outward: bool = False,
    lip: bool = False,
    clear: bool = False,
    track_ids: list[int] | None = None,
    motion_score: float = 0.0,
    foreground_score: float = 0.0,
    foreground_depth_diameters: float | None = None,
) -> PocketVisualObservationFrame:
    return PocketVisualObservationFrame(
        frame_id=frame_id,
        ts_cam_ns=ts_ns,
        observations=[
            PocketVisualObservation(
                pocket_index=0,
                inward_crossing=inward,
                outward_crossing=outward,
                lip_occupied=lip,
                clear=clear,
                group=group,
                confidence=0.9,
                associated_track_ids=list(track_ids or []),
                evidence_sources=["frame_difference", "ball_sized_motion", "foreground_motion"],
                motion_score=motion_score,
                foreground_score=foreground_score,
                foreground_depth_diameters=foreground_depth_diameters,
            )
        ],
        latency_ms=1.0,
    )


def _machine() -> ModernMatchStateMachine:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_visual_confirmation_ms=1300))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(500, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE
    return sm


def test_visual_crossing_confirms_once_after_1_3_seconds() -> None:
    sm = _machine()
    sm.update(TracksFrame(1, 1_000_000_000, [_ball(2, "solid", 500, 170)]), _visual(1, 1_000_000_000, clear=True))
    candidate = sm.update(TracksFrame(2, 1_100_000_000, []), _visual(2, 1_100_000_000, inward=True, track_ids=[2]))
    early = sm.update(TracksFrame(3, 2_399_000_000, []), _visual(3, 2_399_000_000, clear=True, track_ids=[2]))
    detected = sm.update(TracksFrame(4, 2_400_000_000, []), _visual(4, 2_400_000_000, clear=True, track_ids=[2]))
    duplicate = sm.update(TracksFrame(5, 2_800_000_000, []), _visual(5, 2_800_000_000, clear=True, track_ids=[2]))

    assert any(event.name == "POCKET_CANDIDATE" for event in candidate.events)
    assert not any(event.name == "POCKET_DETECTED" for event in early.events)
    event = next(event for event in detected.events if event.name == "POCKET_DETECTED")
    assert event.payload["group"] == "solid"
    assert event.payload["decision_latency_ms"] == 1300
    assert "pocket_visual_inward" in event.payload["evidence_sources"]
    assert not any(event.name == "POCKET_DETECTED" for event in duplicate.events)


def test_visual_outward_crossing_rejects_bounce() -> None:
    sm = _machine()
    sm.update(TracksFrame(1, 1_000_000_000, [_ball(2, "stripe", 500, 100)]), _visual(1, 1_000_000_000, group="stripe", clear=True))
    sm.update(TracksFrame(2, 1_100_000_000, []), _visual(2, 1_100_000_000, group="stripe", inward=True, track_ids=[2]))
    bounced = sm.update(TracksFrame(3, 1_250_000_000, []), _visual(3, 1_250_000_000, group="stripe", outward=True, track_ids=[2]))

    rejection = next(event for event in bounced.events if event.name == "POCKET_REJECTED")
    assert "visual_outward_crossing" in rejection.payload["reason_codes"]


def test_persistent_lip_occupancy_vetoes_cue_ball_false_goal() -> None:
    sm = _machine()
    sm.update(TracksFrame(1, 1_000_000_000, [_ball(1, "cue", 500, 70)]), _visual(1, 1_000_000_000, group="cue", inward=True, track_ids=[1]))
    sm.update(TracksFrame(2, 1_500_000_000, []), _visual(2, 1_500_000_000, group="cue", lip=True, track_ids=[1]))
    rejected = sm.update(TracksFrame(3, 2_700_000_000, []), _visual(3, 2_700_000_000, group="cue", lip=True, track_ids=[1]))

    assert any(event.name == "POCKET_REJECTED" for event in rejected.events)
    assert not any(event.name == "POCKET_DETECTED" for event in rejected.events)


def test_visible_ball_cannot_be_announced_before_lip_veto_arrives() -> None:
    """Regression: shot 3 announced bb with missing_ms=0, then rejected the same decision."""

    sm = _machine()
    black = _ball(55, "black", 500, 15)
    sm.update(
        TracksFrame(1, 1_000_000_000, [black]),
        _visual(1, 1_000_000_000, group="black", inward=True, track_ids=[55]),
    )
    premature = sm.update(
        TracksFrame(2, 2_400_000_000, [black]),
        _visual(2, 2_400_000_000, group="black", clear=True, track_ids=[55]),
    )
    sm.update(
        TracksFrame(3, 2_500_000_000, [black]),
        _visual(3, 2_500_000_000, group="black", lip=True, track_ids=[55]),
    )
    rejected = sm.update(
        TracksFrame(4, 3_700_000_000, [black]),
        _visual(4, 3_700_000_000, group="black", lip=True, track_ids=[55]),
    )

    assert not any(event.name == "POCKET_DETECTED" for event in premature.events)
    assert any(event.name == "POCKET_REJECTED" for event in rejected.events)


def test_strong_crossing_can_start_candidate_for_recently_disappeared_stationary_ball() -> None:
    """A ball can be stationary until impact and cross the pocket between detector frames."""

    sm = _machine()
    stripe = _ball(14, "stripe", 500, 260)
    stripe.velocity_mm_s = (0.0, 0.0)
    sm.update(
        TracksFrame(1, 1_000_000_000, [stripe]),
        _visual(1, 1_000_000_000, group="stripe", clear=True),
    )
    candidate = sm.update(
        TracksFrame(2, 1_100_000_000, []),
        _visual(
            2,
            1_100_000_000,
            group="stripe",
            inward=True,
            track_ids=[14],
            motion_score=0.97,
            foreground_score=0.91,
            foreground_depth_diameters=0.44,
        ),
    )
    detected = sm.update(
        TracksFrame(3, 2_400_000_000, []),
        _visual(3, 2_400_000_000, group="stripe", clear=True, track_ids=[14]),
    )

    assert any(event.name == "POCKET_CANDIDATE" for event in candidate.events)
    assert any(event.name == "POCKET_DETECTED" for event in detected.events)


def test_deep_crossing_can_start_candidate_for_newborn_visible_track() -> None:
    """Regression: shot 27 had a two-frame zero-speed track at the crossing."""

    sm = _machine()
    stripe = _ball(1019, "stripe", 500, 90)
    stripe.velocity_mm_s = (0.0, 0.0)
    candidate = sm.update(
        TracksFrame(1, 1_000_000_000, [stripe]),
        _visual(
            1,
            1_000_000_000,
            group="stripe",
            inward=True,
            track_ids=[1019],
            motion_score=1.0,
            foreground_score=1.0,
            foreground_depth_diameters=0.44,
        ),
    )
    early = sm.update(
        TracksFrame(2, 1_100_000_000, []),
        _visual(2, 1_100_000_000, group="stripe", clear=True, track_ids=[1019]),
    )
    detected = sm.update(
        TracksFrame(3, 2_300_000_000, []),
        _visual(3, 2_300_000_000, group="stripe", clear=True, track_ids=[1019]),
    )

    assert any(event.name == "POCKET_CANDIDATE" for event in candidate.events)
    assert not any(event.name == "POCKET_DETECTED" for event in early.events)
    assert any(event.name == "POCKET_DETECTED" for event in detected.events)


def test_strong_crossing_is_relayed_when_shot_phase_starts_late() -> None:
    """Regression: an old replay entered SHOT_ACTIVE three frames after shot 4 crossed."""

    config = StateConfig(engine="modern", pocket_visual_confirmation_ms=1300, pocket_entry_handoff_ms=450)
    fsm = PerBallPocketFSM(config)
    fsm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(500, 0)],
        ball_diameter_mm=56,
    )
    stripe = _ball(14, "stripe", 500, 260)
    stripe.velocity_mm_s = (0.0, 0.0)
    fsm.update(TracksFrame(1, 1_000_000_000, [stripe]), MatchPhase.STABLE_IDLE, _visual(1, 1_000_000_000))
    deferred = fsm.update(
        TracksFrame(2, 1_100_000_000, []),
        MatchPhase.STABLE_IDLE,
        _visual(
            2,
            1_100_000_000,
            group="stripe",
            inward=True,
            track_ids=[14],
            motion_score=0.97,
            foreground_score=0.91,
            foreground_depth_diameters=0.44,
        ),
    )
    activated = fsm.update(
        TracksFrame(3, 1_300_000_000, []),
        MatchPhase.SHOT_ACTIVE,
        _visual(3, 1_300_000_000, group="stripe", clear=True, track_ids=[14]),
    )
    detected = fsm.update(
        TracksFrame(4, 2_400_000_000, []),
        MatchPhase.SHOT_ACTIVE,
        _visual(4, 2_400_000_000, group="stripe", clear=True, track_ids=[14]),
    )

    assert not any(event.name == "POCKET_CANDIDATE" for event in deferred)
    assert any(event.name == "POCKET_CANDIDATE" for event in activated)
    assert any(event.name == "POCKET_DETECTED" for event in detected)


def test_projected_only_blurred_ball_is_rejected_and_round_detection_is_ignored() -> None:
    config = StateConfig(engine="modern", pocket_visual_confirmation_ms=1300)

    def run(*, elongated: bool, quality: float) -> list[str]:
        fsm = PerBallPocketFSM(config)
        fsm.set_table_context(
            inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
            pockets_mm=[(500, 0)],
            ball_diameter_mm=56,
        )
        ball = _ball(88, "solid", 500, 180)
        ball.quality = quality
        ball.confidence = quality
        ball.bbox = (480, 145, 520, 215) if elongated else (480, 160, 520, 200)
        names: list[str] = []
        for frame in (
            TracksFrame(1, 1_000_000_000, [ball]),
            TracksFrame(2, 1_100_000_000, [ball]),
            TracksFrame(3, 1_600_000_000, []),
            TracksFrame(4, 2_899_000_000, []),
            TracksFrame(5, 2_900_000_000, []),
        ):
            visual = PocketVisualObservationFrame(frame.frame_id, frame.ts_cam_ns, [])
            names.extend(event.name for event in fsm.update(frame, MatchPhase.SHOT_ACTIVE, visual))
        return names

    blurred_names = run(elongated=True, quality=0.70)
    round_names = run(elongated=False, quality=0.95)

    assert "POCKET_CANDIDATE" in blurred_names
    assert blurred_names.count("POCKET_REJECTED") == 1
    assert "POCKET_DETECTED" not in blurred_names
    assert "POCKET_CANDIDATE" not in round_names
    assert "POCKET_DETECTED" not in round_names


def test_oversized_blurred_fragment_cannot_become_goal() -> None:
    """Regression: the frame-1502 black false positive was a rail-sized fragment."""

    config = StateConfig(engine="modern", pocket_visual_confirmation_ms=1300)
    fsm = PerBallPocketFSM(config)
    fsm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(500, 0)],
        ball_diameter_mm=56,
    )
    fragment = _ball(864, "black", 500, 180)
    fragment.quality = 0.70
    fragment.confidence = 0.70
    fragment.bbox = (471, 55, 529, 305)

    fsm.update(TracksFrame(1, 1_000_000_000, [fragment]), MatchPhase.STABLE_IDLE)
    fsm.update(TracksFrame(2, 1_100_000_000, [fragment]), MatchPhase.STABLE_IDLE)
    fsm.update(TracksFrame(3, 1_700_000_000, []), MatchPhase.STABLE_IDLE)

    events = []
    for frame in (
        TracksFrame(4, 1_900_000_000, []),
        TracksFrame(5, 2_300_000_000, []),
        TracksFrame(6, 3_300_000_000, []),
    ):
        visual = PocketVisualObservationFrame(frame.frame_id, frame.ts_cam_ns, [])
        events.extend(fsm.update(frame, MatchPhase.SHOT_ACTIVE, visual))

    assert not any(event.name == "POCKET_CANDIDATE" for event in events)
    assert not any(event.name == "POCKET_DETECTED" for event in events)


def test_visual_inward_candidate_rejects_persistent_position_reversal() -> None:
    """Regression: a jaw rebound must win over a lagging inward velocity estimate."""

    config = StateConfig(engine="modern", pocket_visual_confirmation_ms=1300)
    fsm = PerBallPocketFSM(config)
    fsm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(500, 0)],
        ball_diameter_mm=56,
    )
    far = _ball(55, "solid", 500, 197)
    near = _ball(55, "solid", 500, 50)
    rebound = _ball(55, "solid", 505, 115)
    # Match the real false positive: position has moved back by more than one
    # ball diameter while the smoothed velocity still points at the pocket.
    rebound.velocity_mm_s = (0.0, -200.0)

    fsm.update(
        TracksFrame(1, 1_000_000_000, [far]),
        MatchPhase.SHOT_ACTIVE,
        _visual(1, 1_000_000_000, clear=True, track_ids=[55]),
    )
    candidate = fsm.update(
        TracksFrame(2, 1_100_000_000, [near]),
        MatchPhase.SHOT_ACTIVE,
        _visual(2, 1_100_000_000, inward=True, track_ids=[55]),
    )
    first_reversal = fsm.update(
        TracksFrame(3, 1_200_000_000, [rebound]),
        MatchPhase.SHOT_ACTIVE,
        _visual(3, 1_200_000_000, clear=True, track_ids=[55]),
    )
    persistent_reversal = fsm.update(
        TracksFrame(4, 1_300_000_000, [rebound]),
        MatchPhase.SHOT_ACTIVE,
        _visual(4, 1_300_000_000, clear=True, track_ids=[55]),
    )
    later = fsm.update(
        TracksFrame(5, 2_700_000_000, []),
        MatchPhase.SHOT_ACTIVE,
        _visual(5, 2_700_000_000, clear=True, track_ids=[55]),
    )

    assert any(event.name == "POCKET_CANDIDATE" for event in candidate)
    assert not any(event.name == "POCKET_REJECTED" for event in first_reversal)
    rejection = next(event for event in persistent_reversal if event.name == "POCKET_REJECTED")
    assert "projected_entry_reversed" in rejection.payload["reason_codes"]
    assert not any(event.name == "POCKET_DETECTED" for event in [*first_reversal, *persistent_reversal, *later])
