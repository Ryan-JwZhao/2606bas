from __future__ import annotations

import pytest

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
    entry_depth_diameters: float | None = None,
    lip_track_ids: list[int] | None = None,
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
                lip_track_ids=list(lip_track_ids or []),
                evidence_sources=["frame_difference", "ball_sized_motion", "foreground_motion"],
                motion_score=motion_score,
                foreground_score=foreground_score,
                foreground_depth_diameters=foreground_depth_diameters,
                entry_depth_diameters=entry_depth_diameters,
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


@pytest.mark.parametrize(
    ("second_track_id", "second_group", "second_delay_ms"),
    [
        (112, "solid", 1100),
        (113, "solid", 1100),
        (112, "stripe", 1100),
        (112, "solid", 3500),
    ],
)
def test_two_same_pocket_goals_within_five_seconds_keep_independent_identities(
    second_track_id: int,
    second_group: str,
    second_delay_ms: int,
) -> None:
    """01:35: fast physical balls stay independent across tracker id churn and reuse."""

    sm = _machine()
    events = []
    events.extend(
        sm.update(
            TracksFrame(1, 1_000_000_000, [_ball(112, "solid", 500, 170)]),
            _visual(1, 1_000_000_000, clear=True, track_ids=[112]),
        ).events
    )
    events.extend(
        sm.update(
            TracksFrame(2, 1_100_000_000, []),
            _visual(2, 1_100_000_000, inward=True, track_ids=[112]),
        ).events
    )
    events.extend(
        sm.update(
            TracksFrame(3, 2_400_000_000, []),
            _visual(3, 2_400_000_000, clear=True, track_ids=[112]),
        ).events
    )

    # A second physical ball reaches the same pocket quickly enough for both
    # detections to remain inside five seconds. Trackers may reuse the id,
    # allocate a new id, or also revise the colour classification.
    second_visible_ns = 2_400_000_000 + second_delay_ms * 1_000_000
    second_crossing_ns = second_visible_ns + 100_000_000
    second_detected_ns = second_crossing_ns + 1_300_000_000
    events.extend(
        sm.update(
            TracksFrame(4, second_visible_ns, [_ball(second_track_id, second_group, 500, 170)]),
            _visual(
                4,
                second_visible_ns,
                group=second_group,
                clear=True,
                track_ids=[second_track_id],
            ),
        ).events
    )
    events.extend(
        sm.update(
            TracksFrame(5, second_crossing_ns, []),
            _visual(
                5,
                second_crossing_ns,
                group=second_group,
                inward=True,
                track_ids=[second_track_id],
            ),
        ).events
    )
    events.extend(
        sm.update(
            TracksFrame(6, second_detected_ns, []),
            _visual(
                6,
                second_detected_ns,
                group=second_group,
                clear=True,
                track_ids=[second_track_id],
            ),
        ).events
    )

    detected = [event for event in events if event.name == "POCKET_DETECTED"]
    assert len(detected) == 2
    assert len({event.payload["decision_id"] for event in detected}) == 2
    assert not any(event.name in {"POCKET_REAPPEARED", "POCKET_REJECTED"} for event in events)


def test_three_rapid_same_track_goals_are_all_committed_at_turn_resolve() -> None:
    sm = _machine()
    sm.rule_state.table_state = "closed"
    sm.rule_state.actor_group = "solid"
    sm.rule_state.opponent_group = "stripe"
    events = []

    for sequence in range(3):
        frame_id = sequence * 3 + 1
        visible_ns = 1_000_000_000 + sequence * 1_500_000_000
        crossing_ns = visible_ns + 100_000_000
        detected_ns = crossing_ns + 1_300_000_000
        events.extend(
            sm.update(
                TracksFrame(frame_id, visible_ns, [_ball(112, "solid", 500, 170)]),
                _visual(frame_id, visible_ns, clear=True, track_ids=[112]),
            ).events
        )
        events.extend(
            sm.update(
                TracksFrame(frame_id + 1, crossing_ns, []),
                _visual(frame_id + 1, crossing_ns, inward=True, track_ids=[112]),
            ).events
        )
        events.extend(
            sm.update(
                TracksFrame(frame_id + 2, detected_ns, []),
                _visual(frame_id + 2, detected_ns, clear=True, track_ids=[112]),
            ).events
        )

    sm.force_phase(MatchPhase.TURN_RESOLVE, frame_id=10, ts_cam_ns=5_600_000_000)
    resolved = sm.update(TracksFrame(10, 5_600_000_000, [_ball(1, "cue", 700, 250)]))

    assert sum(event.name == "POCKET_DETECTED" for event in events) == 3
    assert sum(event.name == "POCKET_CONFIRMED" for event in resolved.events) == 3
    assert sm.ledger.remaining["solid"] == 4


def test_visual_outward_crossing_rejects_bounce() -> None:
    sm = _machine()
    sm.update(TracksFrame(1, 1_000_000_000, [_ball(2, "stripe", 500, 100)]), _visual(1, 1_000_000_000, group="stripe", clear=True))
    sm.update(TracksFrame(2, 1_100_000_000, []), _visual(2, 1_100_000_000, group="stripe", inward=True, track_ids=[2]))
    bounced = sm.update(TracksFrame(3, 1_250_000_000, []), _visual(3, 1_250_000_000, group="stripe", outward=True, track_ids=[2]))

    rejection = next(event for event in bounced.events if event.name == "POCKET_REJECTED")
    assert "visual_outward_crossing" in rejection.payload["reason_codes"]


def test_video_projected_entry_with_associated_lip_motion_can_confirm_after_clear() -> None:
    """01:39: a fast same-track ball vanished at the lip and never reappeared."""

    sm = _machine()
    far = _ball(2, "solid", 500, 178)
    far.velocity_mm_s = (-23.0, -362.0)
    near = _ball(2, "solid", 500, 52)
    near.velocity_mm_s = (-3.0, -416.0)
    sm.update(
        TracksFrame(1, 1_000_000_000, [far]),
        _visual(1, 1_000_000_000, clear=True, track_ids=[2]),
    )
    candidate = sm.update(
        TracksFrame(2, 1_100_000_000, [near]),
        _visual(2, 1_100_000_000, lip=True, track_ids=[2], motion_score=0.95, foreground_score=0.9),
    )
    sm.update(
        TracksFrame(3, 1_200_000_000, []),
        _visual(3, 1_200_000_000, lip=True, track_ids=[2], motion_score=0.9, foreground_score=0.85),
    )
    before_window = sm.update(
        TracksFrame(4, 2_250_000_000, []),
        _visual(4, 2_250_000_000, lip=True, track_ids=[2], motion_score=0.8, foreground_score=0.75),
    )
    confirmed = sm.update(
        TracksFrame(5, 2_500_000_000, []),
        _visual(5, 2_500_000_000, clear=True, track_ids=[2]),
    )

    assert any(event.name == "POCKET_CANDIDATE" for event in candidate.events)
    assert not any(event.name == "POCKET_REJECTED" for event in before_window.events)
    assert any(event.name == "POCKET_DETECTED" for event in confirmed.events)


def test_projected_visual_entry_still_rejects_persistent_lip_after_extended_window() -> None:
    sm = _machine()
    far = _ball(2, "solid", 500, 178)
    far.velocity_mm_s = (0.0, -362.0)
    near = _ball(2, "solid", 500, 52)
    near.velocity_mm_s = (0.0, -416.0)
    sm.update(TracksFrame(1, 1_000_000_000, [far]), _visual(1, 1_000_000_000, clear=True, track_ids=[2]))
    sm.update(
        TracksFrame(2, 1_100_000_000, [near]),
        _visual(2, 1_100_000_000, lip=True, track_ids=[2], motion_score=0.95, foreground_score=0.9),
    )
    sm.update(
        TracksFrame(3, 1_200_000_000, []),
        _visual(3, 1_200_000_000, lip=True, track_ids=[2], motion_score=0.9, foreground_score=0.85),
    )
    rejected = sm.update(
        TracksFrame(4, 2_900_000_000, []),
        _visual(4, 2_900_000_000, lip=True, track_ids=[2], motion_score=0.8, foreground_score=0.75),
    )

    event = next(event for event in rejected.events if event.name == "POCKET_REJECTED")
    assert "persistent_lip_occupancy" in event.payload["reason_codes"]
    assert not any(event.name == "POCKET_DETECTED" for event in rejected.events)


def test_stale_lip_cannot_be_rearmed_by_history_only_foreground() -> None:
    sm = _machine()
    far = _ball(2, "solid", 500, 178)
    far.velocity_mm_s = (0.0, -362.0)
    near = _ball(2, "solid", 500, 52)
    near.velocity_mm_s = (0.0, -416.0)
    sm.update(TracksFrame(1, 1_000_000_000, [far]), _visual(1, 1_000_000_000, clear=True, track_ids=[2]))
    sm.update(
        TracksFrame(2, 1_100_000_000, [near]),
        _visual(2, 1_100_000_000, lip=True, track_ids=[2], motion_score=0.95, foreground_score=0.9),
    )
    sm.update(
        TracksFrame(3, 1_200_000_000, []),
        _visual(3, 1_200_000_000, lip=True, track_ids=[2], motion_score=0.9, foreground_score=0.85),
    )
    sm.update(TracksFrame(4, 1_700_000_000, []), PocketVisualObservationFrame(4, 1_700_000_000, []))
    history_only = sm.update(
        TracksFrame(5, 2_100_000_000, []),
        _visual(5, 2_100_000_000, lip=True, track_ids=[2], motion_score=0.8, foreground_score=0.75),
    )
    detected = sm.update(
        TracksFrame(6, 2_500_000_000, []),
        PocketVisualObservationFrame(6, 2_500_000_000, []),
    )

    assert not any(event.name == "POCKET_REJECTED" for event in history_only.events)
    assert any(event.name == "POCKET_DETECTED" for event in detected.events)


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
        TracksFrame(4, 2_600_000_000, []),
        MatchPhase.SHOT_ACTIVE,
        _visual(4, 2_600_000_000, group="stripe", clear=True, track_ids=[14]),
    )

    assert not any(event.name == "POCKET_CANDIDATE" for event in deferred)
    assert any(event.name == "POCKET_CANDIDATE" for event in activated)
    assert any(event.name == "POCKET_DETECTED" for event in detected)


def test_lip_departure_is_relayed_across_two_second_shot_start_delay() -> None:
    """00:13: a lip ball vanished before fragmented cue tracking armed the shot."""

    fsm = PerBallPocketFSM(StateConfig(engine="modern", pocket_visual_confirmation_ms=1300))
    fsm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(500, 0)],
        ball_diameter_mm=56,
    )
    ball = _ball(9, "solid", 500, 15)
    ball.velocity_mm_s = (0.0, 0.0)
    fsm.update(
        TracksFrame(1, 1_000_000_000, [ball]),
        MatchPhase.STABLE_IDLE,
        _visual(1, 1_000_000_000, lip=True, track_ids=[9]),
    )
    fsm.update(
        TracksFrame(2, 1_350_000_000, []),
        MatchPhase.STABLE_IDLE,
        _visual(
            2,
            1_350_000_000,
            inward=True,
            track_ids=[9],
            motion_score=0.55,
            foreground_score=1.0,
            foreground_depth_diameters=-1.05,
        ),
    )
    activated = fsm.update(
        TracksFrame(3, 3_300_000_000, []),
        MatchPhase.SHOT_ACTIVE,
        _visual(3, 3_300_000_000, lip=True, track_ids=[]),
    )
    detected = fsm.update(
        TracksFrame(4, 4_600_000_000, []),
        MatchPhase.SHOT_ACTIVE,
        _visual(4, 4_600_000_000, clear=True, track_ids=[]),
    )

    assert any(event.name == "POCKET_CANDIDATE" for event in activated)
    assert any(event.name == "POCKET_DETECTED" for event in detected)


def test_newborn_mouth_track_uses_handoff_and_terminal_disappearance() -> None:
    """01:33: the target changed from track 72 to one-frame track 73 at the mouth."""

    sm = _machine()
    source = _ball(72, "solid", 500, 235)
    source.velocity_mm_s = (0.0, 0.0)
    newborn = _ball(73, "solid", 500, 62)
    newborn.velocity_mm_s = (0.0, 0.0)
    sm.update(
        TracksFrame(1, 1_000_000_000, [source]),
        _visual(1, 1_000_000_000, clear=True),
    )
    candidate = sm.update(
        TracksFrame(2, 1_116_000_000, [newborn]),
        _visual(
            2,
            1_116_000_000,
            track_ids=[73],
            motion_score=0.44,
            foreground_score=0.55,
            foreground_depth_diameters=-0.41,
        ),
    )
    sm.update(
        TracksFrame(3, 1_232_000_000, []),
        _visual(3, 1_232_000_000, track_ids=[73], motion_score=1.0, foreground_score=1.0),
    )
    sm.update(
        TracksFrame(4, 1_950_000_000, []),
        _visual(4, 1_950_000_000, clear=True, track_ids=[73]),
    )
    detected = sm.update(
        TracksFrame(5, 2_450_000_000, []),
        _visual(5, 2_450_000_000, clear=True, track_ids=[73]),
    )

    event = next(event for event in candidate.events if event.name == "POCKET_CANDIDATE")
    assert event.payload["candidate_reason"] == "projected_entry_track_handoff"
    assert event.payload["evidence"]["entry_source_track_id"] == 72
    assert any(event.name == "POCKET_DETECTED" for event in detected.events)


def test_narrow_same_track_terminal_disappearance_confirms_without_lip_flag() -> None:
    """01:40: a narrow same-track entry was visible, but the observer stayed clear."""

    sm = _machine()
    far = _ball(4, "solid", 500, 178)
    far.velocity_mm_s = (-23.0, -362.0)
    near = _ball(4, "solid", 500, 52)
    near.velocity_mm_s = (-3.0, -416.0)
    sm.update(
        TracksFrame(1, 1_000_000_000, [far]),
        _visual(1, 1_000_000_000, clear=True),
    )
    candidate = sm.update(
        TracksFrame(2, 1_100_000_000, [near]),
        _visual(2, 1_100_000_000, track_ids=[4], motion_score=1.0, foreground_score=1.0),
    )
    sm.update(
        TracksFrame(3, 1_216_000_000, []),
        _visual(3, 1_216_000_000, track_ids=[4], motion_score=1.0, foreground_score=1.0),
    )
    sm.update(
        TracksFrame(4, 1_950_000_000, []),
        _visual(4, 1_950_000_000, clear=True, track_ids=[4]),
    )
    detected = sm.update(
        TracksFrame(5, 2_450_000_000, []),
        _visual(5, 2_450_000_000, clear=True, track_ids=[4]),
    )

    assert any(event.name == "POCKET_CANDIDATE" for event in candidate.events)
    assert any(event.name == "POCKET_DETECTED" for event in detected.events)


def test_low_fps_terminal_visual_crossing_can_start_stationary_lip_departure() -> None:
    """A collision-potted lip ball relies on terminal visual motion when its last track was static."""

    sm = _machine()
    ball = _ball(41, "stripe", 500, 72)
    ball.velocity_mm_s = (0.0, 0.0)
    sm.update(
        TracksFrame(1, 1_000_000_000, [ball]),
        _visual(1, 1_000_000_000, clear=True, track_ids=[41]),
    )
    candidate = sm.update(
        TracksFrame(2, 1_116_000_000, []),
        _visual(
            2,
            1_116_000_000,
            group="stripe",
            inward=True,
            track_ids=[41],
            motion_score=1.0,
            foreground_score=1.0,
            foreground_depth_diameters=-0.65,
            entry_depth_diameters=-0.65,
        ),
    )
    detected = sm.update(
        TracksFrame(3, 2_416_000_000, []),
        _visual(3, 2_416_000_000, group="stripe", clear=True, track_ids=[41]),
    )

    assert any(event.name == "POCKET_CANDIDATE" for event in candidate.events)
    assert any(event.name == "POCKET_DETECTED" for event in detected.events)


def test_other_live_lip_ball_does_not_veto_disappeared_crossing_track() -> None:
    """Two same-group balls at one pocket keep independent crossing and lip identities."""

    sm = _machine()
    target = _ball(14, "solid", 500, 72)
    target.velocity_mm_s = (0.0, -320.0)
    static = _ball(4, "solid", 535, 45)
    static.velocity_mm_s = (0.0, 0.0)
    sm.update(
        TracksFrame(1, 1_000_000_000, [target, static]),
        _visual(1, 1_000_000_000, clear=True),
    )
    candidate = sm.update(
        TracksFrame(2, 1_100_000_000, [static]),
        _visual(
            2,
            1_100_000_000,
            inward=True,
            lip=True,
            track_ids=[14],
            lip_track_ids=[4],
            motion_score=1.0,
            foreground_score=1.0,
            foreground_depth_diameters=-0.4,
            entry_depth_diameters=-0.4,
        ),
    )
    detected = sm.update(
        TracksFrame(3, 2_400_000_000, [static]),
        _visual(3, 2_400_000_000, lip=True, track_ids=[14], lip_track_ids=[4]),
    )

    assert any(event.name == "POCKET_CANDIDATE" for event in candidate.events)
    assert any(event.name == "POCKET_DETECTED" for event in detected.events)


def test_calibrated_terminal_corridor_accepts_projected_safe_off_axis_entry() -> None:
    """Terminal confirmation uses the calibrated mouth corridor accepted by trajectory projection."""

    sm = _machine()
    far = _ball(52, "stripe", 570, 178)
    far.velocity_mm_s = (-160.0, -362.0)
    near = _ball(52, "stripe", 550, 52)
    near.velocity_mm_s = (-200.0, -416.0)
    sm.update(
        TracksFrame(1, 1_000_000_000, [far]),
        _visual(1, 1_000_000_000, clear=True),
    )
    candidate = sm.update(
        TracksFrame(2, 1_100_000_000, [near]),
        _visual(2, 1_100_000_000, group="stripe", track_ids=[52], motion_score=1.0, foreground_score=1.0),
    )
    sm.update(
        TracksFrame(3, 1_216_000_000, []),
        _visual(3, 1_216_000_000, group="stripe", track_ids=[52], motion_score=1.0, foreground_score=1.0),
    )
    sm.update(
        TracksFrame(4, 1_950_000_000, []),
        _visual(4, 1_950_000_000, group="stripe", clear=True, track_ids=[52]),
    )
    detected = sm.update(
        TracksFrame(5, 2_450_000_000, []),
        _visual(5, 2_450_000_000, group="stripe", clear=True, track_ids=[52]),
    )

    event = next(event for event in candidate.events if event.name == "POCKET_CANDIDATE")
    assert abs(float(event.payload["evidence"]["entry_lateral_mm"])) > 0.75 * 56.0
    assert any(event.name == "POCKET_DETECTED" for event in detected.events)


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
