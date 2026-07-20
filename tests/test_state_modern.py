from __future__ import annotations

import json

from bas.config import AppConfig, StateConfig
from bas.schemas import Event, MatchPhase, TrackObservation, TracksFrame
from bas.state import LegacyMatchStateMachine, ModernMatchStateMachine, create_match_state_machine
from bas.state.models import InventoryLedger, MatchRuleState, ShotContext
from bas.state.reconcile import ObservationReconciler
from bas.state.referee import RefereeAdapter, ShotContextAggregator
from bas.user_settings import UserSettings


def _ball(
    track_id: int,
    group: str,
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
    *,
    visibility: str = "visible",
    lost_frames: int = 0,
    quality: float = 0.9,
) -> TrackObservation:
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
        quality=quality,
        visibility=visibility,
        lost_frames=lost_frames,
    )


def test_state_factory_defaults_to_legacy_and_can_create_modern() -> None:
    assert isinstance(create_match_state_machine(StateConfig()), LegacyMatchStateMachine)
    assert isinstance(create_match_state_machine(StateConfig(engine="modern")), ModernMatchStateMachine)


def test_user_settings_persists_state_machine_engine(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps({"state_machine_engine": "modern"}), encoding="utf-8")
    cfg = AppConfig()

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.state.engine == "modern"
    saved = UserSettings.from_config(cfg)
    assert saved.state_machine_engine == "modern"


def test_modern_mouth_rest_disappearance_requires_review_not_commit() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", -20, -10, -20, -10)]))
    sm.update(TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[_ball(2, "solid", -20, -10)]))
    sm.update(TracksFrame(frame_id=3, ts_cam_ns=1_200_000_000, tracks=[]))
    tentative = sm.update(TracksFrame(frame_id=4, ts_cam_ns=1_500_000_000, tracks=[]))
    reviewed = sm.update(TracksFrame(frame_id=5, ts_cam_ns=1_900_000_000, tracks=[]))

    assert any(event.name == "POCKET_TENTATIVE" for event in tentative.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in reviewed.events)
    review = next(event for event in reviewed.events if event.name == "POCKET_REVIEW_REQUIRED")
    assert review.payload["review_required"] is True
    assert "mouth_rest_disappear_requires_review" in review.payload["reason_codes"]


def test_modern_pocket_confirmation_waits_for_missing_window() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    first = sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", -50, -25, -20, -10)]))
    sm.update(TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[]))
    early = sm.update(TracksFrame(frame_id=3, ts_cam_ns=1_400_000_000, tracks=[]))
    commit_ready = sm.update(TracksFrame(frame_id=4, ts_cam_ns=1_800_000_000, tracks=[]))
    sm.force_phase(MatchPhase.TURN_RESOLVE, frame_id=5, ts_cam_ns=1_850_000_000)
    deferred = sm.update(TracksFrame(frame_id=5, ts_cam_ns=1_850_000_000, tracks=[_ball(1, "cue", 100, 100)]))
    confirmed = sm.update(TracksFrame(frame_id=6, ts_cam_ns=1_900_000_000, tracks=[_ball(1, "cue", 100, 100)]))

    assert any(event.name == "POCKET_CANDIDATE" for event in first.events)
    assert any(event.name == "POCKET_TENTATIVE" for event in early.events)
    assert any(event.name == "POCKET_COMMIT_READY" for event in commit_ready.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in commit_ready.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in deferred.events)
    assert any(event.name == "POCKET_CONFIRMED" for event in confirmed.events)
    assert any(event.name == "POT_PROBABLE" for event in confirmed.events)


def test_modern_mouth_inward_disappearance_requires_review_not_commit() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    first = sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", -20, -10, -20, -10)]))
    sm.update(TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[]))
    tentative = sm.update(TracksFrame(frame_id=3, ts_cam_ns=1_400_000_000, tracks=[]))
    reviewed = sm.update(TracksFrame(frame_id=4, ts_cam_ns=1_800_000_000, tracks=[]))

    candidate = next(event for event in first.events if event.name == "POCKET_CANDIDATE")
    assert candidate.payload["candidate_reason"] == "mouth_inward_trend"
    assert any(event.name == "POCKET_TENTATIVE" for event in tentative.events)
    assert not any(event.name == "POCKET_COMMIT_READY" for event in reviewed.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in reviewed.events)
    review = next(event for event in reviewed.events if event.name == "POCKET_REVIEW_REQUIRED")
    assert "insufficient_pocket_evidence" in review.payload["reason_codes"]


def test_modern_pocket_reappearing_with_new_track_is_rejected() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", -50, -25, -20, -10)]))
    sm.update(TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[]))
    out = sm.update(TracksFrame(frame_id=3, ts_cam_ns=1_180_000_000, tracks=[_ball(99, "solid", -45, -22)]))

    assert any(event.name == "POCKET_REAPPEARED" for event in out.events)
    assert any(event.name == "POCKET_REJECTED" for event in out.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in out.events)


def test_modern_cross_group_reappearing_near_same_pocket_requires_review() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", -50, -25, -20, -10)]))
    sm.update(TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[]))
    out = sm.update(TracksFrame(frame_id=3, ts_cam_ns=1_400_000_000, tracks=[_ball(99, "black", -45, -22)]))

    assert any(event.name == "POCKET_REAPPEARED" for event in out.events)
    assert any(event.name == "POCKET_REVIEW_REQUIRED" for event in out.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in out.events)
    review = next(event for event in out.events if event.name == "POCKET_REVIEW_REQUIRED")
    assert "reappeared_group_changed" in review.payload["reason_codes"]


def test_occluded_frames_without_true_loss_do_not_confirm() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", -50, -25, -20, -10)]))
    first_occluded = sm.update(
        TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[_ball(2, "solid", -50, -25, -20, -10, visibility="occluded", lost_frames=1)])
    )
    second_occluded = sm.update(
        TracksFrame(frame_id=3, ts_cam_ns=1_400_000_000, tracks=[_ball(2, "solid", -50, -25, -20, -10, visibility="occluded", lost_frames=2)])
    )

    assert not any(event.name == "POCKET_TENTATIVE" for event in first_occluded.events)
    assert any(event.name == "POCKET_TENTATIVE" for event in second_occluded.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in second_occluded.events)
    assert not any(event.name == "POCKET_REVIEW_REQUIRED" for event in second_occluded.events)


def test_referee_uses_ledger_not_visible_counts_for_black_upgrade() -> None:
    ledger = InventoryLedger()
    ledger.remaining["solid"] = 2
    rule_state = MatchRuleState(table_state="closed", actor_group="solid", opponent_group="stripe")
    shot = ShotContext(shot_id=1, ts_start_ms=0, table_state_before="closed", actor_group="solid")
    shot.potted_confirmed["solid"] = 1
    ledger.apply(shot)

    intent = RefereeAdapter().evaluate(shot, ledger, rule_state)

    assert ledger.remaining["solid"] == 1
    assert intent.next_group_hint == "solid"
    assert intent.next_actor_changed is False


def test_referee_switches_to_opponent_when_actor_does_not_pot_own_group() -> None:
    ledger = InventoryLedger()
    rule_state = MatchRuleState(table_state="closed", actor_group="solid", opponent_group="stripe")
    shot = ShotContext(shot_id=1, ts_start_ms=0, table_state_before="closed", actor_group="solid")
    shot.first_contact_group = "solid"
    shot.potted_confirmed["stripe"] = 1
    ledger.apply(shot)

    intent = RefereeAdapter().evaluate(shot, ledger, rule_state)

    assert intent.next_group_hint == "stripe"
    assert intent.next_actor_changed is True
    assert intent.foul_flags["wrong_first_contact"] is False


def test_referee_exposes_wrong_first_contact_and_ball_in_hand_scope() -> None:
    ledger = InventoryLedger()
    rule_state = MatchRuleState(table_state="closed", actor_group="solid", opponent_group="stripe")
    shot = ShotContext(shot_id=1, ts_start_ms=0, table_state_before="closed", actor_group="solid")
    shot.first_contact_group = "stripe"

    intent = RefereeAdapter().evaluate(shot, ledger, rule_state)

    assert intent.foul_flags["wrong_first_contact"] is True
    assert intent.ball_in_hand_scope == "table_anywhere"
    assert intent.next_group_hint == "stripe"


def test_referee_open_table_single_group_pot_assigns_groups() -> None:
    ledger = InventoryLedger()
    rule_state = MatchRuleState(table_state="open")
    shot = ShotContext(shot_id=1, ts_start_ms=0, break_shot=False, table_state_before="open")
    shot.potted_confirmed["solid"] = 1
    ledger.apply(shot)

    intent = RefereeAdapter().evaluate(shot, ledger, rule_state)

    assert intent.table_state_after == "closed"
    assert intent.actor_group_after == "solid"
    assert intent.opponent_group_after == "stripe"
    assert intent.next_group_hint == "solid"


def test_modern_operator_force_turn_resolve_finalizes_context() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern"))
    sm.force_phase(MatchPhase.TURN_RESOLVE, frame_id=10, ts_cam_ns=1_000_000_000)

    out = sm.update(TracksFrame(frame_id=10, ts_cam_ns=1_000_000_000, tracks=[_ball(1, "cue", 100, 100)]))

    assert out.phase == "TURN_RESOLVE"
    assert any(event.name == "REFEREE_INTENT" for event in out.events)


def test_modern_defers_turn_resolve_until_pending_pocket_confirms() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700, turn_resolve_grace_ms=900))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", -50, -25, -20, -10)]))
    resolve_frame = TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[])
    deferred_events = [Event("TURN_RESOLVE", 1_100_000_000, 2)]
    deferred_events.extend(sm.pocket_fsm.update(resolve_frame, MatchPhase.TURN_RESOLVE))
    sm._annotate_shot_ids(deferred_events, ts_ms=1100)
    sm.aggregator.ingest(deferred_events, ts_ms=1100, rule_state=sm.rule_state)
    sm._process_turn_resolve_if_needed(resolve_frame, deferred_events)

    assert any(event.name == "TURN_RESOLVE_DEFERRED" for event in deferred_events)
    assert not any(event.name == "REFEREE_INTENT" for event in deferred_events)

    mid = sm.update(TracksFrame(frame_id=3, ts_cam_ns=1_400_000_000, tracks=[]))
    ready = sm.update(TracksFrame(frame_id=4, ts_cam_ns=1_800_000_000, tracks=[]))
    out = sm.update(TracksFrame(frame_id=5, ts_cam_ns=1_900_000_000, tracks=[]))

    assert not any(event.name == "POCKET_CONFIRMED" for event in mid.events)
    assert any(event.name == "POCKET_COMMIT_READY" for event in ready.events)
    assert any(event.name == "POCKET_CONFIRMED" for event in out.events)
    assert any(event.name == "TURN_RESOLVE_COMMITTED" for event in out.events)
    assert any(event.name == "REFEREE_INTENT" for event in out.events)
    assert sm.ledger.remaining["solid"] == 6


def test_350ms_missing_then_same_track_reappears_without_confirmation() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(0, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(1, 1_000_000_000, [_ball(2, "solid", -50, -25, -20, -10)]))
    sm.update(TracksFrame(2, 1_100_000_000, []))
    tentative = sm.update(TracksFrame(3, 1_450_000_000, []))
    reappeared = sm.update(TracksFrame(4, 1_500_000_000, [_ball(2, "solid", -45, -22)]))

    assert any(event.name == "POCKET_TENTATIVE" for event in tentative.events)
    assert any(event.name == "POCKET_REJECTED" for event in reappeared.events)
    assert not any(event.name in {"POCKET_COMMIT_READY", "POCKET_CONFIRMED"} for event in reappeared.events)


def test_visible_ball_leaving_pocket_along_rail_rejects_stale_evidence() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        table_edge_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        ball_center_reachable_polygon_mm=[(30, 30), (970, 30), (970, 470), (30, 470)],
        pockets_mm=[(0, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    candidate = sm.update(TracksFrame(1, 1_000_000_000, [_ball(2, "solid", -50, -25, -20, -10)]))
    departed = sm.update(TracksFrame(2, 1_050_000_000, [_ball(2, "solid", 300, 10, 50, 0)]))
    later_events = []
    for frame in (
        TracksFrame(3, 1_100_000_000, []),
        TracksFrame(4, 1_400_000_000, []),
        TracksFrame(5, 1_800_000_000, []),
    ):
        later_events.extend(sm.update(frame).events)

    assert any(event.name == "POCKET_CANDIDATE" for event in candidate.events)
    assert any(event.name == "POCKET_REJECTED" for event in departed.events)
    assert not any(event.name in {"POCKET_TENTATIVE", "POCKET_COMMIT_READY", "POCKET_CONFIRMED"} for event in later_events)


def test_visible_pocket_candidate_is_resolved_before_turn_can_finish() -> None:
    sm = ModernMatchStateMachine(
        StateConfig(engine="modern", pocket_commit_ready_missing_ms=700, turn_resolve_grace_ms=900)
    )
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(0, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(1, 1_000_000_000, [_ball(2, "solid", -50, -25)]))
    sm.force_phase(MatchPhase.TURN_RESOLVE, frame_id=2, ts_cam_ns=1_100_000_000)
    deferred = sm.update(
        TracksFrame(
            2,
            1_100_000_000,
            [_ball(2, "solid", -50, -25), _ball(1, "cue", 100, 100)],
        )
    )
    reviewed = sm.update(
        TracksFrame(
            3,
            2_010_000_000,
            [_ball(2, "solid", -50, -25), _ball(1, "cue", 100, 100)],
        )
    )

    assert any(event.name == "TURN_RESOLVE_DEFERRED" for event in deferred.events)
    assert not any(event.name == "SHOT_CONTEXT_FINALIZED" for event in deferred.events)
    assert any(event.name == "POCKET_REVIEW_REQUIRED" for event in reviewed.events)
    assert sm.debug_snapshot()["pending_review"]


def test_two_previously_tracked_same_group_balls_can_enter_the_same_pocket() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(0, 0)],
        ball_diameter_mm=56,
    )
    sm.rule_state.table_state = "closed"
    sm.rule_state.actor_group = "solid"
    sm.rule_state.opponent_group = "stripe"
    sm.phase = MatchPhase.SHOT_ACTIVE

    first = sm.update(
        TracksFrame(
            1,
            1_000_000_000,
            [_ball(2, "solid", -50, -25, -20, -10), _ball(3, "solid", 250, 100, -50, -20)],
        )
    )
    middle = sm.update(TracksFrame(2, 1_100_000_000, [_ball(3, "solid", 120, 60, -80, -40)]))
    second = sm.update(TracksFrame(3, 1_200_000_000, [_ball(3, "solid", -45, -22, -20, -10)]))
    sm.update(TracksFrame(4, 1_300_000_000, []))
    sm.update(TracksFrame(5, 1_600_000_000, []))
    sm.update(TracksFrame(6, 1_900_000_000, []))
    sm.update(TracksFrame(7, 2_000_000_000, []))
    sm.force_phase(MatchPhase.TURN_RESOLVE, frame_id=8, ts_cam_ns=2_100_000_000)
    resolved = sm.update(TracksFrame(8, 2_100_000_000, [_ball(1, "cue", 100, 100)]))

    assert sum(event.name == "POCKET_CANDIDATE" for event in [*first.events, *middle.events, *second.events]) == 2
    assert not any(event.name in {"POCKET_REAPPEARED", "POCKET_REJECTED"} for event in second.events)
    assert sum(event.name == "POCKET_CONFIRMED" for event in resolved.events) == 2
    assert sm.ledger.remaining["solid"] == 5


def test_video_high_speed_same_track_entry_confirms_after_detector_loses_ball() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(500, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(1, 1_000_000_000, [_ball(215, "solid", 515, 180, 0, -700)]))
    candidate = sm.update(TracksFrame(2, 1_100_000_000, [_ball(215, "solid", 514, 72, 0, -1_080)]))
    sm.update(TracksFrame(3, 1_200_000_000, []))
    sm.update(TracksFrame(4, 1_500_000_000, []))
    ready = sm.update(TracksFrame(5, 1_900_000_000, []))
    sm.force_phase(MatchPhase.TURN_RESOLVE, frame_id=6, ts_cam_ns=2_000_000_000)
    resolved = sm.update(TracksFrame(6, 2_000_000_000, [_ball(1, "cue", 300, 250)]))

    event = next(event for event in candidate.events if event.name == "POCKET_CANDIDATE")
    assert event.payload["candidate_reason"] == "projected_entry_same_track"
    assert event.payload["evidence"]["projected_entry"] is True
    assert any(event.name == "POCKET_COMMIT_READY" for event in ready.events)
    detected = next(event for event in resolved.events if event.name == "POCKET_DETECTED")
    assert detected.payload["shot_id"] == event.payload["shot_id"]
    assert any(event.name == "POCKET_CONFIRMED" for event in resolved.events)


def test_video_high_speed_cross_track_entry_inherits_motion_and_confirms() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(500, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(1, 1_000_000_000, [_ball(597, "solid", 486, 197)]))
    # The runtime repeats cached detections between inference frames.  That
    # duplicate must not shorten the velocity-estimation interval to 31 ms.
    sm.update(TracksFrame(2, 1_078_000_000, [_ball(597, "solid", 486, 197)]))
    candidate = sm.update(TracksFrame(3, 1_109_000_000, [_ball(599, "solid", 485, 36)]))
    sm.update(TracksFrame(4, 1_200_000_000, []))
    sm.update(TracksFrame(5, 1_500_000_000, []))
    ready = sm.update(TracksFrame(6, 1_900_000_000, []))
    sm.force_phase(MatchPhase.TURN_RESOLVE, frame_id=7, ts_cam_ns=2_000_000_000)
    resolved = sm.update(TracksFrame(7, 2_000_000_000, [_ball(1, "cue", 300, 250)]))

    event = next(event for event in candidate.events if event.name == "POCKET_CANDIDATE")
    assert event.payload["candidate_reason"] == "projected_entry_track_handoff"
    assert event.payload["evidence"]["entry_source_track_id"] == 597
    assert event.payload["evidence"]["entry_speed_mm_s"] > 1_000
    assert any(event.name == "POCKET_COMMIT_READY" for event in ready.events)
    assert any(event.name == "POCKET_DETECTED" for event in resolved.events)
    assert any(event.name == "POCKET_CONFIRMED" for event in resolved.events)


def test_video_pocket_jaw_bounce_rejects_projected_entry() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(500, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(1, 1_000_000_000, [_ball(66, "solid", 500, 197)]))
    candidate = sm.update(TracksFrame(2, 1_100_000_000, [_ball(67, "solid", 500, 50)]))
    bounced = sm.update(TracksFrame(3, 1_200_000_000, [_ball(67, "solid", 505, 115, 50, 650)]))
    later_events = []
    for frame in (
        TracksFrame(4, 1_300_000_000, []),
        TracksFrame(5, 1_600_000_000, []),
        TracksFrame(6, 2_000_000_000, []),
    ):
        later_events.extend(sm.update(frame).events)

    assert any(event.name == "POCKET_CANDIDATE" for event in candidate.events)
    rejection = next(event for event in bounced.events if event.name == "POCKET_REJECTED")
    assert "projected_entry_reversed" in rejection.payload["reason_codes"]
    assert not any(event.name in {"POCKET_COMMIT_READY", "POCKET_CONFIRMED"} for event in later_events)


def test_stationary_ball_disappearing_in_entry_corridor_is_not_a_goal() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(500, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    visible = sm.update(TracksFrame(1, 1_000_000_000, [_ball(88, "stripe", 500, 55)]))
    missing = sm.update(TracksFrame(2, 1_900_000_000, []))

    assert not any(event.name == "POCKET_CANDIDATE" for event in visible.events)
    assert not any(event.name.startswith("POCKET_") for event in missing.events)


def test_legacy_confirm_missing_config_is_commit_ready_alias() -> None:
    sm = ModernMatchStateMachine(
        StateConfig(
            engine="modern",
            pocket_commit_ready_missing_ms=None,
            pocket_confirm_missing_ms=650,
        )
    )
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(0, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(1, 1_000_000_000, [_ball(2, "solid", -50, -25, -20, -10)]))
    sm.update(TracksFrame(2, 1_100_000_000, []))
    before = sm.update(TracksFrame(3, 1_749_000_000, []))
    ready = sm.update(TracksFrame(4, 1_750_000_000, []))

    assert not any(event.name == "POCKET_COMMIT_READY" for event in before.events)
    assert any(event.name == "POCKET_COMMIT_READY" for event in ready.events)


def test_commit_ready_reappearance_rolls_back_aggregated_count() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(0, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(1, 1_000_000_000, [_ball(2, "solid", -50, -25, -20, -10)]))
    sm.update(TracksFrame(2, 1_100_000_000, []))
    sm.update(TracksFrame(3, 1_400_000_000, []))
    ready = sm.update(TracksFrame(4, 1_800_000_000, []))
    assert any(event.name == "POCKET_COMMIT_READY" for event in ready.events)
    assert sm.aggregator.active is not None
    assert sm.aggregator.active.potted_confirmed["solid"] == 1

    reappeared = sm.update(TracksFrame(5, 1_850_000_000, [_ball(2, "solid", -45, -22)]))

    assert any(event.name == "POCKET_REJECTED" for event in reappeared.events)
    assert sm.aggregator.active is not None
    assert sm.aggregator.active.potted_confirmed["solid"] == 0
    assert sm.aggregator.active.committed_pockets == []


def test_operator_turn_resolve_frame_still_checks_commit_ready_reappearance() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(0, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(1, 1_000_000_000, [_ball(2, "solid", -50, -25, -20, -10)]))
    sm.update(TracksFrame(2, 1_100_000_000, []))
    sm.update(TracksFrame(3, 1_400_000_000, []))
    sm.update(TracksFrame(4, 1_800_000_000, []))
    sm.force_phase(MatchPhase.TURN_RESOLVE, frame_id=5, ts_cam_ns=1_850_000_000)

    out = sm.update(TracksFrame(5, 1_850_000_000, [_ball(2, "solid", -45, -22)]))

    assert any(event.name == "POCKET_REAPPEARED" for event in out.events)
    assert any(event.name == "POCKET_REJECTED" for event in out.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in out.events)
    assert sm.aggregator.active is None or sm.aggregator.active.potted_confirmed["solid"] == 0


def test_operator_turn_resolve_with_short_grace_cannot_bypass_reappear_window() -> None:
    sm = ModernMatchStateMachine(
        StateConfig(
            engine="modern",
            pocket_commit_ready_missing_ms=700,
            turn_resolve_grace_ms=50,
        )
    )
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(0, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(1, 1_000_000_000, [_ball(2, "solid", -50, -25, -20, -10)]))
    sm.update(TracksFrame(2, 1_100_000_000, []))
    sm.update(TracksFrame(3, 1_400_000_000, []))
    sm.force_phase(MatchPhase.TURN_RESOLVE, frame_id=4, ts_cam_ns=1_450_000_000)
    deferred = sm.update(TracksFrame(4, 1_450_000_000, [_ball(1, "cue", 100, 100)]))
    reviewed = sm.update(TracksFrame(5, 1_510_000_000, [_ball(1, "cue", 100, 100)]))

    assert any(event.name == "TURN_RESOLVE_DEFERRED" for event in deferred.events)
    assert any(event.name == "POCKET_REVIEW_REQUIRED" for event in reviewed.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in [*deferred.events, *reviewed.events])
    assert sm.debug_snapshot()["pending_review"]


def test_cue_same_track_reappearance_rolls_back_scratch() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(0, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(1, 1_000_000_000, [_ball(1, "cue", -50, -25, -20, -10)]))
    sm.update(TracksFrame(2, 1_100_000_000, []))
    sm.update(TracksFrame(3, 1_400_000_000, []))
    sm.update(TracksFrame(4, 1_800_000_000, []))
    assert sm.aggregator.active is not None and sm.aggregator.active.cue_scratch_candidate is True

    out = sm.update(TracksFrame(5, 1_850_000_000, [_ball(1, "cue", -45, -22)]))

    assert any(event.name == "POCKET_REJECTED" for event in out.events)
    assert sm.aggregator.active is not None
    assert sm.aggregator.active.cue_scratch_candidate is False
    assert sm.aggregator.active.potted_confirmed["cue"] == 0


def test_cue_new_track_reappearance_rolls_back_scratch() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(0, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(1, 1_000_000_000, [_ball(1, "cue", -50, -25, -20, -10)]))
    sm.update(TracksFrame(2, 1_100_000_000, []))
    sm.update(TracksFrame(3, 1_400_000_000, []))
    sm.update(TracksFrame(4, 1_800_000_000, []))
    out = sm.update(TracksFrame(5, 1_850_000_000, [_ball(101, "cue", -45, -22)]))

    assert any(event.name == "POCKET_REAPPEARED" for event in out.events)
    assert any(event.name == "POCKET_REJECTED" for event in out.events)
    assert sm.aggregator.active is not None
    assert sm.aggregator.active.cue_scratch_candidate is False


def test_pocket_evidence_cannot_migrate_between_pockets() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        pockets_mm=[(0, 0), (1000, 0)],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    first = sm.update(TracksFrame(1, 1_000_000_000, [_ball(2, "solid", -50, -25, -20, -10)]))
    first_candidate = next(event for event in first.events if event.name == "POCKET_CANDIDATE")
    changed = sm.update(TracksFrame(2, 1_050_000_000, [_ball(2, "solid", 1050, -25, 20, -10)]))
    rejected = next(event for event in changed.events if event.name == "POCKET_REJECTED")
    next_frame = sm.update(TracksFrame(3, 1_100_000_000, [_ball(2, "solid", 1050, -25, 20, -10)]))
    second_candidate = next(event for event in next_frame.events if event.name == "POCKET_CANDIDATE")

    assert first_candidate.payload["pocket_index"] == 0
    assert rejected.payload["pocket_index"] == 0
    assert second_candidate.payload["pocket_index"] == 1
    assert second_candidate.payload["decision_id"] != first_candidate.payload["decision_id"]


def test_rejected_and_review_events_retract_commit_ready_by_decision_id() -> None:
    aggregator = ShotContextAggregator()
    rule_state = MatchRuleState()
    commit_a = Event(
        "POCKET_COMMIT_READY",
        1_000_000_000,
        1,
        payload={"decision_id": "pocket:a", "group": "solid", "track_id": 2, "pocket_index": 0},
    )
    commit_b = Event(
        "POCKET_COMMIT_READY",
        1_000_000_000,
        1,
        payload={"decision_id": "pocket:b", "group": "stripe", "track_id": 3, "pocket_index": 1},
    )
    aggregator.ingest([commit_a, commit_b, commit_a], ts_ms=1000, rule_state=rule_state)
    assert aggregator.active is not None
    assert aggregator.active.potted_confirmed["solid"] == 1
    assert aggregator.active.potted_confirmed["stripe"] == 1

    aggregator.ingest(
        [Event("POCKET_REJECTED", 1_100_000_000, 2, payload=dict(commit_a.payload))],
        ts_ms=1100,
        rule_state=rule_state,
    )
    assert aggregator.active.potted_confirmed["solid"] == 0
    assert aggregator.active.potted_confirmed["stripe"] == 1

    aggregator.ingest(
        [Event("POCKET_REVIEW_REQUIRED", 1_200_000_000, 3, payload=dict(commit_b.payload))],
        ts_ms=1200,
        rule_state=rule_state,
    )
    assert aggregator.active.potted_confirmed["stripe"] == 0
    assert aggregator.active.committed_pockets == []


def test_invalid_geometry_cannot_create_commit_ready_from_missing_track() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_commit_ready_missing_ms=700))
    sm.set_table_context(
        inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
        ball_center_reachable_polygon_mm=[(30, 30), (970, 30), (970, 470), (30, 470)],
        pockets_mm=[],
        pocket_curves_mm=[[(100, 100), (100, 100)]],
        ball_diameter_mm=56,
    )
    sm.phase = MatchPhase.SHOT_ACTIVE

    all_events = []
    for frame in (
        TracksFrame(1, 1_000_000_000, [_ball(2, "solid", 100, 100, -20, -10)]),
        TracksFrame(2, 1_100_000_000, []),
        TracksFrame(3, 1_400_000_000, []),
        TracksFrame(4, 1_800_000_000, []),
    ):
        all_events.extend(sm.update(frame).events)

    assert sm.debug_snapshot()["pocket_geometry"]["valid"] is False
    assert not any(event.name.startswith("POCKET_") for event in all_events)


def test_referee_effective_remaining_prevents_false_black_upgrade() -> None:
    ledger = InventoryLedger()
    ledger.remaining["solid"] = 0
    rule_state = MatchRuleState(table_state="closed", actor_group="solid", opponent_group="stripe")
    shot = ShotContext(shot_id=1, ts_start_ms=0, table_state_before="closed", actor_group="solid")
    shot.potted_confirmed["solid"] = 1

    intent = RefereeAdapter().evaluate(
        shot,
        ledger,
        rule_state,
        effective_remaining={"cue": 1, "solid": 1, "stripe": 7, "black": 1},
        review_reasons=["visible_exceeds_ledger"],
    )

    assert intent.next_group_hint == "solid"
    assert intent.next_actor_changed is False
    assert intent.review_required is True


def test_observation_reconciler_uses_stable_yolo_as_effective_remaining_only() -> None:
    cfg = StateConfig(engine="modern", observation_reconcile_stable_frames=2)
    reconciler = ObservationReconciler(cfg)
    ledger = InventoryLedger()
    ledger.remaining["solid"] = 0

    frame = TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[_ball(2, "solid", 120, 120)])
    reconciler.update_observation(frame)
    reconciler.update_observation(TracksFrame(frame_id=2, ts_cam_ns=2, tracks=[_ball(2, "solid", 120, 120)]))
    result = reconciler.reconcile(ledger)

    assert result.effective_remaining["solid"] == 1
    assert ledger.remaining["solid"] == 0
    assert result.review_required is True
    assert result.mismatches[0]["mode"] == "visible_exceeds_ledger"


def test_observation_reconciler_does_not_lower_count_without_event_support() -> None:
    cfg = StateConfig(engine="modern", observation_reconcile_stable_frames=2)
    reconciler = ObservationReconciler(cfg)
    ledger = InventoryLedger()
    ledger.remaining["solid"] = 2

    reconciler.update_observation(TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[]))
    reconciler.update_observation(TracksFrame(frame_id=2, ts_cam_ns=2, tracks=[]))
    result = reconciler.reconcile(ledger)

    assert result.effective_remaining["solid"] == 2
    assert result.review_required is False


def test_observation_reconciler_can_lower_count_with_event_support() -> None:
    cfg = StateConfig(engine="modern", observation_reconcile_stable_frames=2)
    reconciler = ObservationReconciler(cfg)
    ledger = InventoryLedger()
    ledger.remaining["solid"] = 2

    reconciler.update_observation(TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[]))
    reconciler.update_observation(TracksFrame(frame_id=2, ts_cam_ns=2, tracks=[]))
    result = reconciler.reconcile(ledger, supporting_events=[Event("BALL_LOST_UNCONFIRMED", 2, 2, payload={"group": "solid"})])

    assert result.effective_remaining["solid"] == 0
    assert result.review_required is True
    assert result.mismatches[0]["mode"] == "event_supported_visible_below_ledger"


def test_modern_keeps_current_target_when_only_stable_zero_visibility_suggests_clear() -> None:
    cfg = StateConfig(engine="modern", observation_reconcile_stable_frames=2)
    sm = ModernMatchStateMachine(cfg)
    sm.set_turn_target_group("stripe")

    layout = [
        _ball(1, "cue", 100, 250),
        _ball(2, "solid", 430, 250),
        _ball(8, "black", 650, 250),
    ]
    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1, tracks=layout))
    out = sm.update(TracksFrame(frame_id=2, ts_cam_ns=2, tracks=layout))

    assert out.turn_target_group == "stripe"
    assert sm.debug_snapshot()["target_resolution"]["review_required"] is True


def test_black_commit_emits_game_status_change_only_once() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern"))
    sm.rule_state.table_state = "closed"
    sm.rule_state.actor_group = "solid"
    sm.rule_state.opponent_group = "stripe"
    sm.ledger.remaining["solid"] = 0
    sm._turn_target_group = "black"

    sm.aggregator.begin_if_needed(ts_ms=1000, rule_state=sm.rule_state)
    sm.aggregator.ingest(
        [Event("POCKET_COMMIT_READY", 1_000_000_000, 1, payload={"group": "black", "track_id": 8, "pocket_index": 0, "shot_id": 1, "decision_id": "pocket:1"})],
        ts_ms=1000,
        rule_state=sm.rule_state,
    )
    first_events = [Event("TURN_RESOLVE", 1_000_000_000, 1)]
    sm._process_turn_resolve_if_needed(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(1, "cue", 100, 100)]), first_events)

    second_events = [Event("TURN_RESOLVE", 1_100_000_000, 2)]
    sm._process_turn_resolve_if_needed(TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[_ball(1, "cue", 100, 100)]), second_events)

    assert any(event.name == "GAME_STATUS_CHANGED" for event in first_events)
    assert any(event.name == "GAME_OVER_CANDIDATE" for event in first_events)
    assert not any(event.name == "GAME_STATUS_CHANGED" for event in second_events)
    assert not any(event.name == "GAME_OVER_CANDIDATE" for event in second_events)


def test_black_visible_conflict_blocks_game_status_change() -> None:
    cfg = StateConfig(engine="modern", observation_reconcile_stable_frames=2)
    sm = ModernMatchStateMachine(cfg)
    sm.rule_state.table_state = "closed"
    sm.rule_state.actor_group = "solid"
    sm.rule_state.opponent_group = "stripe"
    sm.ledger.remaining["solid"] = 0
    sm._turn_target_group = "black"

    visible_black = TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(8, "black", 480, 220)])
    sm.reconciler.update_observation(visible_black)
    sm.reconciler.update_observation(TracksFrame(frame_id=2, ts_cam_ns=1_050_000_000, tracks=[_ball(8, "black", 480, 220)]))
    sm.aggregator.begin_if_needed(ts_ms=1000, rule_state=sm.rule_state)
    sm.aggregator.ingest(
        [Event("POCKET_COMMIT_READY", 1_000_000_000, 1, payload={"group": "black", "track_id": 8, "pocket_index": 0, "shot_id": 1, "decision_id": "pocket:1"})],
        ts_ms=1000,
        rule_state=sm.rule_state,
    )

    events = [Event("TURN_RESOLVE", 1_100_000_000, 3)]
    sm._process_turn_resolve_if_needed(
        TracksFrame(frame_id=3, ts_cam_ns=1_100_000_000, tracks=[_ball(8, "black", 480, 220)]),
        events,
    )

    assert any(event.name == "LEDGER_OBSERVATION_MISMATCH" for event in events)
    assert not any(event.name == "GAME_STATUS_CHANGED" for event in events)
    assert sm.rule_state.game_status == "in_progress"
    assert sm.ledger.remaining["black"] == 1


def test_review_required_turn_keeps_previous_committed_state() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern"))
    sm.rule_state.table_state = "closed"
    sm.rule_state.actor_group = "solid"
    sm.rule_state.opponent_group = "stripe"
    sm.set_turn_target_group("solid")

    sm.aggregator.begin_if_needed(ts_ms=1000, rule_state=sm.rule_state)
    sm.aggregator.ingest(
        [
            Event(
                "POCKET_REVIEW_REQUIRED",
                1_000_000_000,
                1,
                payload={
                    "group": "solid",
                    "track_id": 2,
                    "pocket_index": 0,
                    "shot_id": 1,
                    "decision_id": "pocket:review",
                    "reason_codes": ["mouth_rest_disappear_requires_review"],
                },
            )
        ],
        ts_ms=1000,
        rule_state=sm.rule_state,
    )
    events = [Event("TURN_RESOLVE", 1_000_000_000, 1)]
    sm._process_turn_resolve_if_needed(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(1, "cue", 100, 100)]), events)

    intent = next(event for event in events if event.name == "REFEREE_INTENT")
    assert intent.payload["review_required"] is True
    assert intent.payload["next_actor_changed"] is False
    assert intent.payload["next_group_hint"] == "solid"
    assert sm.rule_state.actor_group == "solid"
    assert sm.rule_state.opponent_group == "stripe"
    assert sm.turn_target_group == "solid"


def test_confirm_episode_commits_frozen_reviewed_pocket() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern"))
    sm.rule_state.table_state = "closed"
    sm.rule_state.actor_group = "solid"
    sm.rule_state.opponent_group = "stripe"
    sm.set_turn_target_group("solid")

    sm.aggregator.begin_if_needed(ts_ms=1000, rule_state=sm.rule_state)
    sm.aggregator.ingest(
        [
            Event(
                "POCKET_REVIEW_REQUIRED",
                1_000_000_000,
                1,
                payload={
                    "group": "solid",
                    "track_id": 2,
                    "pocket_index": 0,
                    "shot_id": 1,
                    "decision_id": "pocket:review",
                    "reason_codes": ["mouth_rest_disappear_requires_review"],
                },
            )
        ],
        ts_ms=1000,
        rule_state=sm.rule_state,
    )
    events = [Event("TURN_RESOLVE", 1_000_000_000, 1)]
    sm._process_turn_resolve_if_needed(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(1, "cue", 100, 100)]), events)

    assert sm.debug_snapshot()["pending_review"]["decision_ids"] == ["pocket:review"]

    confirmed = sm.confirm_episode(frame_id=2, ts_cam_ns=1_100_000_000, reason="test")

    assert confirmed is True
    assert sm.debug_snapshot()["pending_review"] == {}
    assert sm.ledger.remaining["solid"] == 6
    assert sm.rule_state.shot_number == 1
    assert sm.rule_state.actor_group == "solid"


def test_reject_episode_discards_frozen_reviewed_pocket() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern"))
    sm.rule_state.table_state = "closed"
    sm.rule_state.actor_group = "solid"
    sm.rule_state.opponent_group = "stripe"
    sm.set_turn_target_group("solid")

    sm.aggregator.begin_if_needed(ts_ms=1000, rule_state=sm.rule_state)
    sm.aggregator.ingest(
        [
            Event(
                "POCKET_REVIEW_REQUIRED",
                1_000_000_000,
                1,
                payload={
                    "group": "solid",
                    "track_id": 2,
                    "pocket_index": 0,
                    "shot_id": 1,
                    "decision_id": "pocket:review",
                    "reason_codes": ["mouth_rest_disappear_requires_review"],
                },
            )
        ],
        ts_ms=1000,
        rule_state=sm.rule_state,
    )
    events = [Event("TURN_RESOLVE", 1_000_000_000, 1)]
    sm._process_turn_resolve_if_needed(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(1, "cue", 100, 100)]), events)

    rejected = sm.reject_episode(frame_id=2, ts_cam_ns=1_100_000_000, reason="test")

    assert rejected is True
    assert sm.debug_snapshot()["pending_review"] == {}
    assert sm.ledger.remaining["solid"] == 7
    assert sm.rule_state.shot_number == 1
    assert sm.rule_state.actor_group == "stripe"


def test_resolve_open_table_group_commits_group_choice() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern"))
    sm.rule_state.table_state = "open"
    sm.rule_state.shot_number = 1

    sm.aggregator.begin_if_needed(ts_ms=1000, rule_state=sm.rule_state)
    sm.aggregator.ingest(
        [
            Event(
                "POCKET_COMMIT_READY",
                1_000_000_000,
                1,
                payload={"group": "solid", "track_id": 2, "pocket_index": 0, "shot_id": 1, "decision_id": "pocket:solid"},
            ),
            Event(
                "POCKET_COMMIT_READY",
                1_000_000_000,
                1,
                payload={"group": "stripe", "track_id": 9, "pocket_index": 1, "shot_id": 1, "decision_id": "pocket:stripe"},
            ),
        ],
        ts_ms=1000,
        rule_state=sm.rule_state,
    )
    events = [Event("TURN_RESOLVE", 1_000_000_000, 1)]
    sm._process_turn_resolve_if_needed(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(1, "cue", 100, 100)]), events)

    pending = sm.debug_snapshot()["pending_review"]
    assert pending["group_choice_required"] is True

    resolved = sm.resolve_open_table_group("solid", frame_id=2, ts_cam_ns=1_100_000_000, reason="test")

    assert resolved is True
    assert sm.debug_snapshot()["pending_review"] == {}
    assert sm.rule_state.table_state == "closed"
    assert sm.rule_state.actor_group == "solid"
    assert sm.rule_state.opponent_group == "stripe"
    assert sm.rule_state.shot_number == 2
    assert sm.ledger.remaining["solid"] == 6
    assert sm.ledger.remaining["stripe"] == 6
