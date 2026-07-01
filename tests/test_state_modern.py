from __future__ import annotations

import json

from bas.config import AppConfig, StateConfig
from bas.schemas import Event, MatchPhase, TrackObservation, TracksFrame
from bas.state import LegacyMatchStateMachine, ModernMatchStateMachine, create_match_state_machine
from bas.state.models import InventoryLedger, MatchRuleState, ShotContext
from bas.state.reconcile import ObservationReconciler
from bas.state.referee import RefereeAdapter
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
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_confirm_missing_ms=300))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", 90, 0)]))
    tentative = sm.update(TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[]))
    reviewed = sm.update(TracksFrame(frame_id=3, ts_cam_ns=1_400_000_000, tracks=[]))

    assert any(event.name == "POCKET_TENTATIVE" for event in tentative.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in reviewed.events)
    review = next(event for event in reviewed.events if event.name == "POCKET_REVIEW_REQUIRED")
    assert review.payload["review_required"] is True
    assert "mouth_rest_disappear_requires_review" in review.payload["reason_codes"]


def test_modern_pocket_confirmation_waits_for_missing_window() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_confirm_missing_ms=300))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    first = sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", 35, 35, -20, -20)]))
    early = sm.update(TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[]))
    confirmed = sm.update(TracksFrame(frame_id=3, ts_cam_ns=1_400_000_000, tracks=[]))

    assert any(event.name == "POCKET_CANDIDATE" for event in first.events)
    assert any(event.name == "POCKET_TENTATIVE" for event in early.events)
    assert any(event.name == "POCKET_CONFIRMED" for event in confirmed.events)
    assert any(event.name == "POT_PROBABLE" for event in confirmed.events)


def test_modern_pocket_reappearing_with_new_track_is_rejected() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_confirm_missing_ms=300))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", 35, 35, -20, -20)]))
    sm.update(TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[]))
    out = sm.update(TracksFrame(frame_id=3, ts_cam_ns=1_180_000_000, tracks=[_ball(99, "solid", 42, 42)]))

    assert any(event.name == "POCKET_REAPPEARED" for event in out.events)
    assert any(event.name == "POCKET_REJECTED" for event in out.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in out.events)


def test_occluded_frames_without_true_loss_do_not_confirm() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_confirm_missing_ms=300))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", 35, 35, -20, -20)]))
    first_occluded = sm.update(
        TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[_ball(2, "solid", 35, 35, -20, -20, visibility="occluded", lost_frames=1)])
    )
    second_occluded = sm.update(
        TracksFrame(frame_id=3, ts_cam_ns=1_300_000_000, tracks=[_ball(2, "solid", 35, 35, -20, -20, visibility="occluded", lost_frames=2)])
    )

    assert any(event.name == "POCKET_TENTATIVE" for event in first_occluded.events)
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
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_confirm_missing_ms=300, turn_resolve_grace_ms=900))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", 35, 35, -20, -20)]))
    resolve_frame = TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[])
    deferred_events = [Event("TURN_RESOLVE", 1_100_000_000, 2)]
    deferred_events.extend(sm.pocket_fsm.update(resolve_frame, MatchPhase.TURN_RESOLVE))
    sm._annotate_shot_ids(deferred_events, ts_ms=1100)
    sm.aggregator.ingest(deferred_events, ts_ms=1100, rule_state=sm.rule_state)
    sm._process_turn_resolve_if_needed(resolve_frame, deferred_events)

    assert any(event.name == "TURN_RESOLVE_DEFERRED" for event in deferred_events)
    assert not any(event.name == "REFEREE_INTENT" for event in deferred_events)

    out = sm.update(TracksFrame(frame_id=3, ts_cam_ns=1_400_000_000, tracks=[]))

    assert any(event.name == "POCKET_CONFIRMED" for event in out.events)
    assert any(event.name == "TURN_RESOLVE_COMMITTED" for event in out.events)
    assert any(event.name == "REFEREE_INTENT" for event in out.events)
    assert sm.ledger.remaining["solid"] == 6


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
        [Event("POCKET_CONFIRMED", 1_000_000_000, 1, payload={"group": "black", "track_id": 8, "pocket_index": 0, "shot_id": 1, "decision_id": "pocket:1"})],
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
