from __future__ import annotations

from bas.config import StateConfig
from bas.schemas import Event, TrackObservation, TracksFrame
from bas.state.models import InventoryLedger, MatchRuleState, ShotContext
from bas.state.modern import ModernMatchStateMachine
from bas.state.reconcile import ObservationReconciler
from bas.state.referee import RefereeAdapter, ShotContextAggregator
from bas.state.targeting import resolve_turn_target_group


def _ball(
    track_id: int,
    group: str,
    *,
    quality: float = 0.95,
    visibility: str = "visible",
    velocity_mm_s: tuple[float, float] = (0.0, 0.0),
    lost_frames: int = 0,
    confirmed: bool = True,
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(0.0, 0.0, 10.0, 10.0),
        center_px=(5.0, 5.0),
        center_mm=(100.0 + track_id * 60.0, 300.0),
        radius_px=5.0,
        radius_mm=28.575,
        cls_name=group,
        group=group,
        confidence=0.95,
        quality=quality,
        confirmed=confirmed,
        visibility=visibility,
        velocity_mm_s=velocity_mm_s,
        lost_frames=lost_frames,
    )


def _commit(sm: ModernMatchStateMachine, *, frame_id: int = 10) -> list[Event]:
    events = [Event("TURN_RESOLVE", frame_id * 100_000_000, frame_id)]
    sm._process_turn_resolve_if_needed(
        TracksFrame(frame_id, frame_id * 100_000_000, [_ball(1, "cue")]),
        events,
    )
    return events


def _ingest_single_solid_pot(sm: ModernMatchStateMachine) -> None:
    sm.aggregator.ingest(
        [
            Event("SHOT_STARTED", 100_000_000, 1),
            Event(
                "POCKET_COMMIT_READY",
                500_000_000,
                5,
                payload={"group": "solid", "track_id": 2, "decision_id": "pocket:solid"},
            ),
        ],
        ts_ms=500,
        rule_state=sm.rule_state,
    )


def test_first_observed_midgame_shot_is_not_implicitly_a_break() -> None:
    sm = ModernMatchStateMachine(
        StateConfig(engine="modern", observation_reconcile_enabled=False, turn_resolve_grace_ms=0)
    )

    _ingest_single_solid_pot(sm)
    _commit(sm)

    assert sm.rule_state.actor_group == "solid"
    assert sm.rule_state.shot_number == 1


def test_explicit_break_shot_keeps_open_table_open() -> None:
    sm = ModernMatchStateMachine(
        StateConfig(engine="modern", observation_reconcile_enabled=False, turn_resolve_grace_ms=0)
    )
    sm.rule_state.active_shot_is_break = True

    _ingest_single_solid_pot(sm)
    _commit(sm)

    assert sm.rule_state.table_state == "open"
    assert sm.rule_state.actor_group is None


def test_empty_turn_resolve_does_not_create_a_shot_or_advance_number() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", turn_resolve_grace_ms=0))

    events = _commit(sm, frame_id=1)

    assert sm.rule_state.shot_number == 0
    assert any(event.name == "TURN_RESOLVE_IGNORED" for event in events)
    assert not any(event.name == "REFEREE_INTENT" for event in events)


def test_low_quality_occluded_cue_motion_cannot_start_a_shot() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern"))
    stale_cue = _ball(
        1,
        "cue",
        quality=0.01,
        visibility="occluded",
        velocity_mm_s=(100.0, 0.0),
        lost_frames=8,
    )

    out = sm.update(TracksFrame(1, 100_000_000, [stale_cue]))

    assert out.phase == "STABLE_IDLE"
    assert not any(event.name == "SHOT_STARTED" for event in out.events)


def test_unconfirmed_moving_detection_cannot_start_a_shot() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern"))

    for frame_id in range(1, 4):
        out = sm.update(
            TracksFrame(
                frame_id,
                frame_id * 100_000_000,
                [_ball(1, "cue", confirmed=False, velocity_mm_s=(500.0, 0.0))],
            )
        )

    assert out.phase == "STABLE_IDLE"
    assert not any(event.name == "SHOT_STARTED" for event in out.events)


def test_runtime_state_round_trip_preserves_rules_ledger_and_break_lifecycle() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern"))
    sm.rule_state.table_state = "closed"
    sm.rule_state.actor_group = "stripe"
    sm.rule_state.opponent_group = "solid"
    sm.rule_state.shot_number = 4
    sm.ledger.remaining.update({"solid": 5, "stripe": 2, "black": 1})
    sm.ledger.removed_confirmed.update({"solid": 2, "stripe": 5, "black": 0})
    sm.set_turn_target_group("stripe", reason="test")
    sm.break_lifecycle.mark_shot_started()
    sm.break_lifecycle.mark_shot_resolved(valid=True)

    restored = ModernMatchStateMachine(StateConfig(engine="modern"))
    restored.restore_runtime_state(sm.export_runtime_state())

    assert restored.rule_state.table_state == "closed"
    assert restored.rule_state.actor_group == "stripe"
    assert restored.rule_state.opponent_group == "solid"
    assert restored.rule_state.shot_number == 4
    assert restored.ledger.remaining == sm.ledger.remaining
    assert restored.ledger.removed_confirmed == sm.ledger.removed_confirmed
    assert restored.turn_target_group == "stripe"
    assert restored.break_lifecycle.completed is True


def test_rejected_pocket_cannot_clear_group_in_effective_view() -> None:
    config = StateConfig(
        engine="modern",
        observation_reconcile_enabled=True,
        observation_reconcile_stable_frames=3,
    )
    reconciler = ObservationReconciler(config)
    ledger = InventoryLedger()
    ledger.remaining["solid"] = 1
    for frame_id in range(1, 4):
        reconciler.update_observation(
            TracksFrame(frame_id, frame_id * 100_000_000, [_ball(1, "cue"), _ball(9, "stripe"), _ball(8, "black")])
        )

    result = reconciler.reconcile(
        ledger,
        supporting_events=[
            Event(
                "POCKET_CANDIDATE",
                300_000_000,
                3,
                payload={"group": "solid", "decision_id": "pocket:solid"},
            ),
            Event(
                "POCKET_REJECTED",
                400_000_000,
                4,
                payload={"group": "solid", "decision_id": "pocket:solid"},
            ),
        ],
    )
    target = resolve_turn_target_group(
        "solid",
        actor_group="solid",
        ledger_remaining=ledger.remaining,
        observation_effective_remaining=result.effective_remaining,
        visible_counts=result.visible_counts,
        stable_frames=result.stable_frames,
        stable_frames_required=3,
    )

    assert result.effective_remaining["solid"] == 1
    assert target.target_group == "solid"


def test_rejection_cancels_prior_commit_support_for_same_decision() -> None:
    reconciler = ObservationReconciler(
        StateConfig(engine="modern", observation_reconcile_stable_frames=2)
    )
    ledger = InventoryLedger()
    ledger.remaining["solid"] = 1
    reconciler.update_observation(TracksFrame(1, 100_000_000, [_ball(1, "cue")]))
    reconciler.update_observation(TracksFrame(2, 200_000_000, [_ball(1, "cue")]))

    result = reconciler.reconcile(
        ledger,
        supporting_events=[
            Event(
                "POCKET_COMMIT_READY",
                150_000_000,
                1,
                payload={"group": "solid", "decision_id": "same-decision"},
            ),
            Event(
                "POCKET_REJECTED",
                200_000_000,
                2,
                payload={"group": "solid", "decision_id": "same-decision"},
            ),
        ],
    )

    assert result.effective_remaining["solid"] == 1
    assert result.mismatches == []


def test_prior_shot_pocket_event_cannot_change_current_shot_target() -> None:
    sm = ModernMatchStateMachine(
        StateConfig(
            engine="modern",
            observation_reconcile_enabled=True,
            observation_reconcile_stable_frames=3,
            turn_resolve_grace_ms=0,
        )
    )
    sm.rule_state.table_state = "closed"
    sm.rule_state.actor_group = "stripe"
    sm.rule_state.opponent_group = "solid"
    sm.ledger.remaining["solid"] = 1
    sm.set_turn_target_group("stripe", reason="test")
    for frame_id in range(1, 4):
        sm.reconciler.update_observation(
            TracksFrame(frame_id, frame_id * 100_000_000, [_ball(1, "cue"), _ball(8, "black")])
        )
    # Simulate the legacy cross-shot buffer.  Current reconciliation must not
    # read this attribute even if an old runtime/test harness attaches it.
    sm._recent_events = [  # type: ignore[attr-defined]
        Event(
            "POCKET_CONFIRMED",
            50_000_000,
            0,
            payload={"group": "solid", "decision_id": "prior-shot", "shot_id": 1},
        )
    ]
    sm.aggregator.ingest(
        [
            Event("SHOT_STARTED", 400_000_000, 4),
            Event(
                "BALL_COLLISION_CANDIDATE",
                410_000_000,
                4,
                payload={"group_a": "cue", "group_b": "stripe", "track_a": 1, "track_b": 9},
            ),
            Event(
                "RAIL_COLLISION_CANDIDATE",
                420_000_000,
                4,
                payload={"group": "stripe", "track_id": 9},
            ),
        ],
        ts_ms=420,
        rule_state=sm.rule_state,
    )

    _commit(sm, frame_id=5)

    assert sm.rule_state.actor_group == "solid"
    assert sm.turn_target_group == "solid"


def test_nonbreak_contact_without_pot_or_rail_is_a_foul() -> None:
    ledger = InventoryLedger()
    rule_state = MatchRuleState(table_state="closed", actor_group="solid", opponent_group="stripe")
    shot = ShotContext(shot_id=1, ts_start_ms=0, actor_group="solid", first_contact_group="solid")

    intent = RefereeAdapter().evaluate(shot, ledger, rule_state)

    assert intent.foul_flags["no_rail_after_contact"] is True
    assert intent.ball_in_hand_scope == "table_anywhere"


def test_break_rail_count_excludes_cue_ball_contact() -> None:
    rule_state = MatchRuleState(active_shot_is_break=True)
    aggregator = ShotContextAggregator()
    events = [
        Event("SHOT_STARTED", 100_000_000, 1),
        Event("BALL_COLLISION_CANDIDATE", 120_000_000, 2, payload={"group_a": "cue", "group_b": "solid"}),
        Event("RAIL_COLLISION_CANDIDATE", 130_000_000, 3, payload={"track_id": 1, "group": "cue"}),
    ]
    events.extend(
        Event(
            "RAIL_COLLISION_CANDIDATE",
            140_000_000 + track_id,
            4,
            payload={"track_id": track_id, "group": "solid"},
        )
        for track_id in (2, 3, 4)
    )
    aggregator.ingest(events, ts_ms=140, rule_state=rule_state)

    shot = aggregator.finalize(ts_ms=500, rule_state=rule_state)
    assert shot is not None
    assert shot.rail_contact_track_ids == {2, 3, 4}
    intent = RefereeAdapter().evaluate(shot, InventoryLedger(), rule_state)
    assert intent.foul_flags["break_foul"] is True


def test_rail_before_first_contact_does_not_satisfy_nonbreak_rail_rule() -> None:
    rule_state = MatchRuleState(table_state="closed", actor_group="solid", opponent_group="stripe")
    aggregator = ShotContextAggregator()
    aggregator.ingest(
        [
            Event("SHOT_STARTED", 100_000_000, 1),
            Event("RAIL_COLLISION_CANDIDATE", 120_000_000, 2, payload={"track_id": 2, "group": "solid"}),
            Event("BALL_COLLISION_CANDIDATE", 150_000_000, 3, payload={"group_a": "cue", "group_b": "solid"}),
        ],
        ts_ms=150,
        rule_state=rule_state,
    )

    shot = aggregator.finalize(ts_ms=500, rule_state=rule_state)
    assert shot is not None
    assert shot.rail_contact_seen is True
    assert shot.rail_contact_after_first_seen is False
    intent = RefereeAdapter().evaluate(shot, InventoryLedger(), rule_state)
    assert intent.foul_flags["no_rail_after_contact"] is True


def test_rail_after_first_contact_satisfies_nonbreak_rail_rule() -> None:
    rule_state = MatchRuleState(table_state="closed", actor_group="solid", opponent_group="stripe")
    aggregator = ShotContextAggregator()
    aggregator.ingest(
        [
            Event("SHOT_STARTED", 100_000_000, 1),
            Event("BALL_COLLISION_CANDIDATE", 120_000_000, 2, payload={"group_a": "cue", "group_b": "solid"}),
            Event("RAIL_COLLISION_CANDIDATE", 150_000_000, 3, payload={"track_id": 2, "group": "solid"}),
        ],
        ts_ms=150,
        rule_state=rule_state,
    )

    shot = aggregator.finalize(ts_ms=500, rule_state=rule_state)
    assert shot is not None
    assert shot.rail_contact_after_first_seen is True
    intent = RefereeAdapter().evaluate(shot, InventoryLedger(), rule_state)
    assert intent.foul_flags["no_rail_after_contact"] is False


def test_open_table_nonbreak_cannot_contact_black_first() -> None:
    ledger = InventoryLedger()
    rule_state = MatchRuleState(table_state="open")
    shot = ShotContext(shot_id=1, ts_start_ms=0, break_shot=False, first_contact_group="black")

    intent = RefereeAdapter().evaluate(shot, ledger, rule_state)

    assert intent.foul_flags["wrong_first_contact"] is True


def test_open_table_legal_pot_can_assign_a_group() -> None:
    ledger = InventoryLedger()
    rule_state = MatchRuleState(table_state="open")
    shot = ShotContext(shot_id=1, ts_start_ms=0, break_shot=False, first_contact_group="solid")
    shot.potted_confirmed["solid"] = 1
    ledger.apply(shot)

    intent = RefereeAdapter().evaluate(shot, ledger, rule_state)

    assert intent.foul_flags["no_rail_after_contact"] is False
    assert intent.table_state_after == "closed"
    assert intent.actor_group_after == "solid"


def test_open_table_early_black_ends_without_assigning_a_group() -> None:
    ledger = InventoryLedger()
    rule_state = MatchRuleState(table_state="open")
    shot = ShotContext(shot_id=1, ts_start_ms=0, break_shot=False, first_contact_group="black")
    shot.potted_confirmed["black"] = 1
    ledger.apply(shot)

    intent = RefereeAdapter().evaluate(shot, ledger, rule_state)

    assert intent.game_outcome == "illegal_black_loss"
    assert intent.next_group_hint is None
    assert intent.table_state_after == "open"


def test_illegal_early_black_records_loss_outcome() -> None:
    ledger = InventoryLedger()
    rule_state = MatchRuleState(table_state="closed", actor_group="solid", opponent_group="stripe")
    shot = ShotContext(shot_id=1, ts_start_ms=0, actor_group="solid", first_contact_group="black")
    shot.potted_confirmed["black"] = 1
    ledger.apply(shot)

    intent = RefereeAdapter().evaluate(shot, ledger, rule_state)

    assert intent.game_status == "ended"
    assert intent.game_outcome == "illegal_black_loss"
    assert intent.winner_group == "stripe"


def test_potting_last_group_ball_uses_pre_shot_target_for_first_contact() -> None:
    before = InventoryLedger()
    before.remaining["solid"] = 1
    rule_state = MatchRuleState(table_state="closed", actor_group="solid", opponent_group="stripe")
    shot = ShotContext(
        shot_id=1,
        ts_start_ms=0,
        actor_group="solid",
        legal_first_group="solid",
        first_contact_group="solid",
    )
    shot.potted_confirmed["solid"] = 1
    after = before.applied_copy(shot)

    intent = RefereeAdapter().evaluate(
        shot,
        after,
        rule_state,
        ledger_before=before,
        legal_first_group=shot.legal_first_group,
    )

    assert intent.foul_flags["wrong_first_contact"] is False
    assert intent.next_group_hint == "black"
    assert intent.actor_group_after == "solid"


def test_potting_last_group_ball_and_black_together_is_not_a_legal_black_win() -> None:
    before = InventoryLedger()
    before.remaining["solid"] = 1
    rule_state = MatchRuleState(table_state="closed", actor_group="solid", opponent_group="stripe")
    shot = ShotContext(
        shot_id=1,
        ts_start_ms=0,
        actor_group="solid",
        legal_first_group="solid",
        first_contact_group="solid",
    )
    shot.potted_confirmed.update({"solid": 1, "black": 1})
    after = before.applied_copy(shot)

    intent = RefereeAdapter().evaluate(
        shot,
        after,
        rule_state,
        ledger_before=before,
        legal_first_group=shot.legal_first_group,
    )

    assert intent.game_outcome == "illegal_black_loss"
    assert intent.winner_group == "stripe"


def test_black_is_a_legal_win_only_when_it_was_the_pre_shot_target() -> None:
    before = InventoryLedger()
    before.remaining["solid"] = 0
    rule_state = MatchRuleState(table_state="closed", actor_group="solid", opponent_group="stripe")
    shot = ShotContext(
        shot_id=1,
        ts_start_ms=0,
        actor_group="solid",
        legal_first_group="black",
        first_contact_group="black",
    )
    shot.potted_confirmed["black"] = 1
    after = before.applied_copy(shot)

    intent = RefereeAdapter().evaluate(
        shot,
        after,
        rule_state,
        ledger_before=before,
        legal_first_group=shot.legal_first_group,
    )

    assert intent.foul_flags["wrong_first_contact"] is False
    assert intent.game_outcome == "legal_black_win"
    assert intent.winner_group == "solid"
