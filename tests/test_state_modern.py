from __future__ import annotations

import json

from bas.config import AppConfig, StateConfig
from bas.schemas import Event, MatchPhase, TrackObservation, TracksFrame
from bas.state import LegacyMatchStateMachine, ModernMatchStateMachine, create_match_state_machine
from bas.state.models import InventoryLedger, MatchRuleState, ShotContext
from bas.state.referee import RefereeAdapter
from bas.user_settings import UserSettings


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


def test_modern_pocket_confirmation_waits_for_missing_window() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_confirm_missing_ms=300))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    first = sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", 35, 35, -20, -20)]))
    early = sm.update(TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[]))
    confirmed = sm.update(TracksFrame(frame_id=3, ts_cam_ns=1_400_000_000, tracks=[]))

    assert any(event.name == "POCKET_CANDIDATE" for event in first.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in early.events)
    assert any(event.name == "POCKET_CONFIRMED" for event in confirmed.events)
    assert any(event.name == "POT_PROBABLE" for event in confirmed.events)


def test_modern_pocket_candidate_reappearing_is_not_counted() -> None:
    sm = ModernMatchStateMachine(StateConfig(engine="modern", pocket_confirm_missing_ms=300))
    sm.set_table_context(inner_polygon_mm=[(0, 0), (1000, 0), (1000, 500), (0, 500)], pockets_mm=[(0, 0)], ball_diameter_mm=56)
    sm.phase = MatchPhase.SHOT_ACTIVE

    sm.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(2, "solid", 35, 35, -20, -20)]))
    sm.update(TracksFrame(frame_id=2, ts_cam_ns=1_100_000_000, tracks=[]))
    out = sm.update(TracksFrame(frame_id=3, ts_cam_ns=1_180_000_000, tracks=[_ball(2, "solid", 160, 160, 0, 0)]))

    assert any(event.name == "POCKET_REAPPEARED" for event in out.events)
    assert not any(event.name == "POCKET_CONFIRMED" for event in out.events)


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
