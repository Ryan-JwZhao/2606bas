from __future__ import annotations

from pathlib import Path

from bas.state.replay import run_pocket_trace


FIXTURES = Path(__file__).parent / "fixtures"


def test_pocket_trace_replay_flags_mouth_review_case() -> None:
    result = run_pocket_trace(FIXTURES / "pocket_trace_mouth_review.json")

    names = [event["name"] for event in result["events"]]
    assert names[:2] == ["POCKET_TENTATIVE", "POCKET_REVIEW_REQUIRED"]
    assert "POCKET_CONFIRMED" not in names
    assert result["last_referee_intent"]["review_required"] is True


def test_pocket_trace_replay_confirms_real_pocket_case() -> None:
    result = run_pocket_trace(FIXTURES / "pocket_trace_true_confirm.json")

    names = [event["name"] for event in result["events"]]
    assert names[:3] == ["POCKET_CANDIDATE", "POCKET_TENTATIVE", "POCKET_CONFIRMED"]
    assert result["last_shot_context"]["potted_confirmed"]["solid"] == 1
    assert result["ledger_remaining"]["solid"] == 6


def test_pocket_trace_replay_emits_single_victory_transition() -> None:
    result = run_pocket_trace(FIXTURES / "pocket_trace_black_victory_once.json")

    names = [event["name"] for event in result["events"]]
    assert names.count("GAME_STATUS_CHANGED") == 1
    assert names.count("GAME_OVER_CANDIDATE") == 1
    assert result["rule_state"]["game_status"] == "ended_pending_review"
