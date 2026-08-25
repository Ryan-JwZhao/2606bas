from __future__ import annotations

from bas.schemas import Event
from bas.web_control.pocket_notice import PocketNoticeTracker


def test_pocket_notice_tracker_expires_notices_and_keeps_sequences_monotonic_after_reset() -> None:
    tracker = PocketNoticeTracker(retention_s=1.0)
    first = Event(
        "POCKET_CONFIRMED",
        1,
        1,
        payload={"decision_id": "pocket:first", "group": "solid", "track_id": 2, "pocket_index": 0},
    )
    second = Event(
        "POCKET_CONFIRMED",
        2,
        2,
        payload={"decision_id": "pocket:second", "group": "black", "track_id": 8, "pocket_index": 1},
    )

    created = tracker.observe([first, first], now_s=10.0)

    assert len(created) == 1
    assert created[0]["ball_code"] == "sob"
    assert tracker.current(now_s=10.999) == created
    assert tracker.current(now_s=11.0) == []

    tracker.reset()
    after_reset = tracker.observe([second], now_s=20.0)

    assert after_reset[0]["ball_code"] == "bb"
    assert after_reset[0]["sequence"] > created[0]["sequence"]
    assert after_reset[0]["notice_id"] != created[0]["notice_id"]


def test_pocket_notice_tracker_waits_for_confirmation_and_deduplicates_aliases() -> None:
    tracker = PocketNoticeTracker()
    legacy = Event(
        "POT_PROBABLE",
        1,
        1,
        payload={"group": "stripe", "track_id": 9, "pocket_index": 0},
    )
    modern_detected = Event(
        "POCKET_DETECTED",
        2,
        2,
        payload={"decision_id": "pocket:solid", "group": "solid", "track_id": 2, "pocket_index": 1},
    )
    modern_confirmed = Event(
        "POCKET_CONFIRMED",
        3,
        3,
        payload=dict(modern_detected.payload),
    )
    modern_alias = Event(
        "POT_PROBABLE",
        3,
        3,
        payload=dict(modern_confirmed.payload),
    )

    detected = tracker.observe([modern_detected], now_s=9.0)

    created = tracker.observe([legacy, modern_confirmed, modern_alias], now_s=10.0)

    assert detected == []
    assert [notice["message"] for notice in created] == ["花色球进洞", "全色球进洞"]


def test_pocket_notice_waits_for_irreversible_confirmation() -> None:
    tracker = PocketNoticeTracker()
    detected = Event(
        "POCKET_DETECTED",
        1,
        1,
        payload={"decision_id": "pocket:provisional", "group": "solid", "track_id": 2, "pocket_index": 3},
    )
    confirmed = Event(
        "POCKET_CONFIRMED",
        2,
        2,
        payload=dict(detected.payload),
    )

    assert tracker.observe([detected], now_s=1.0) == []
    created = tracker.observe([confirmed], now_s=2.0)

    assert [notice["message"] for notice in created] == ["全色球进洞"]
