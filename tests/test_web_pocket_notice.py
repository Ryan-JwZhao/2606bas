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


def test_pocket_notice_tracker_announces_detection_and_deduplicates_confirmation_aliases() -> None:
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

    assert [notice["message"] for notice in detected] == ["全色球进洞"]
    assert [notice["message"] for notice in created] == ["花色球进洞"]


def test_confirmation_alias_does_not_repeat_an_immediate_detection_notice() -> None:
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

    created = tracker.observe([detected], now_s=1.0)
    confirmed_alias = tracker.observe([confirmed], now_s=2.0)

    assert [notice["message"] for notice in created] == ["全色球进洞"]
    assert confirmed_alias == []


def test_pocket_notice_announces_detection_and_reports_later_retraction() -> None:
    tracker = PocketNoticeTracker()
    detected = Event(
        "POCKET_DETECTED",
        1,
        1,
        payload={"decision_id": "pocket:retractable", "group": "stripe", "track_id": 9, "pocket_index": 3},
    )
    rejected = Event(
        "POCKET_REJECTED",
        2,
        2,
        payload={**detected.payload, "reason_codes": ["reappeared_same_track"]},
    )

    created = tracker.observe([detected], now_s=1.0)
    retracted = tracker.observe([rejected], now_s=2.0)

    assert created[0]["status"] == "detected"
    assert created[0]["message"] == "花色球进洞"
    assert retracted[0]["status"] == "rejected"
    assert retracted[0]["message"] == "花色球进洞判定撤销"
    assert tracker.current(now_s=2.0) == retracted
