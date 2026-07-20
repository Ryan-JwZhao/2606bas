from __future__ import annotations

from scripts.evaluate_pocket_video import _evaluate


def test_goal_matching_uses_contact_frame_when_shot_phase_has_false_starts() -> None:
    labels = {
        "max_notice_delay_ms": 1500,
        "max_match_frame_gap": 90,
        "goals": [
            {
                "shot": 4,
                "contact_frame": 750,
                "pocket_index": 2,
                "group": "stripe",
                "ball_code": "stb",
            }
        ],
        "no_goals": [{"shot": 3, "reason": "black ball remained on the lip"}],
    }
    detections = [
        {
            "shot": 6,
            "frame_id": 774,
            "group": "stripe",
            "ball_code": "stb",
            "pocket_index": 2,
            "decision_id": "pocket:9",
            "decision_latency_ms": 1328,
        }
    ]

    result = _evaluate(labels, detections, [1.0], [80.0])

    assert result["passed"] is True
    assert result["matched_goals"] == 1
    assert result["false_positives"] == []


def test_no_goal_shot_still_rejects_unmatched_detection() -> None:
    labels = {"goals": [], "no_goals": [{"shot": 3, "reason": "lip stop"}]}
    detection = {
        "shot": 3,
        "frame_id": 430,
        "group": "black",
        "pocket_index": 0,
        "decision_latency_ms": 1300,
    }

    result = _evaluate(labels, [detection], [1.0], [80.0])

    assert result["passed"] is False
    assert result["false_positives"] == [detection]
