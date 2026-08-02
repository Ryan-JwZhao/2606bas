from __future__ import annotations

import json

import numpy as np

from bas.calibration.audit import load_calibration_audit_summaries, start_calibration_audit


def test_calibration_audit_writes_durable_events_and_summary(tmp_path) -> None:
    audit = start_calibration_audit(
        tmp_path,
        "linked_projection",
        context={"frame_size": (1920, 1080)},
    )
    audit.event(
        "pattern_captured",
        metrics={"matched_points": np.int64(18), "error": np.float32(0.25)},
        details={"zone": "far_left"},
    )
    summary_path = audit.finish(
        "success",
        quality={"p95_px": np.float64(0.42)},
        artifacts=[tmp_path / "projection.json"],
    )

    events = [json.loads(line) for line in audit.events_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [event["action"] for event in events] == [
        "session_started",
        "pattern_captured",
        "session_finished",
    ]
    assert events[1]["metrics"] == {"matched_points": 18, "error": 0.25}
    assert summary["status"] == "success"
    assert summary["event_count"] == 3
    assert summary["quality"]["p95_px"] == 0.42
    assert summary["context"]["frame_size"] == [1920, 1080]


def test_calibration_audit_records_failure_and_is_idempotent(tmp_path) -> None:
    audit = start_calibration_audit(tmp_path, "ball_center")
    failure = RuntimeError("holdout failed")

    first = audit.finish("failed", error=failure)
    second = audit.finish("success")
    summary = json.loads(first.read_text(encoding="utf-8"))

    assert first == second
    assert summary["status"] == "failed"
    assert summary["error"] == {"type": "RuntimeError", "message": "holdout failed"}
    assert len(audit.events_path.read_text(encoding="utf-8").splitlines()) == 2


def test_calibration_audit_summary_loader_filters_workflows(tmp_path) -> None:
    first = start_calibration_audit(tmp_path, "linked_projection")
    first.finish("success")
    second = start_calibration_audit(tmp_path, "ball_center")
    second.finish("aborted")

    rows = load_calibration_audit_summaries(tmp_path, workflow="ball_center")

    assert len(rows) == 1
    assert rows[0]["workflow"] == "ball_center"
    assert rows[0]["status"] == "aborted"
