from __future__ import annotations

import json
from pathlib import Path

from bas.config import StateConfig
from bas.schemas import DetectionsFrame, Event, FramePacket, MatchStateFrame, ShotCandidate, ShotPlan, TrackObservation, TracksFrame
from bas.state import MatchStateMachine
from bas.state_debug import StateDebugSession


def _ball(track_id: int, group: str, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(x - 5, y - 5, x + 5, y + 5),
        center_px=(x, y),
        radius_px=5,
        cls_name=group,
        group=group,
        confidence=0.95,
        velocity_px_s=(vx, vy),
        center_mm=(x, y),
        velocity_mm_s=(vx, vy),
        radius_mm=28.0,
        quality=0.95,
    )


def _cue_stick(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(x1, y1, x2, y2),
        center_px=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        radius_px=max(abs(x2 - x1), abs(y2 - y1)) * 0.5,
        cls_name="cue_stick",
        group="cue_stick",
        confidence=0.9,
        velocity_px_s=(0.0, 0.0),
        quality=0.9,
    )


def _candidate(candidate_id: str, target_track_id: int, *, pocket_index: int = 2, score: float = 1.2, risk: float = 0.1) -> ShotCandidate:
    return ShotCandidate(
        candidate_id=candidate_id,
        cue_track_id=1,
        target_track_id=target_track_id,
        target_group="solid",
        pocket_index=pocket_index,
        cue_ball=(100.0, 100.0),
        object_ball=(220.0, 120.0),
        ghost_ball=(200.0, 120.0),
        pocket_point=(400.0, 60.0),
        aim_line=[(100.0, 100.0), (200.0, 120.0)],
        object_line=[(220.0, 120.0), (400.0, 60.0)],
        cut_angle_deg=22.0,
        cue_distance_mm=120.0,
        object_distance_mm=210.0,
        score=score,
        risk=risk,
    )


def test_state_machine_debug_snapshot_exposes_runtime_signals() -> None:
    sm = MatchStateMachine(StateConfig(stable_frames=2, settle_frames=2))

    sm.update(
        TracksFrame(
            frame_id=1,
            ts_cam_ns=1_000_000_000,
            tracks=[_ball(1, "cue", 100, 100, 40.0, 0.0), _cue_stick(2, 40, 95, 90, 105)],
        )
    )

    snapshot = sm.debug_snapshot()

    assert snapshot["phase"] == "SHOT_ACTIVE"
    assert snapshot["signals"]["shot_start_voted"] is True
    assert snapshot["signals"]["moving"] is True
    assert snapshot["counters"]["moving_count"] >= 1
    assert snapshot["visible_group_counts"]["solid"] == 0


def test_state_debug_session_writes_jsonl_srt_and_summary(tmp_path: Path) -> None:
    session = StateDebugSession(
        tmp_path,
        capture_fps=30.0,
        raw_video_path=tmp_path / "raw.mp4",
        route_video_path=tmp_path / "route.mp4",
    )
    candidate = _candidate("c1", 3)
    raw_plan = ShotPlan(
        plan_id="raw_1",
        frame_id=1,
        ts_cam_ns=1_000_000_000,
        best=candidate,
        candidates=[candidate],
        shot_mode="rule",
    )
    display_plan = ShotPlan(
        plan_id="display_1",
        frame_id=1,
        ts_cam_ns=1_000_000_000,
        best=candidate,
        candidates=[candidate],
        shot_mode="rule",
    )
    state = MatchStateFrame(
        frame_id=1,
        ts_cam_ns=1_000_000_000,
        phase="SHOT_ACTIVE",
        events=[Event(name="SHOT_STARTED", ts_cam_ns=1_000_000_000, frame_id=1, payload={"track_id": 1})],
    )
    debug_state = {
        "signals": {
            "moving": True,
            "stable": False,
            "anomaly": False,
            "cue_stick_seen": True,
            "cue_motion": True,
            "shot_start_voted": True,
        },
        "counters": {
            "stable_count": 0,
            "moving_count": 2,
            "armed_count": 1,
            "settle_count": 0,
            "anomaly_count": 0,
        },
    }

    session.record_frame(
        frame=FramePacket(frame_id=1, ts_cam_ns=1_000_000_000, camera_id="cam"),
        detections=DetectionsFrame(frame_id=1, ts_cam_ns=1_000_000_000, detections=[]),
        tracks=TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=[_ball(1, "cue", 100, 100, 40.0, 0.0)]),
        state=state,
        raw_plan=raw_plan,
        display_plan=display_plan,
        state_debug=debug_state,
        route_status_text="frozen_motion 2/2",
    )
    session.record_frame(
        frame=FramePacket(frame_id=2, ts_cam_ns=1_033_333_333, camera_id="cam"),
        detections=DetectionsFrame(frame_id=2, ts_cam_ns=1_033_333_333, detections=[]),
        tracks=TracksFrame(frame_id=2, ts_cam_ns=1_033_333_333, tracks=[_ball(1, "cue", 104, 100, 0.0, 0.0)]),
        state=MatchStateFrame(frame_id=2, ts_cam_ns=1_033_333_333, phase="SETTLING"),
        raw_plan=ShotPlan(plan_id="raw_2", frame_id=2, ts_cam_ns=1_033_333_333, shot_mode="rule"),
        display_plan=ShotPlan(plan_id="display_2", frame_id=2, ts_cam_ns=1_033_333_333, shot_mode="rule"),
        state_debug={"signals": {"moving": False, "stable": True}, "counters": {"stable_count": 1, "moving_count": 0}},
        route_status_text="release_pending 1/8",
    )
    result = session.close(status="completed")

    assert result.frame_count == 2
    assert result.jsonl_path.exists()
    assert result.srt_path.exists()
    assert result.summary_path.exists()

    records = [json.loads(line) for line in result.jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["phase"] == "SHOT_ACTIVE"
    assert records[0]["events"][0]["name"] == "SHOT_STARTED"
    assert records[0]["plan_raw"]["best"]["candidate_id"] == "c1"

    srt_text = result.srt_path.read_text(encoding="utf-8")
    assert "phase=SHOT_ACTIVE" in srt_text
    assert "SHOT_STARTED" in srt_text
    assert "route frozen_motion 2/2" in srt_text

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["frames"] == 2
    assert summary["event_counts"]["SHOT_STARTED"] == 1
    assert summary["artifacts"]["raw_video_path"].endswith("raw.mp4")
