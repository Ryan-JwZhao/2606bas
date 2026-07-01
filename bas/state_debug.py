from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .schemas import DetectionsFrame, Event, FramePacket, MatchStateFrame, ShotCandidate, ShotPlan, TracksFrame, to_jsonable
from .utils import wall_time_id


@dataclass(frozen=True)
class StateDebugSessionResult:
    session_dir: Path
    jsonl_path: Path
    srt_path: Path
    summary_path: Path
    frame_count: int


class StateDebugSession:
    def __init__(
        self,
        root_dir: str | Path,
        *,
        capture_fps: float,
        raw_video_path: str | Path | None = None,
        route_video_path: str | Path | None = None,
    ):
        self.capture_fps = max(1.0, float(capture_fps))
        self.session_id = f"state_debug_{wall_time_id()}"
        self.session_dir = Path(root_dir) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.session_dir / "state_machine_frames.jsonl"
        self.srt_path = self.session_dir / "state_machine_timeline.srt"
        self.summary_path = self.session_dir / "session_summary.json"
        self.started_at_local = time.strftime("%Y-%m-%d %H:%M:%S")
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self._first_ts_cam_ns: Optional[int] = None
        self._raw_video_path = str(raw_video_path) if raw_video_path is not None else None
        self._route_video_path = str(route_video_path) if route_video_path is not None else None
        self._records: list[dict[str, Any]] = []
        self._phase_counts: Counter[str] = Counter()
        self._event_counts: Counter[str] = Counter()

    def record_frame(
        self,
        *,
        frame: FramePacket,
        detections: DetectionsFrame,
        tracks: TracksFrame,
        state: MatchStateFrame,
        raw_plan: ShotPlan,
        display_plan: ShotPlan,
        state_debug: Dict[str, Any],
        route_status_text: str,
    ) -> None:
        ts_cam_ns = int(frame.ts_cam_ns)
        if self._first_ts_cam_ns is None:
            self._first_ts_cam_ns = ts_cam_ns
        rel_time_s = self._relative_time_s(ts_cam_ns, index=len(self._records))
        events = [_event_payload(event) for event in state.events]
        raw_plan_summary = _plan_summary(raw_plan)
        display_plan_summary = _plan_summary(display_plan)
        record = {
            "frame_id": int(frame.frame_id),
            "ts_cam_ns": ts_cam_ns,
            "t_rel_s": rel_time_s,
            "camera_id": str(frame.camera_id),
            "phase": str(state.phase),
            "turn_target_group": state.turn_target_group,
            "state_confidence": float(state.confidence),
            "detections": int(len(detections.detections)),
            "tracks": int(len(tracks.tracks)),
            "events": events,
            "state_debug": to_jsonable(state_debug),
            "plan_raw": raw_plan_summary,
            "plan_displayed": display_plan_summary,
            "route_display_status": str(route_status_text or ""),
            "recordings": {
                "raw_video_path": self._raw_video_path,
                "route_video_path": self._route_video_path,
            },
        }
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        self._records.append(record)
        self._phase_counts[str(state.phase)] += 1
        for event in events:
            self._event_counts[str(event["name"])] += 1

    def close(
        self,
        *,
        raw_video_path: str | Path | None = None,
        route_video_path: str | Path | None = None,
        status: str = "completed",
    ) -> StateDebugSessionResult:
        if raw_video_path is not None:
            self._raw_video_path = str(raw_video_path)
        if route_video_path is not None:
            self._route_video_path = str(route_video_path)
        self._jsonl.close()
        self._write_srt()
        self._write_summary(status=status)
        return StateDebugSessionResult(
            session_dir=self.session_dir,
            jsonl_path=self.jsonl_path,
            srt_path=self.srt_path,
            summary_path=self.summary_path,
            frame_count=len(self._records),
        )

    def _relative_time_s(self, ts_cam_ns: int, *, index: int) -> float:
        if self._first_ts_cam_ns is None:
            return index / self.capture_fps
        delta_ns = max(0, int(ts_cam_ns) - int(self._first_ts_cam_ns))
        delta_s = delta_ns / 1e9
        fallback_s = index / self.capture_fps
        if delta_s <= 0.0 and index > 0:
            return fallback_s
        if delta_s < fallback_s * 0.25:
            return fallback_s
        return delta_s

    def _write_srt(self) -> None:
        frame_duration_s = 1.0 / self.capture_fps
        lines: list[str] = []
        for idx, record in enumerate(self._records, start=1):
            start_s = float(record["t_rel_s"])
            if idx < len(self._records):
                end_s = float(self._records[idx]["t_rel_s"])
            else:
                end_s = start_s + frame_duration_s
            if end_s <= start_s:
                end_s = start_s + frame_duration_s
            lines.append(str(idx))
            lines.append(f"{_srt_time(start_s)} --> {_srt_time(end_s)}")
            lines.append(_subtitle_text(record))
            lines.append("")
        self.srt_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_summary(self, *, status: str) -> None:
        payload = {
            "session_id": self.session_id,
            "status": str(status),
            "started_at_local": self.started_at_local,
            "ended_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "capture_fps": float(self.capture_fps),
            "frames": int(len(self._records)),
            "first_frame_id": int(self._records[0]["frame_id"]) if self._records else None,
            "last_frame_id": int(self._records[-1]["frame_id"]) if self._records else None,
            "phase_counts": dict(sorted(self._phase_counts.items())),
            "event_counts": dict(sorted(self._event_counts.items())),
            "artifacts": {
                "jsonl": str(self.jsonl_path),
                "srt": str(self.srt_path),
                "raw_video_path": self._raw_video_path,
                "route_video_path": self._route_video_path,
            },
        }
        self.summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _event_payload(event: Event) -> dict[str, Any]:
    return {
        "name": str(event.name),
        "frame_id": int(event.frame_id),
        "ts_cam_ns": int(event.ts_cam_ns),
        "confidence": float(event.confidence),
        "payload": to_jsonable(event.payload),
    }


def _plan_summary(plan: ShotPlan) -> dict[str, Any]:
    return {
        "plan_id": str(plan.plan_id),
        "shot_mode": str(plan.shot_mode),
        "free_status": str(plan.free_status),
        "planner_version": str(plan.planner_version),
        "locked_target_id": plan.locked_target_id,
        "target_lock_status": str(plan.target_lock_status),
        "target_shot_status": str(plan.target_shot_status),
        "best": _candidate_summary(plan.best),
        "candidates": [_candidate_summary(candidate) for candidate in list(plan.candidates or [])],
    }


def _candidate_summary(candidate: Optional[ShotCandidate]) -> Optional[dict[str, Any]]:
    if candidate is None:
        return None
    explanation = dict(candidate.explanation or {})
    summary = {
        "candidate_id": str(candidate.candidate_id),
        "target_track_id": int(candidate.target_track_id),
        "target_group": str(candidate.target_group),
        "pocket_index": int(candidate.pocket_index),
        "score": float(candidate.score),
        "risk": float(candidate.risk),
        "cut_angle_deg": float(candidate.cut_angle_deg),
    }
    if "target_shot_rebounds" in explanation:
        summary["target_shot_rebounds"] = int(explanation["target_shot_rebounds"])
    return summary


def _subtitle_text(record: dict[str, Any]) -> str:
    state_debug = dict(record.get("state_debug") or {})
    signals = dict(state_debug.get("signals") or {})
    counters = dict(state_debug.get("counters") or {})
    pocket_rows = list(state_debug.get("pocket_fsm") or [])
    events = list(record.get("events") or [])
    raw_plan = dict(record.get("plan_raw") or {})
    display_plan = dict(record.get("plan_displayed") or {})
    lines = [
        f"f{int(record['frame_id'])}  T+{float(record['t_rel_s']):.3f}s  phase={record['phase']}  det={record['detections']}  track={record['tracks']}",
        (
            "cnt "
            f"S={int(counters.get('stable_count', 0))} "
            f"M={int(counters.get('moving_count', 0))} "
            f"A={int(counters.get('armed_count', 0))} "
            f"Set={int(counters.get('settle_count', 0))} "
            f"X={int(counters.get('anomaly_count', 0))}"
        ),
        (
            "sig "
            f"moving={_bit(signals.get('moving'))} "
            f"stable={_bit(signals.get('stable'))} "
            f"anomaly={_bit(signals.get('anomaly'))} "
            f"cueStick={_bit(signals.get('cue_stick_seen'))} "
            f"cueMotion={_bit(signals.get('cue_motion'))} "
            f"shotVote={_bit(signals.get('shot_start_voted'))}"
        ),
    ]
    if events:
        lines.append("events " + " | ".join(_compact_events(events)))
    lines.extend(_compact_pocket_rows(pocket_rows))
    lines.append("raw " + _compact_plan(raw_plan))
    if raw_plan != display_plan:
        lines.append("show " + _compact_plan(display_plan))
    route_status = str(record.get("route_display_status") or "").strip()
    if route_status:
        lines.append(f"route {route_status}")
    return "\n".join(lines)


def _compact_events(events: Iterable[dict[str, Any]]) -> list[str]:
    event_list = list(events)
    parts: list[str] = []
    for event in event_list[:4]:
        payload = dict(event.get("payload") or {})
        summary_bits = []
        for key in ("track_id", "track_a", "track_b", "pocket_index", "group", "phase"):
            if key in payload:
                summary_bits.append(f"{key}={payload[key]}")
        label = str(event.get("name") or "EVENT")
        if summary_bits:
            label += f"({', '.join(summary_bits)})"
        parts.append(label)
    if len(event_list) > 4:
        parts.append(f"+{len(event_list) - 4}")
    return parts


def _compact_pocket_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    interesting: list[dict[str, Any]] = []
    for row in rows:
        state = str(row.get("state") or "")
        decision = str(row.get("decision") or "")
        if state in {"candidate", "tentative", "commit_ready", "confirmed", "review_required", "rejected"} or decision not in {"", "none"}:
            interesting.append(dict(row))
    for row in interesting[:3]:
        reasons = ",".join(str(item) for item in list(row.get("reason_codes") or [])[:2])
        parts = [
            f"pocket#{row.get('track_id')}",
            f"z={row.get('zone') or row.get('last_zone') or '-'}",
            f"in={float(row.get('inward_speed_mm_s', 0.0)):.1f}",
            f"miss={int(row.get('missing_ms', 0))}",
            f"decision={row.get('decision') or row.get('state')}",
        ]
        if row.get("candidate_reason"):
            parts.append(f"why={row.get('candidate_reason')}")
        if reasons:
            parts.append(f"reasons={reasons}")
        lines.append(" ".join(parts))
    if len(interesting) > 3:
        lines.append(f"pocket +{len(interesting) - 3}")
    return lines


def _compact_plan(plan: dict[str, Any]) -> str:
    shot_mode = str(plan.get("shot_mode") or "rule")
    best = plan.get("best")
    candidates = list(plan.get("candidates") or [])
    if shot_mode == "free":
        return f"free status={plan.get('free_status')}"
    lock = ""
    if plan.get("locked_target_id") is not None:
        lock = f" lock=t{plan.get('locked_target_id')}:{plan.get('target_lock_status')}"
    if shot_mode == "target":
        status = str(plan.get("target_shot_status") or "off")
        if best is None:
            return f"target best=none cand={len(candidates)} status={status}{lock}"
        rebounds = best.get("target_shot_rebounds", "?")
        return f"target t{best.get('target_track_id')}/p{best.get('pocket_index')} r={rebounds} s={float(best.get('score') or 0.0):.2f} cand={len(candidates)} status={status}{lock}"
    if best is None:
        return f"rule best=none cand={len(candidates)}{lock}"
    best_id = best.get("candidate_id")
    target = best.get("target_track_id")
    pocket = best.get("pocket_index")
    score = float(best.get("score") or 0.0)
    risk = float(best.get("risk") or 0.0)
    tops = ",".join(str(item.get("candidate_id")) for item in candidates[:3]) or "none"
    return f"rule best={best_id}/t{target}/p{pocket} s={score:.2f} r={risk:.2f} cand={len(candidates)} top={tops}{lock}"


def _bit(value: Any) -> int:
    return 1 if bool(value) else 0


def _srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
