from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import StateConfig
from ..schemas import Event, MatchPhase, TrackObservation, TracksFrame, to_jsonable
from .modern import ModernMatchStateMachine


def run_pocket_trace(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config_payload = dict(payload.get("config") or {})
    config_payload["engine"] = "modern"
    state_config = StateConfig(**{k: v for k, v in config_payload.items() if k in StateConfig.__dataclass_fields__})
    machine = ModernMatchStateMachine(state_config)

    table_context = dict(payload.get("table_context") or {})
    machine.set_table_context(
        inner_polygon_mm=[tuple(point) for point in list(table_context.get("inner_polygon_mm") or [])],
        pockets_mm=[tuple(point) for point in list(table_context.get("pockets_mm") or [])],
        ball_diameter_mm=table_context.get("ball_diameter_mm"),
        pocket_curves_mm=[[tuple(point) for point in list(curve or [])] for curve in list(table_context.get("pocket_curves_mm") or [])],
    )

    initial = dict(payload.get("initial_state") or {})
    if "table_state" in initial:
        machine.rule_state.table_state = str(initial["table_state"])
    if "actor_group" in initial:
        machine.rule_state.actor_group = initial["actor_group"]
    if "opponent_group" in initial:
        machine.rule_state.opponent_group = initial["opponent_group"]
    if "game_status" in initial:
        machine.rule_state.game_status = str(initial["game_status"])
    if "shot_number" in initial:
        machine.rule_state.shot_number = int(initial["shot_number"])
    if "turn_target_group" in initial:
        machine.set_turn_target_group(initial["turn_target_group"])
    for group, count in dict(initial.get("ledger_remaining") or {}).items():
        if group in machine.ledger.remaining:
            machine.ledger.remaining[group] = int(count)

    frame_summaries: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for frame_payload in list(payload.get("frames") or []):
        frame_id = int(frame_payload["frame_id"])
        ts_cam_ns = int(frame_payload["ts_cam_ns"])
        if frame_payload.get("phase"):
            machine.phase = MatchPhase(str(frame_payload["phase"]))
        if frame_payload.get("force_phase"):
            machine.force_phase(str(frame_payload["force_phase"]), frame_id=frame_id, ts_cam_ns=ts_cam_ns, reason="fixture")
        for operator_event in list(frame_payload.get("operator_events") or []):
            if isinstance(operator_event, str):
                machine._queue_operator_event(operator_event, frame_id=frame_id, ts_cam_ns=ts_cam_ns)  # type: ignore[attr-defined]
                continue
            operator_event = dict(operator_event or {})
            machine._queue_operator_event(  # type: ignore[attr-defined]
                str(operator_event.get("name") or ""),
                frame_id=frame_id,
                ts_cam_ns=ts_cam_ns,
                payload=dict(operator_event.get("payload") or {}),
            )
        tracks = TracksFrame(
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            tracks=[_track_from_payload(item) for item in list(frame_payload.get("tracks") or [])],
        )
        state = machine.update(tracks)
        frame_events = [to_jsonable(event) for event in state.events]
        all_events.extend(frame_events)
        frame_summaries.append(
            {
                "frame_id": frame_id,
                "phase": state.phase,
                "events": frame_events,
            }
        )

    event_filter = set(str(name) for name in list(payload.get("event_filter") or []))
    filtered_events = [event for event in all_events if not event_filter or str(event.get("name")) in event_filter]
    return {
        "frames": frame_summaries,
        "events": filtered_events,
        "ledger_remaining": dict(machine.ledger.remaining),
        "rule_state": {
            "table_state": machine.rule_state.table_state,
            "actor_group": machine.rule_state.actor_group,
            "opponent_group": machine.rule_state.opponent_group,
            "shot_number": machine.rule_state.shot_number,
            "game_status": machine.rule_state.game_status,
        },
        "last_shot_context": to_jsonable(machine.debug_snapshot().get("last_shot_context") or {}),
        "last_referee_intent": to_jsonable(machine.debug_snapshot().get("referee_intent") or {}),
        "pocket_debug": to_jsonable(machine.debug_snapshot().get("pocket_fsm") or []),
    }


def _track_from_payload(payload: dict[str, Any]) -> TrackObservation:
    center = tuple(payload.get("center_mm") or payload.get("center_px") or (0.0, 0.0))
    velocity = tuple(payload.get("velocity_mm_s") or payload.get("velocity_px_s") or (0.0, 0.0))
    radius_mm = payload.get("radius_mm")
    radius_px = payload.get("radius_px")
    if radius_mm is None and radius_px is None:
        radius_mm = 28.0
        radius_px = 5.0
    bbox = tuple(
        payload.get("bbox")
        or (
            float(center[0]) - float(radius_px or 5.0),
            float(center[1]) - float(radius_px or 5.0),
            float(center[0]) + float(radius_px or 5.0),
            float(center[1]) + float(radius_px or 5.0),
        )
    )
    return TrackObservation(
        track_id=int(payload["track_id"]),
        bbox=bbox,  # type: ignore[arg-type]
        center_px=tuple(payload.get("center_px") or center),  # type: ignore[arg-type]
        radius_px=float(radius_px or 5.0),
        cls_name=str(payload.get("cls_name") or payload.get("group") or "unknown"),
        group=str(payload.get("group") or "other"),
        confidence=float(payload.get("confidence", 0.9)),
        velocity_px_s=tuple(payload.get("velocity_px_s") or velocity),  # type: ignore[arg-type]
        center_mm=tuple(payload.get("center_mm") or center),  # type: ignore[arg-type]
        velocity_mm_s=tuple(payload.get("velocity_mm_s") or velocity),  # type: ignore[arg-type]
        radius_mm=float(radius_mm or 28.0),
        quality=float(payload.get("quality", 0.9)),
        age=int(payload.get("age", 1)),
        lost_frames=int(payload.get("lost_frames", 0)),
        visibility=str(payload.get("visibility", "visible")),
        geometry_quality=float(payload.get("geometry_quality", 1.0)),
        geometry_method=str(payload.get("geometry_method", "unknown")),
    )
