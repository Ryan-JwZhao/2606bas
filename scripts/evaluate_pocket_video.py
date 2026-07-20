from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bas.calibration import create_calibration_service
from bas.config import AppConfig
from bas.geometry import TableGeometry, TableGeometryLoader
from bas.perception import PocketObserver, build_detection_region_policy
from bas.schemas import FramePacket, MatchPhase, TrackObservation, TracksFrame
from bas.state.pocket import PerBallPocketFSM
from bas.table_boundaries import EdgeInsets, derive_table_boundaries
from bas.web_control.pocket_notice import BALL_CODE_BY_GROUP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重放轨迹与原视频，验收自动进球检测。")
    parser.add_argument("--replay", type=Path, required=True, help="events.jsonl 或其所在目录")
    parser.add_argument("--video", type=Path, required=True, help="与回放 frame_id 对齐的原视频")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("tests/fixtures/long_video_goal_labels.json"),
        help="小型人工复核标签清单",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--trace-shot", type=int, action="append", default=[], help="输出指定杆号的袋口证据")
    parser.add_argument("--stop-frame", type=int, default=None, help="诊断时在指定帧结束回放")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    replay_path = args.replay / "events.jsonl" if args.replay.is_dir() else args.replay
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    contact_by_shot = {int(item["shot"]): int(item["contact_frame"]) for item in labels.get("goals") or []}
    config = AppConfig.load(args.config).resolve_paths()
    config.state.engine = "modern"
    config.state.pocket_entry_candidate_depth_mm = 125.0
    config.state.pocket_entry_handoff_ms = 450
    config.state.pocket_entry_history_depth_mm = 450.0
    config.state.pocket_entry_history_ms = 1500
    config.state.pocket_visual_confirmation_ms = 1300

    phases, shot_starts = _read_replay_index(replay_path)
    calibration = create_calibration_service(config.calibration, frame_undistorted=True)
    geometry = TableGeometryLoader.load_optional(
        config.geometry.outline_path,
        config.geometry.inline_path,
        config.geometry.pocket_path,
    )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    policy, table_context = _build_context(config, calibration, geometry, width, height)
    observer = PocketObserver(history_ms=config.state.pocket_entry_history_ms)
    fsm = PerBallPocketFSM(config.state)
    fsm.set_table_context(**table_context)

    video_offset = int(labels.get("video_frame_offset", 0))
    video_cursor = -1
    detected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    observer_timings: list[float] = []
    source_intervals_ms: list[float] = []
    previous_ts_ns: int | None = None
    visual_trace: list[dict[str, Any]] = []
    try:
        for tracks in _iter_tracks(replay_path):
            if previous_ts_ns is not None and tracks.ts_cam_ns > previous_ts_ns:
                source_intervals_ms.append((tracks.ts_cam_ns - previous_ts_ns) / 1_000_000.0)
            previous_ts_ns = tracks.ts_cam_ns
            target_video_frame = tracks.frame_id + video_offset
            image = None
            while video_cursor < target_video_frame:
                ok, image = capture.read()
                video_cursor += 1
                if not ok:
                    raise RuntimeError(f"视频在 frame={video_cursor} 提前结束")
            if image is None:
                continue
            visual = observer.update(
                FramePacket(tracks.frame_id, tracks.ts_cam_ns, "replay", image=image),
                tracks,
                policy,
            )
            observer_timings.append(float(visual.latency_ms))
            shot = _shot_for_frame(tracks.frame_id, shot_starts)
            trace_contact = contact_by_shot.get(int(shot or -1))
            if shot in set(args.trace_shot) and (trace_contact is None or abs(tracks.frame_id - trace_contact) <= 90):
                for observed in visual.observations:
                    if not (
                        observed.inward_crossing
                        or observed.outward_crossing
                        or observed.lip_occupied
                        or observed.foreground_score >= 0.05
                    ):
                        continue
                    visual_trace.append(
                        {
                            "shot": shot,
                            "frame_id": tracks.frame_id,
                            "pocket_index": observed.pocket_index,
                            "inward": observed.inward_crossing,
                            "outward": observed.outward_crossing,
                            "lip": observed.lip_occupied,
                            "clear": observed.clear,
                            "group": observed.group,
                            "track_ids": observed.associated_track_ids,
                            "sources": observed.evidence_sources,
                            "motion_score": round(float(observed.motion_score), 3),
                            "foreground_score": round(float(observed.foreground_score), 3),
                            "foreground_center_px": observed.foreground_center_px,
                            "foreground_depth_diameters": (
                                round(float(observed.foreground_depth_diameters), 3)
                                if observed.foreground_depth_diameters is not None
                                else None
                            ),
                        }
                    )
            phase = MatchPhase(phases.get(tracks.frame_id, MatchPhase.STABLE_IDLE.value))
            for event in fsm.update(tracks, phase, visual):
                if event.name in {"POCKET_CANDIDATE", "POCKET_REJECTED", "POCKET_DETECTED"}:
                    diagnostics.append(
                        {
                            "shot": _shot_for_frame(event.frame_id, shot_starts),
                            "frame_id": event.frame_id,
                            "name": event.name,
                            "group": event.payload.get("group"),
                            "track_id": event.payload.get("track_id"),
                            "pocket_index": event.payload.get("pocket_index"),
                            "candidate_reason": event.payload.get("candidate_reason"),
                            "reason_codes": event.payload.get("reason_codes", []),
                            "visual_status": event.payload.get("visual_status"),
                            "evidence": event.payload.get("evidence", {}),
                        }
                    )
                if event.name != "POCKET_DETECTED":
                    continue
                shot = _shot_for_frame(event.frame_id, shot_starts)
                payload = dict(event.payload)
                detected.append(
                    {
                        "shot": shot,
                        "frame_id": event.frame_id,
                        "group": payload.get("group"),
                        "ball_code": BALL_CODE_BY_GROUP.get(str(payload.get("group") or "")),
                        "pocket_index": payload.get("pocket_index"),
                        "decision_id": payload.get("decision_id"),
                        "decision_latency_ms": payload.get("decision_latency_ms"),
                        "evidence_sources": payload.get("evidence_sources", []),
                    }
                )
            if args.stop_frame is not None and tracks.frame_id >= args.stop_frame:
                break
    finally:
        capture.release()

    result = _evaluate(labels, detected, observer_timings, source_intervals_ms)
    if args.stop_frame is not None:
        result["state_snapshot"] = fsm.debug_snapshot()
    if visual_trace:
        result["visual_trace"] = visual_trace
    interesting_shots = {
        int(item["shot"])
        for item in [*result["misses"], *(labels.get("no_goals") or [])]
    }
    interesting_shots.update(
        int(item["shot"])
        for item in [*result["false_positives"], *result["late"]]
        if item.get("shot") is not None
    )
    result["diagnostics"] = [item for item in diagnostics if item.get("shot") in interesting_shots]
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_report(result)
    return 0 if result["passed"] else 1


def _read_replay_index(path: Path) -> tuple[dict[int, str], list[tuple[int, int]]]:
    phases: dict[int, str] = {}
    shot_starts: list[tuple[int, int]] = []
    shot_number = 0
    for row in _iter_jsonl(path):
        if row.get("type") != "state":
            continue
        payload = dict(row.get("payload") or {})
        frame_id = int(payload["frame_id"])
        phases[frame_id] = str(payload.get("phase") or MatchPhase.STABLE_IDLE.value)
        for event in payload.get("events") or []:
            if str(event.get("name")) == "SHOT_STARTED":
                shot_number += 1
                shot_starts.append((shot_number, frame_id))
    return phases, shot_starts


def _iter_tracks(path: Path) -> Iterator[TracksFrame]:
    for row in _iter_jsonl(path):
        if row.get("type") != "tracks":
            continue
        payload = dict(row.get("payload") or {})
        yield TracksFrame(
            frame_id=int(payload["frame_id"]),
            ts_cam_ns=int(payload["ts_cam_ns"]),
            tracks=[_track(item) for item in payload.get("tracks") or []],
        )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _track(payload: dict[str, Any]) -> TrackObservation:
    return TrackObservation(
        track_id=int(payload["track_id"]),
        bbox=tuple(payload["bbox"]),
        center_px=tuple(payload["center_px"]),
        radius_px=float(payload["radius_px"]),
        cls_name=str(payload["cls_name"]),
        group=str(payload["group"]),
        confidence=float(payload["confidence"]),
        velocity_px_s=tuple(payload.get("velocity_px_s") or (0.0, 0.0)),
        center_mm=tuple(payload.get("center_mm") or payload["center_px"]),
        velocity_mm_s=tuple(payload.get("velocity_mm_s") or (0.0, 0.0)),
        radius_mm=float(payload.get("radius_mm") or 28.575),
        quality=float(payload.get("quality", 1.0)),
        age=int(payload.get("age", 1)),
        lost_frames=int(payload.get("lost_frames", 0)),
        visibility=str(payload.get("visibility", "visible")),
    )


def _build_context(
    config: AppConfig,
    calibration: Any,
    geometry: TableGeometry,
    width: int,
    height: int,
) -> tuple[Any, dict[str, Any]]:
    _, inner_px, pocket_curves_px = geometry.scaled(width, height)
    visible_mm = calibration.camera_px_to_table_mm(inner_px)
    pocket_curves_mm = [
        calibration.camera_px_to_table_mm(np.asarray(curve, dtype=np.float32))
        for curve in pocket_curves_px
    ]
    boundaries = derive_table_boundaries(
        visible_mm,
        pocket_curves_mm,
        table_width_mm=float(calibration.table.width_mm),
        table_height_mm=float(calibration.table.height_mm),
        ball_diameter_mm=float(calibration.table.ball_diameter_mm),
        projection_visible_insets=EdgeInsets(
            top_mm=float(config.calibration.projection_visible_inset_top_mm),
            right_mm=float(config.calibration.projection_visible_inset_right_mm),
            bottom_mm=float(config.calibration.projection_visible_inset_bottom_mm),
            left_mm=float(config.calibration.projection_visible_inset_left_mm),
        ),
        physical_rail_insets=EdgeInsets(
            top_mm=float(config.calibration.physical_rail_inset_top_mm),
            right_mm=float(config.calibration.physical_rail_inset_right_mm),
            bottom_mm=float(config.calibration.physical_rail_inset_bottom_mm),
            left_mm=float(config.calibration.physical_rail_inset_left_mm),
        ),
        physical_middle_pocket_relief_top_mm=float(config.calibration.physical_middle_pocket_relief_top_mm),
        physical_middle_pocket_relief_bottom_mm=float(config.calibration.physical_middle_pocket_relief_bottom_mm),
        center_reachable_extra_margin_mm=float(config.calibration.center_reachable_extra_margin_mm),
    )
    diameters = _ball_diameters_px(calibration, pocket_curves_px)
    policy = build_detection_region_policy(
        (height, width, 3),
        geometry,
        ball_diameter_px_by_pocket=diameters,
    )
    context = {
        "inner_polygon_mm": [tuple(point) for point in boundaries.physical_rail_polygon_mm],
        "table_edge_polygon_mm": [tuple(point) for point in visible_mm],
        "ball_center_reachable_polygon_mm": [tuple(point) for point in boundaries.center_playable_polygon_mm],
        "pockets_mm": list(boundaries.physical_pocket_points_mm),
        "ball_diameter_mm": float(calibration.table.ball_diameter_mm),
        "pocket_curves_mm": [[tuple(point) for point in curve] for curve in pocket_curves_mm],
    }
    return policy, context


def _ball_diameters_px(calibration: Any, pocket_curves: list[np.ndarray]) -> list[float]:
    diameter_mm = float(calibration.table.ball_diameter_mm)
    output: list[float] = []
    for curve in pocket_curves:
        center_px = np.mean(np.asarray(curve, dtype=np.float32), axis=0)
        center_mm = calibration.camera_px_to_table_mm(np.asarray([center_px], dtype=np.float32))[0]
        half = diameter_mm * 0.5
        samples = calibration.table_mm_to_camera_px(
            np.asarray(
                [
                    [center_mm[0] - half, center_mm[1]],
                    [center_mm[0] + half, center_mm[1]],
                    [center_mm[0], center_mm[1] - half],
                    [center_mm[0], center_mm[1] + half],
                ],
                dtype=np.float32,
            )
        )
        output.append(float(np.median([np.linalg.norm(samples[1] - samples[0]), np.linalg.norm(samples[3] - samples[2])])))
    return output


def _shot_for_frame(frame_id: int, starts: list[tuple[int, int]]) -> int | None:
    shot = None
    for shot_number, start in starts:
        if start > frame_id:
            break
        shot = shot_number
    return shot


def _evaluate(
    labels: dict[str, Any],
    detected: list[dict[str, Any]],
    timings: list[float],
    source_intervals_ms: list[float],
) -> dict[str, Any]:
    expected = [dict(item) for item in labels.get("goals") or []]
    forbidden = {int(item["shot"]) for item in labels.get("no_goals") or []}
    max_delay = int(labels.get("max_notice_delay_ms", 1500))
    max_frame_gap = max(1, int(labels.get("max_match_frame_gap", 90)))
    by_shot: dict[int, list[dict[str, Any]]] = {}
    for item in detected:
        if item["shot"] is not None:
            by_shot.setdefault(int(item["shot"]), []).append(item)
    misses: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    wrong: list[dict[str, Any]] = []
    late: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    duplicate_indices: set[int] = set()
    wrong_indices: set[int] = set()
    for goal in sorted(expected, key=lambda item: int(item.get("contact_frame", 0))):
        shot = int(goal["shot"])
        contact_frame = goal.get("contact_frame")
        if contact_frame is None:
            found_indices = [index for index, item in enumerate(detected) if item.get("shot") == shot]
        else:
            contact = int(contact_frame)
            found_indices = [
                index
                for index, item in enumerate(detected)
                if index not in used_indices
                and contact - 5 <= int(item["frame_id"]) <= contact + max_frame_gap
            ]
        matching_indices = [
            index
            for index in found_indices
            if detected[index]["group"] == goal["group"]
            and int(detected[index]["pocket_index"]) == int(goal["pocket_index"])
        ]
        matching_indices.sort(
            key=lambda index: abs(int(detected[index]["frame_id"]) - int(contact_frame or detected[index]["frame_id"]))
        )
        matching = [detected[index] for index in matching_indices]
        if not matching:
            misses.append(goal)
            legacy_wrong = [index for index in found_indices if detected[index].get("shot") == shot]
            wrong.extend(detected[index] for index in legacy_wrong)
            wrong_indices.update(legacy_wrong)
            continue
        used_indices.add(matching_indices[0])
        if len(matching) > 1:
            duplicates.extend(matching[1:])
            duplicate_indices.update(matching_indices[1:])
        for item in matching:
            if int(item.get("decision_latency_ms") or 0) > max_delay:
                late.append(item)
    false_positives = [
        item
        for index, item in enumerate(detected)
        if index not in used_indices
        and index not in duplicate_indices
        and index not in wrong_indices
        and (
            int(item.get("shot") or -1) in forbidden
            or not any(
                item["group"] == goal["group"]
                and int(item["pocket_index"]) == int(goal["pocket_index"])
                and (
                    goal.get("contact_frame") is None
                    or int(goal["contact_frame"]) - 5
                    <= int(item["frame_id"])
                    <= int(goal["contact_frame"]) + max_frame_gap
                )
                for goal in expected
            )
        )
    ]
    p95 = float(np.percentile(timings, 95)) if timings else 0.0
    observer_mean = float(np.mean(timings)) if timings else 0.0
    source_interval_mean = float(np.mean(source_intervals_ms)) if source_intervals_ms else 0.0
    estimated_drop = observer_mean / source_interval_mean * 100.0 if source_interval_mean > 0.0 else 0.0
    passed = (
        not (misses or duplicates or wrong or false_positives or late)
        and p95 <= 8.0
        and estimated_drop <= 5.0
    )
    return {
        "passed": passed,
        "expected_goals": len(expected),
        "detected_goals": len(detected),
        "matched_goals": len(expected) - len(misses),
        "misses": misses,
        "wrong": wrong,
        "duplicates": duplicates,
        "false_positives": false_positives,
        "late": late,
        "observer_p95_ms": p95,
        "observer_mean_ms": observer_mean,
        "source_frame_interval_mean_ms": source_interval_mean,
        "estimated_frame_rate_drop_pct": estimated_drop,
        "detections": detected,
    }


def _print_report(result: dict[str, Any]) -> None:
    verdict = "PASS" if result["passed"] else "FAIL"
    print(
        f"{verdict}: matched={result['matched_goals']}/{result['expected_goals']} "
        f"detected={result['detected_goals']} observer_p95={result['observer_p95_ms']:.2f}ms "
        f"estimated_fps_drop={result['estimated_frame_rate_drop_pct']:.2f}%"
    )
    for key in ("misses", "wrong", "duplicates", "false_positives", "late"):
        if result[key]:
            print(f"{key}: {json.dumps(result[key], ensure_ascii=False)}")
    if result.get("diagnostics"):
        print("diagnostics:")
        for item in result["diagnostics"]:
            print(
                f"  shot={item['shot']} frame={item['frame_id']} {item['name']} "
                f"{item['group']} track={item['track_id']} pocket={item['pocket_index']} reason="
                f"{item['candidate_reason'] or item['reason_codes']} visual={item['visual_status']}"
            )
    if result.get("visual_trace"):
        print("visual_trace:")
        for item in result["visual_trace"]:
            print(f"  {json.dumps(item, ensure_ascii=False)}")
    print("detections:")
    for item in result["detections"]:
        print(
            f"  shot={item['shot']} frame={item['frame_id']} {item['ball_code']} "
            f"pocket={item['pocket_index']} latency={item['decision_latency_ms']}ms"
        )


if __name__ == "__main__":
    sys.exit(main())
