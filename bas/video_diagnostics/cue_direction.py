from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from ..config import PlannerConfig
from ..planning.cue_aim import CueStickAimDetector
from ..planning.cue_direction_stability import CueDirectionStabilizer
from ..utils import unit


def diagnose_cue_direction_video(
    *,
    video_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path | None = None,
    preview_path: str | Path | None = None,
) -> dict[str, Any]:
    video = Path(video_path).resolve()
    annotations_file = Path(annotations_path).resolve()
    labels = json.loads(annotations_file.read_text(encoding="utf-8"))
    expected_name = str(labels.get("video_basename") or "").strip()
    if expected_name and video.name != expected_name:
        raise ValueError(f"标注对应视频为 {expected_name}，当前输入为 {video.name}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    detector = CueStickAimDetector()
    case_reports: list[dict[str, Any]] = []
    preview_tiles: list[np.ndarray] = []
    all_failures: list[str] = []
    try:
        for case in labels.get("cases", []):
            report, tiles = _evaluate_case(
                capture=capture,
                detector=detector,
                config=case,
                frame_count=frame_count,
            )
            case_reports.append(report)
            preview_tiles.extend(tiles)
            all_failures.extend(
                f"{report['name']}: {failure}" for failure in report.get("failures", [])
            )
    finally:
        capture.release()

    report = {
        "schema": "bas_cue_direction_video_diagnosis_v1",
        "verdict": "PASS" if not all_failures else "FAIL",
        "failure_count": len(all_failures),
        "failures": all_failures,
        "video": {
            "path": str(video),
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frame_count,
            "duration_s": frame_count / max(0.001, fps),
        },
        "annotations": str(annotations_file),
        "cases": case_reports,
    }

    if output_path is None:
        output = Path(".tmp") / "cue_direction_video_diagnosis" / f"{video.stem}.json"
    else:
        output = Path(output_path)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report["output_path"] = str(output)

    if preview_tiles:
        if preview_path is None:
            preview = output.with_name(f"{output.stem}_preview.jpg")
        else:
            preview = Path(preview_path).resolve()
        preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview), _contact_sheet(preview_tiles))
        report["preview_path"] = str(preview)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _evaluate_case(
    *,
    capture: cv2.VideoCapture,
    detector: CueStickAimDetector,
    config: dict[str, Any],
    frame_count: int,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    name = str(config.get("name") or "unnamed")
    frame_ids = _case_frame_ids(config)
    cue_center = np.asarray(config["cue_center_px"], dtype=np.float32).reshape((2,))
    cue_radius = max(2.0, float(config["cue_radius_px"]))
    expected_value = config.get("expected_direction_px")
    expected = unit(np.asarray(expected_value, dtype=np.float32)) if expected_value is not None else None
    expect_none = bool(config.get("expect_none", expected is None))
    min_dot = float(config.get("min_dot", 0.80))
    min_detection_ratio = float(config.get("min_detection_ratio", 1.0 if not expect_none else 0.0))
    preview_frames = {int(value) for value in config.get("preview_frames", frame_ids[:1])}
    stabilizer = CueDirectionStabilizer(
        PlannerConfig(
            cue_sector_angle_deg=float(config.get("stability_angle_deg", 10.0)),
            cue_sector_edge_margin_deg=1.0,
            cue_sector_switch_confirm_frames=int(config.get("stability_confirm_frames", 6)),
        )
    )

    records: list[dict[str, Any]] = []
    tiles: list[np.ndarray] = []
    detected_count = 0
    raw_wrong_frames: list[int] = []
    stable_wrong_frames: list[int] = []
    unexpected_frames: list[int] = []
    for frame_id in frame_ids:
        if frame_id < 0 or frame_id >= frame_count:
            raise ValueError(f"{name} 的帧号 {frame_id} 超出视频范围 0..{frame_count - 1}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"无法读取 {name} 的第 {frame_id} 帧")
        aim = detector.detect(
            frame_bgr=frame,
            tracks=[],
            cue_center_px=cue_center,
            cue_radius_px=cue_radius,
            min_stick_quality=0.0,
        )
        raw_direction: Optional[np.ndarray] = None
        stable_direction: Optional[np.ndarray] = None
        stability_status = "no_aim"
        raw_dot: Optional[float] = None
        stable_dot: Optional[float] = None
        if aim is None:
            stabilizer.reset()
        else:
            detected_count += 1
            raw_direction = unit(aim.direction_px)
            decision = stabilizer.stabilize(raw_direction, raw_direction)
            stable_direction = unit(np.asarray(decision.direction_px, dtype=np.float32))
            stability_status = decision.status
            if expected is not None:
                raw_dot = float(np.dot(raw_direction, expected))
                stable_dot = float(np.dot(stable_direction, expected))
                if raw_dot < min_dot:
                    raw_wrong_frames.append(frame_id)
                if stable_dot < min_dot:
                    stable_wrong_frames.append(frame_id)
            if expect_none:
                unexpected_frames.append(frame_id)
        records.append(
            {
                "frame": int(frame_id),
                "detected": aim is not None,
                "source": None if aim is None else aim.source,
                "raw_direction_px": _vector_json(raw_direction),
                "stable_direction_px": _vector_json(stable_direction),
                "raw_angle_deg": _angle_json(raw_direction),
                "stable_angle_deg": _angle_json(stable_direction),
                "raw_expected_dot": raw_dot,
                "stable_expected_dot": stable_dot,
                "direction_confidence": None if aim is None else float(aim.direction_confidence),
                "direction_status": None if aim is None else str(aim.direction_status),
                "stability_status": stability_status,
            }
        )
        if frame_id in preview_frames:
            tiles.append(
                _draw_preview_tile(
                    frame,
                    frame_id=frame_id,
                    cue_center=cue_center,
                    cue_radius=cue_radius,
                    expected=expected,
                    raw_direction=raw_direction,
                    stable_direction=stable_direction,
                )
            )

    detection_ratio = detected_count / max(1, len(frame_ids))
    failures: list[str] = []
    if expect_none:
        if unexpected_frames:
            failures.append(f"无球杆预期帧产生方向：{unexpected_frames}")
    else:
        if detection_ratio < min_detection_ratio:
            failures.append(
                f"有效方向比例 {detection_ratio:.3f} 低于要求 {min_detection_ratio:.3f}"
            )
        if raw_wrong_frames:
            failures.append(f"原始方向错误帧：{raw_wrong_frames}")
        if stable_wrong_frames:
            failures.append(f"稳定方向错误帧：{stable_wrong_frames}")
    return (
        {
            "name": name,
            "frame_start": min(frame_ids),
            "frame_end": max(frame_ids),
            "evaluated_frames": len(frame_ids),
            "detected_frames": detected_count,
            "detection_ratio": detection_ratio,
            "expect_none": expect_none,
            "expected_direction_px": _vector_json(expected),
            "min_dot": min_dot,
            "raw_wrong_frames": raw_wrong_frames,
            "stable_wrong_frames": stable_wrong_frames,
            "unexpected_frames": unexpected_frames,
            "failures": failures,
            "records": records,
        },
        tiles,
    )


def _case_frame_ids(config: dict[str, Any]) -> list[int]:
    if "frame" in config:
        return [int(config["frame"])]
    start = int(config["frame_start"])
    end = int(config["frame_end"])
    step = max(1, int(config.get("frame_step", 1)))
    return list(range(start, end + 1, step))


def _vector_json(value: Optional[np.ndarray]) -> Optional[list[float]]:
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float32).reshape((2,))
    return [float(vector[0]), float(vector[1])]


def _angle_json(value: Optional[np.ndarray]) -> Optional[float]:
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float32).reshape((2,))
    return float(math.degrees(math.atan2(float(vector[1]), float(vector[0]))))


def _draw_preview_tile(
    frame: np.ndarray,
    *,
    frame_id: int,
    cue_center: np.ndarray,
    cue_radius: float,
    expected: Optional[np.ndarray],
    raw_direction: Optional[np.ndarray],
    stable_direction: Optional[np.ndarray],
) -> np.ndarray:
    image = frame.copy()
    center = tuple(int(round(float(value))) for value in cue_center)
    cv2.circle(image, center, int(round(cue_radius)), (0, 220, 255), 3, cv2.LINE_AA)
    for direction, color in (
        (expected, (80, 255, 80)),
        (raw_direction, (255, 220, 40)),
        (stable_direction, (255, 80, 255)),
    ):
        if direction is None:
            continue
        endpoint = cue_center + unit(direction) * 180.0
        cv2.arrowedLine(
            image,
            center,
            tuple(int(round(float(value))) for value in endpoint),
            color,
            4,
            cv2.LINE_AA,
            tipLength=0.12,
        )
    cv2.putText(
        image,
        f"frame={frame_id} expected=green raw=cyan stable=magenta",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)


def _contact_sheet(tiles: Sequence[np.ndarray]) -> np.ndarray:
    columns = 2
    blank = np.zeros_like(tiles[0])
    padded = list(tiles)
    while len(padded) % columns:
        padded.append(blank.copy())
    rows = [np.hstack(padded[index : index + columns]) for index in range(0, len(padded), columns)]
    return np.vstack(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="使用人工标注的实际视频验证球杆方向")
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output")
    parser.add_argument("--preview")
    args = parser.parse_args(argv)
    report = diagnose_cue_direction_video(
        video_path=args.video,
        annotations_path=args.annotations,
        output_path=args.output,
        preview_path=args.preview,
    )
    print(json.dumps({key: report[key] for key in ("verdict", "failure_count", "output_path", "preview_path") if key in report}, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
