"""Replay recorded video through BAS and audit overlay stability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Hashable, Iterable, Optional

import cv2
import numpy as np

from ..app import PipelineOutput, RuntimePipeline
from ..config import AppConfig
from ..display_geometry import (
    DISPLAY_SHAPE_DEADBAND_PX,
    DisplayGeometryStabilizer,
    draw_subpixel_circle,
    stabilize_projection_overlay,
)
from ..projection.overlay import render_overlay_image
from ..route_freeze import MotionRouteFreezeController
from ..runtime_env import prepare_runtime_environment, preload_torch_for_backend
from ..schemas import Detection, DetectionsFrame
from ..tracking.confirmation import confirmed_tracks
from ..user_settings import UserSettings


STABLE_PHASES = {"STABLE_IDLE", "PRE_SHOT_ARMED"}
POCKET_EVENT_NAMES = {
    "POT_PROBABLE",
    "POCKET_CONFIRMED",
    "POCKET_REJECTED",
    "BALL_DISAPPEARED",
    "BALL_REAPPEARED",
    "POCKET_REAPPEARED",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return value


def _signature_text(value: Hashable) -> str:
    return repr(value)


@dataclass(frozen=True)
class FlashSample:
    frame_id: int
    value: Hashable
    stable: bool


class TripletFlashDetector:
    """Detect a one-frame A-B-A excursion, the visible shape of a flash."""

    def __init__(self, kind: str, *, limit: int = 500) -> None:
        self.kind = str(kind)
        self.limit = max(1, int(limit))
        self._samples: list[FlashSample] = []
        self.events: list[dict[str, Any]] = []
        self.total = 0

    def push(self, frame_id: int, value: Hashable, *, stable: bool) -> None:
        self._samples.append(FlashSample(int(frame_id), value, bool(stable)))
        if len(self._samples) < 3:
            return
        if len(self._samples) > 3:
            self._samples.pop(0)
        first, middle, last = self._samples
        if not (first.stable and middle.stable and last.stable):
            return
        if first.value != last.value or first.value == middle.value:
            return
        self.total += 1
        if len(self.events) < self.limit:
            self.events.append(
                {
                    "kind": self.kind,
                    "frame": middle.frame_id,
                    "before_after": _signature_text(first.value),
                    "flash": _signature_text(middle.value),
                }
            )


class _DeterministicVideoClock:
    def __init__(self, fps: float) -> None:
        self.period_ns = max(1, int(round(1_000_000_000.0 / max(0.001, float(fps)))))
        self.current_ns = 1_000_000_000

    def __call__(self) -> int:
        value = self.current_ns
        self.current_ns += self.period_ns
        return value


def _detection_to_dict(detection: Detection) -> dict[str, Any]:
    return {
        "bbox": [float(value) for value in detection.bbox],
        "conf": float(detection.conf),
        "cls_id": int(detection.cls_id),
        "cls_name": str(detection.cls_name),
        "refined_center_px": (
            [float(value) for value in detection.refined_center_px]
            if detection.refined_center_px is not None
            else None
        ),
        "refined_radius_px": (
            float(detection.refined_radius_px) if detection.refined_radius_px is not None else None
        ),
        "geometry_quality": float(detection.geometry_quality),
        "geometry_method": str(detection.geometry_method),
    }


def _detection_from_dict(payload: dict[str, Any]) -> Detection:
    center = payload.get("refined_center_px")
    return Detection(
        bbox=tuple(float(value) for value in payload["bbox"]),  # type: ignore[arg-type]
        conf=float(payload["conf"]),
        cls_id=int(payload["cls_id"]),
        cls_name=str(payload["cls_name"]),
        refined_center_px=(float(center[0]), float(center[1])) if center is not None else None,
        refined_radius_px=(
            float(payload["refined_radius_px"]) if payload.get("refined_radius_px") is not None else None
        ),
        geometry_quality=float(payload.get("geometry_quality", 0.45)),
        geometry_method=str(payload.get("geometry_method", "bbox")),
    )


class _DetectionCacheWriter:
    def __init__(self, path: Path, *, video: Path, fps: float) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._handle.write(
            json.dumps(
                {
                    "type": "header",
                    "schema": "bas_detection_cache_v1",
                    "video": str(video.resolve()),
                    "fps": float(fps),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._rows = 0

    def write(self, source_frame: int, detections: DetectionsFrame) -> None:
        self._handle.write(
            json.dumps(
                {
                    "type": "frame",
                    "source_frame": int(source_frame),
                    "detector_version": str(detections.detector_version),
                    "detections": [_detection_to_dict(item) for item in detections.detections],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        self._rows += 1
        if self._rows % 50 == 0:
            self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


class _CachedDetectionService:
    def __init__(self, path: Path, *, video: Path, start_frame: int = 0) -> None:
        self.path = path.resolve()
        self.detector = SimpleNamespace(version="diagnostic_detection_cache")
        self._frames: dict[int, list[Detection]] = {}
        self._video = video.resolve()
        self._next_source_frame = max(0, int(start_frame))
        self._load()

    def _load(self) -> None:
        header: Optional[dict[str, Any]] = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if payload.get("type") == "header":
                    header = payload
                    continue
                if payload.get("type") != "frame":
                    continue
                self._frames[int(payload["source_frame"])] = [
                    _detection_from_dict(item) for item in payload.get("detections", [])
                ]
        if header is None or header.get("schema") != "bas_detection_cache_v1":
            raise RuntimeError(f"Invalid detection cache header: {self.path}")
        cached_video = Path(str(header.get("video", ""))).resolve()
        if cached_video != self._video:
            raise RuntimeError(f"Detection cache belongs to {cached_video}, not {self._video}")

    def reset_cache(self) -> None:
        return

    def process(self, frame: Any, *args: Any, **kwargs: Any) -> DetectionsFrame:
        source_frame = self._next_source_frame
        self._next_source_frame += 1
        if source_frame not in self._frames:
            raise RuntimeError(f"Detection cache has no source frame {source_frame}: {self.path}")
        return DetectionsFrame(
            frame_id=int(frame.frame_id),
            ts_cam_ns=int(frame.ts_cam_ns),
            detections=list(self._frames[source_frame]),
            detector_version="diagnostic_detection_cache",
            latency_ms=0.0,
        )


def _video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = max(0, int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
        return {
            "width": max(0, int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))),
            "height": max(0, int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))),
            "fps": fps,
            "frames": frames,
            "duration_s": float(frames / fps) if fps > 0.0 else None,
        }
    finally:
        cap.release()


def _route_signature(out: PipelineOutput) -> tuple[Any, ...]:
    best = out.plan.best
    best_key = None
    if best is not None:
        best_key = (
            int(best.target_track_id),
            int(best.pocket_index),
            str(best.target_group),
        )
    return (
        best_key,
        tuple(str(line.label or "unlabeled") for line in out.overlay.lines),
        tuple(str(circle.stability_key or "unkeyed") for circle in out.overlay.circles),
    )


def _primitive_image(out: PipelineOutput, stabilizer: DisplayGeometryStabilizer) -> np.ndarray:
    primitive_overlay = replace(out.overlay, labels=[], texts=[], suppress_star_formula=True)
    shown = stabilize_projection_overlay(primitive_overlay, stabilizer)
    image = render_overlay_image(shown)
    return np.max(image, axis=2).astype(np.uint8, copy=False)


def _circle_image(
    out: PipelineOutput,
    center_stabilizer: DisplayGeometryStabilizer,
    shape_stabilizer: DisplayGeometryStabilizer,
) -> tuple[np.ndarray, tuple[int, ...]]:
    frame = out.frame.image
    if frame is None:
        return np.zeros((1, 1), dtype=np.uint8), ()
    image = np.zeros(frame.shape[:2], dtype=np.uint8)
    ids: list[int] = []
    for track in confirmed_tracks(out.tracks.tracks):
        track_id = int(track.track_id)
        ids.append(track_id)
        center = center_stabilizer.stabilize(("center", track_id), np.asarray(track.center_px, dtype=np.float32))
        radius = shape_stabilizer.stabilize(
            ("radius", track_id),
            np.asarray([float(track.radius_px)], dtype=np.float32),
        )
        draw_subpixel_circle(image, center, max(4.0, float(radius[0])), 255, 2)
    return image, tuple(sorted(ids))


def _changed_pixels(first: np.ndarray, second: np.ndarray, *, threshold: int = 8) -> int:
    if first.shape != second.shape:
        return max(int(first.size), int(second.size))
    return int(np.count_nonzero(cv2.absdiff(first, second) >= int(threshold)))


class _PixelFlashDetector:
    def __init__(self, kind: str, *, return_tolerance: int = 100, change_floor: int = 100, limit: int = 200) -> None:
        self.kind = str(kind)
        self.return_tolerance = max(0, int(return_tolerance))
        self.change_floor = max(1, int(change_floor))
        self.limit = max(1, int(limit))
        self._samples: list[tuple[int, bool, np.ndarray]] = []
        self.events: list[dict[str, Any]] = []
        self.total = 0

    def push(self, frame_id: int, image: np.ndarray, *, stable: bool) -> None:
        self._samples.append((int(frame_id), bool(stable), image.copy()))
        if len(self._samples) < 3:
            return
        if len(self._samples) > 3:
            self._samples.pop(0)
        first, middle, last = self._samples
        if not (first[1] and middle[1] and last[1]):
            return
        return_diff = _changed_pixels(first[2], last[2])
        into_diff = _changed_pixels(first[2], middle[2])
        out_diff = _changed_pixels(middle[2], last[2])
        if return_diff > self.return_tolerance or min(into_diff, out_diff) < self.change_floor:
            return
        self.total += 1
        if len(self.events) < self.limit:
            self.events.append(
                {
                    "kind": self.kind,
                    "frame": middle[0],
                    "changed_pixels_in": into_diff,
                    "changed_pixels_out": out_diff,
                    "return_difference_pixels": return_diff,
                }
            )


def _signed_polygon_distance(point: tuple[float, float], polygon: Iterable[Iterable[float]]) -> Optional[float]:
    contour = np.asarray(list(polygon), dtype=np.float32).reshape((-1, 1, 2))
    if contour.shape[0] < 3:
        return None
    return float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))


def _nearest_pocket(point: tuple[float, float], pockets: Iterable[Iterable[float]]) -> tuple[Optional[int], Optional[float]]:
    values = np.asarray(list(pockets), dtype=np.float32).reshape((-1, 2))
    if values.size == 0:
        return None, None
    p = np.asarray(point, dtype=np.float32).reshape((1, 2))
    distances = np.linalg.norm(values - p, axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _polygon_self_intersections(points: Iterable[Iterable[float]]) -> list[dict[str, int]]:
    polygon = np.asarray(list(points), dtype=np.float64).reshape((-1, 2))
    count = int(polygon.shape[0])
    if count < 4:
        return []

    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float(np.cross(b - a, c - a))

    def intersects(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        ab_c = orientation(a, b, c)
        ab_d = orientation(a, b, d)
        cd_a = orientation(c, d, a)
        cd_b = orientation(c, d, b)
        return (ab_c * ab_d < -1e-8) and (cd_a * cd_b < -1e-8)

    found: list[dict[str, int]] = []
    for first in range(count):
        a = polygon[first]
        b = polygon[(first + 1) % count]
        for second in range(first + 1, count):
            if second in {first, (first + 1) % count} or (second + 1) % count == first:
                continue
            c = polygon[second]
            d = polygon[(second + 1) % count]
            if intersects(a, b, c, d):
                found.append({"edge_a": first, "edge_b": second})
    return found


def _static_geometry_report(pipeline: RuntimePipeline) -> dict[str, Any]:
    table = pipeline.calibration.table
    polygons = {
        "inner": list(table.inner_polygon_mm),
        "projection_visible": list(table.projection_visible_polygon_mm),
        "center_playable": list(table.center_playable_polygon_mm),
    }
    polygon_report: dict[str, Any] = {}
    for name, points in polygons.items():
        values = np.asarray(points, dtype=np.float32).reshape((-1, 2)) if points else np.zeros((0, 2), dtype=np.float32)
        polygon_report[name] = {
            "point_count": int(values.shape[0]),
            "area_mm2": float(abs(cv2.contourArea(values))) if values.shape[0] >= 3 else 0.0,
            "self_intersections": _polygon_self_intersections(values),
            "finite": bool(np.isfinite(values).all()),
        }
    judgment = np.asarray(table.pockets_mm, dtype=np.float32).reshape((-1, 2))
    planning = np.asarray(table.planning_pockets_mm, dtype=np.float32).reshape((-1, 2))
    visible = np.asarray(table.projection_visible_pockets_mm, dtype=np.float32).reshape((-1, 2))
    pocket_rows: list[dict[str, Any]] = []
    for index in range(max(len(judgment), len(planning), len(visible))):
        row: dict[str, Any] = {"pocket_index": index}
        for name, values in (("judgment", judgment), ("planning", planning), ("projection_visible", visible)):
            if index < len(values):
                row[name + "_mm"] = [float(values[index, 0]), float(values[index, 1])]
        if index < len(judgment) and index < len(planning):
            row["planning_vs_judgment_mm"] = float(np.linalg.norm(planning[index] - judgment[index]))
        if index < len(judgment) and index < len(visible):
            row["visible_vs_judgment_mm"] = float(np.linalg.norm(visible[index] - judgment[index]))
        pocket_rows.append(row)
    return {
        "table_size_mm": [float(table.width_mm), float(table.height_mm)],
        "ball_diameter_mm": float(table.ball_diameter_mm),
        "polygons": polygon_report,
        "pockets": pocket_rows,
        "runtime_pocket_curve_count": len(getattr(pipeline, "_last_pocket_curves_mm", [])),
        "runtime_table_edge_point_count": len(getattr(pipeline, "_last_table_edge_polygon_mm", [])),
    }


def _event_row(out: PipelineOutput, event: Any, source_frame: int) -> dict[str, Any]:
    return {
        "source_frame": int(source_frame),
        "pipeline_frame": int(out.frame.frame_id),
        "time_s": float(source_frame / max(0.001, float(out.frame.exposure_meta.get("diagnostic_video_fps", 1.0)))),
        "name": str(event.name),
        "confidence": float(event.confidence),
        "payload": _json_value(event.payload),
    }


def _timeline_row(pipeline: RuntimePipeline, out: PipelineOutput, source_frame: int, fps: float) -> dict[str, Any]:
    debug_snapshot: dict[str, Any] = {}
    snapshot = getattr(pipeline.state_machine, "debug_snapshot", None)
    if callable(snapshot):
        debug_snapshot = dict(snapshot() or {})
    cue_debug = getattr(pipeline.planner.cue_sector, "last_debug_view", None)
    return {
        "source_frame": int(source_frame),
        "time_s": float(source_frame / fps),
        "phase": str(out.state.phase),
        "turn_target_group": out.state.turn_target_group,
        "state_debug": _json_value(
            {
                key: debug_snapshot.get(key)
                for key in (
                    "signals",
                    "counters",
                    "visible_group_counts",
                    "visible_counts",
                    "target_resolution",
                    "raw_turn_target_group",
                )
                if key in debug_snapshot
            }
        ),
        "detections": [
            {
                "class": str(item.cls_name),
                "confidence": float(item.conf),
                "center_px": [float(item.center[0]), float(item.center[1])],
                "geometry_quality": float(item.geometry_quality),
                "geometry_method": str(item.geometry_method),
            }
            for item in out.detections.detections
        ],
        "tracks": [
            {
                "track_id": int(item.track_id),
                "group": str(item.group),
                "confirmed": bool(item.confirmed),
                "visibility": str(item.visibility),
                "age": int(item.age),
                "lost_frames": int(item.lost_frames),
                "quality": float(item.quality),
                "geometry_quality": float(item.geometry_quality),
                "geometry_method": str(item.geometry_method),
                "center_px": [float(item.center_px[0]), float(item.center_px[1])],
                "center_mm": (
                    [float(item.center_mm[0]), float(item.center_mm[1])]
                    if item.center_mm is not None
                    else None
                ),
                "velocity_px_s": [float(item.velocity_px_s[0]), float(item.velocity_px_s[1])],
                "velocity_mm_s": (
                    [float(item.velocity_mm_s[0]), float(item.velocity_mm_s[1])]
                    if item.velocity_mm_s is not None
                    else None
                ),
            }
            for item in out.tracks.tracks
        ],
        "planner": {
            "cue_sector_status": str(pipeline.planner.cue_sector.last_status),
            "cue_sector_debug": _json_value(cue_debug),
            "target_lock_status": str(out.plan.target_lock_status),
            "target_shot_status": str(out.plan.target_shot_status),
            "candidates": [
                {
                    "candidate_id": str(item.candidate_id),
                    "target_track_id": int(item.target_track_id),
                    "target_group": str(item.target_group),
                    "pocket_index": int(item.pocket_index),
                    "score": float(item.score),
                    "risk": float(item.risk),
                    "cut_angle_deg": float(item.cut_angle_deg),
                    "cue_ball": list(item.cue_ball),
                    "object_ball": list(item.object_ball),
                    "ghost_ball": list(item.ghost_ball),
                    "pocket_point": list(item.pocket_point),
                    "explanation": _json_value(item.explanation),
                }
                for item in out.plan.candidates
            ],
        },
        "overlay": {
            "line_labels": [str(item.label or "unlabeled") for item in out.overlay.lines],
            "circle_keys": [str(item.stability_key or "unkeyed") for item in out.overlay.circles],
        },
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def diagnose_video(
    video_path: str | Path,
    *,
    config_path: str | Path = "configs/default.yaml",
    output_path: str | Path | None = None,
    target_group: str = "solid",
    start_frame: int = 0,
    max_frames: int = 0,
    progress_every: int = 250,
    write_detections: str | Path | None = None,
    read_detections: str | Path | None = None,
) -> dict[str, Any]:
    from ..capture import service as capture_service_module

    video = Path(video_path).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    metadata = _video_metadata(video)
    fps = float(metadata["fps"] or 0.0)
    if fps <= 0.0:
        raise RuntimeError(f"Video reports invalid FPS: {fps}")

    group = str(target_group).strip().lower()
    if group not in {"solid", "stripe", "black"}:
        raise ValueError("target_group must be solid, stripe, or black")

    cfg = UserSettings.load().apply_to_config(AppConfig.load(config_path)).resolve_paths()
    configured_detector_backend = str(cfg.detector.backend)
    configured_model_path = str(cfg.detector.model_path)
    cfg.camera.backend = "video"
    cfg.camera.video_path = str(video)
    cfg.replay.enabled = False
    cfg.learning.collect_enabled = False
    cfg.projection.auto_pocket_animation_enabled = False
    cfg.projection.auto_victory_animation_enabled = False
    # A recorded frame represents 1 / encoded_fps seconds. Since encoded_fps is
    # below the live 15 Hz detector cap, every recorded frame is detector-due.
    # Disable the wall-clock limiter so offline CPU/GPU speed cannot alter which
    # source frames are inspected.
    cfg.detector.detect_fps_limit_hz = 0.0
    if read_detections is not None:
        cfg.detector.backend = "disabled"

    clock = _DeterministicVideoClock(fps)
    original_clock: Callable[[], int] = capture_service_module.monotonic_ns
    capture_service_module.monotonic_ns = clock

    prepare_runtime_environment()
    preload_torch_for_backend(cfg.detector.backend)
    pipeline: Optional[RuntimePipeline] = None
    cache_writer: Optional[_DetectionCacheWriter] = None
    started = time.perf_counter()
    pocket_observations: Any = None
    try:
        pipeline = RuntimePipeline(cfg)
        if read_detections is not None:
            pipeline.detector = _CachedDetectionService(  # type: ignore[assignment]
                Path(read_detections),
                video=video,
                start_frame=start_frame,
            )
        if write_detections is not None:
            cache_writer = _DetectionCacheWriter(Path(write_detections), video=video, fps=fps)
        if start_frame > 0:
            pipeline.seek_video(int(start_frame))
        pipeline.state_machine.set_turn_target_group(group, reason="diagnostic_video_replay")

        original_observer_update = pipeline.pocket_observer.update

        def observer_update(*args: Any, **kwargs: Any) -> Any:
            nonlocal pocket_observations
            pocket_observations = original_observer_update(*args, **kwargs)
            return pocket_observations

        pipeline.pocket_observer.update = observer_update  # type: ignore[method-assign]

        route_freeze = MotionRouteFreezeController(cfg.planner)
        projection_stabilizer = DisplayGeometryStabilizer()
        preview_center_stabilizer = DisplayGeometryStabilizer()
        preview_shape_stabilizer = DisplayGeometryStabilizer(deadband_px=DISPLAY_SHAPE_DEADBAND_PX)

        semantic_detectors = {
            "ball_circle_set": TripletFlashDetector("ball_circle_set"),
            "route_primitives": TripletFlashDetector("route_primitives"),
            "route_presence": TripletFlashDetector("route_presence"),
        }
        pixel_detectors = {
            "projection_route_pixels": _PixelFlashDetector("projection_route_pixels"),
            "preview_ball_circle_pixels": _PixelFlashDetector(
                "preview_ball_circle_pixels",
                return_tolerance=160,
                change_floor=80,
            ),
        }

        phase_counts: Counter[str] = Counter()
        event_counts: Counter[str] = Counter()
        route_counts: Counter[str] = Counter()
        pocket_summary: dict[int, Counter[str]] = defaultdict(Counter)
        pocket_tracks: dict[int, set[int]] = defaultdict(set)
        event_rows: list[dict[str, Any]] = []
        timeline_rows: list[dict[str, Any]] = []
        geometry_risks: list[dict[str, Any]] = []
        min_center_margin: Optional[float] = None
        route_frame_count = 0
        stable_route_frame_count = 0
        processed = 0
        current_source_frame = int(start_frame)
        previous_visibility: dict[int, str] = {}
        previous_tracks: dict[int, Any] = {}
        loss_episodes: list[dict[str, Any]] = []
        open_losses: dict[int, dict[str, Any]] = {}
        report_geometry: Optional[dict[str, Any]] = None

        while True:
            if max_frames > 0 and processed >= int(max_frames):
                break
            raw_out = pipeline.step()
            if raw_out is None:
                break
            current_source_frame = int(start_frame + processed)
            raw_out.frame.exposure_meta["diagnostic_video_fps"] = fps
            raw_out.frame.exposure_meta["diagnostic_source_frame"] = current_source_frame
            if cache_writer is not None:
                cache_writer.write(current_source_frame, raw_out.detections)
            decision = route_freeze.update(raw_out.state, raw_out.plan, raw_out.overlay)
            out = raw_out
            if decision.plan is not raw_out.plan or decision.overlay is not raw_out.overlay:
                out = replace(raw_out, plan=decision.plan, overlay=decision.overlay)
            if report_geometry is None:
                report_geometry = _static_geometry_report(pipeline)

            stable = str(out.state.phase) in STABLE_PHASES
            phase_counts[str(out.state.phase)] += 1
            circle_image, circle_ids = _circle_image(out, preview_center_stabilizer, preview_shape_stabilizer)
            route_signature = _route_signature(out)
            has_route = bool(out.overlay.lines or out.overlay.circles)
            route_key = "none"
            if out.plan.best is not None:
                route_key = f"target={int(out.plan.best.target_track_id)};pocket={int(out.plan.best.pocket_index)}"
                route_frame_count += 1
                if stable:
                    stable_route_frame_count += 1
            route_counts[route_key] += 1
            timeline_rows.append(_timeline_row(pipeline, out, current_source_frame, fps))

            semantic_detectors["ball_circle_set"].push(current_source_frame, circle_ids, stable=stable)
            semantic_detectors["route_primitives"].push(current_source_frame, route_signature, stable=stable)
            semantic_detectors["route_presence"].push(current_source_frame, has_route, stable=stable)
            pixel_detectors["preview_ball_circle_pixels"].push(current_source_frame, circle_image, stable=stable)
            pixel_detectors["projection_route_pixels"].push(
                current_source_frame,
                _primitive_image(out, projection_stabilizer),
                stable=stable,
            )

            confirmed = confirmed_tracks(out.tracks.tracks)
            current_tracks = {int(track.track_id): track for track in confirmed}
            for track_id, track in current_tracks.items():
                visibility = str(track.visibility)
                previous = previous_visibility.get(track_id)
                if previous == "visible" and visibility != "visible":
                    center_mm = track.center_mm
                    if center_mm is None and track_id in previous_tracks:
                        center_mm = previous_tracks[track_id].center_mm
                    pocket_index, pocket_distance = (None, None)
                    margin = None
                    if center_mm is not None:
                        point = (float(center_mm[0]), float(center_mm[1]))
                        pocket_index, pocket_distance = _nearest_pocket(point, pipeline.calibration.table.pockets_mm)
                        margin = _signed_polygon_distance(point, pipeline.calibration.table.center_playable_polygon_mm)
                    open_losses[track_id] = {
                        "track_id": track_id,
                        "group": str(track.group),
                        "start_frame": current_source_frame,
                        "start_time_s": float(current_source_frame / fps),
                        "nearest_pocket": pocket_index,
                        "pocket_distance_mm": pocket_distance,
                        "center_playable_margin_mm": margin,
                    }
                elif previous is not None and previous != "visible" and visibility == "visible":
                    episode = open_losses.pop(track_id, None)
                    if episode is not None:
                        episode["end_frame"] = current_source_frame
                        episode["duration_frames"] = current_source_frame - int(episode["start_frame"])
                        episode["outcome"] = "reappeared"
                        loss_episodes.append(episode)
                previous_visibility[track_id] = visibility

                if track.center_mm is not None and visibility == "visible":
                    point = (float(track.center_mm[0]), float(track.center_mm[1]))
                    margin = _signed_polygon_distance(point, pipeline.calibration.table.center_playable_polygon_mm)
                    if margin is not None:
                        min_center_margin = margin if min_center_margin is None else min(min_center_margin, margin)
                        if margin < -1.0 and len(geometry_risks) < 500:
                            pocket_index, pocket_distance = _nearest_pocket(point, pipeline.calibration.table.pockets_mm)
                            geometry_risks.append(
                                {
                                    "source_frame": current_source_frame,
                                    "time_s": float(current_source_frame / fps),
                                    "track_id": track_id,
                                    "group": str(track.group),
                                    "center_mm": [point[0], point[1]],
                                    "center_playable_margin_mm": margin,
                                    "nearest_pocket": pocket_index,
                                    "pocket_distance_mm": pocket_distance,
                                }
                            )

            removed_ids = set(previous_tracks) - set(current_tracks)
            for track_id in removed_ids:
                episode = open_losses.pop(track_id, None)
                if episode is not None:
                    episode["end_frame"] = current_source_frame
                    episode["duration_frames"] = current_source_frame - int(episode["start_frame"])
                    episode["outcome"] = "removed"
                    loss_episodes.append(episode)
                previous_visibility.pop(track_id, None)
            previous_tracks = current_tracks

            for event in out.state.events:
                name = str(event.name)
                event_counts[name] += 1
                event_rows.append(_event_row(out, event, current_source_frame))

            if pocket_observations is not None:
                for observation in list(getattr(pocket_observations, "observations", []) or []):
                    pocket_index = int(observation.pocket_index)
                    summary = pocket_summary[pocket_index]
                    summary["frames"] += 1
                    if bool(observation.inward_crossing):
                        summary["inward_crossing_frames"] += 1
                    if bool(observation.outward_crossing):
                        summary["outward_crossing_frames"] += 1
                    if bool(observation.lip_occupied):
                        summary["lip_occupied_frames"] += 1
                    if bool(observation.clear):
                        summary["clear_frames"] += 1
                    summary["max_confidence_milli"] = max(
                        int(summary["max_confidence_milli"]),
                        int(round(float(observation.confidence) * 1000.0)),
                    )
                    pocket_tracks[pocket_index].update(int(value) for value in observation.associated_track_ids)

            processed += 1
            if progress_every > 0 and processed % int(progress_every) == 0:
                elapsed = max(0.001, time.perf_counter() - started)
                print(
                    f"diagnostic progress: {processed} frames, source={current_source_frame}, "
                    f"{processed / elapsed:.2f} fps, route_frames={route_frame_count}, events={sum(event_counts.values())}",
                    flush=True,
                )

        for track_id, episode in list(open_losses.items()):
            episode["end_frame"] = current_source_frame
            episode["duration_frames"] = current_source_frame - int(episode["start_frame"])
            episode["outcome"] = "open_at_end"
            loss_episodes.append(episode)

        semantic_failures = {
            name: detector.total
            for name, detector in semantic_detectors.items()
            if detector.total > 0
        }
        pixel_failures = {
            name: detector.total
            for name, detector in pixel_detectors.items()
            if detector.total > 0
        }
        flicker_event_rows = [
            *[item for detector in semantic_detectors.values() for item in detector.events],
            *[item for detector in pixel_detectors.values() for item in detector.events],
        ]
        failure_count = sum(semantic_failures.values()) + sum(pixel_failures.values())
        elapsed = max(0.001, time.perf_counter() - started)
        report = {
            "schema": "bas_overlay_video_diagnosis_v1",
            "verdict": "FAIL" if failure_count else "PASS",
            "failure_count": int(failure_count),
            "video": {
                "path": str(video),
                "sha256": _hash_file(video),
                **metadata,
            },
            "simulation": {
                "config_path": str(Path(config_path).resolve()),
                "target_group": group,
                "start_frame": int(start_frame),
                "max_frames": int(max_frames),
                "processed_frames": int(processed),
                "deterministic_frame_period_ns": int(clock.period_ns),
                "elapsed_wall_s": float(elapsed),
                "processing_fps": float(processed / elapsed),
                "detector_backend": configured_detector_backend,
                "model_path": configured_model_path,
                "detect_every_recorded_frame": True,
                "detection_cache_read": str(Path(read_detections).resolve()) if read_detections is not None else None,
                "detection_cache_written": str(Path(write_detections).resolve()) if write_detections is not None else None,
                "state_engine": str(cfg.state.engine),
                "route_freeze_enabled": bool(cfg.planner.route_freeze_enabled),
            },
            "flicker": {
                "semantic_failures": semantic_failures,
                "pixel_failures": pixel_failures,
                "events": sorted(flicker_event_rows, key=lambda item: (int(item["frame"]), str(item["kind"]))),
            },
            "route": {
                "route_frame_count": int(route_frame_count),
                "stable_route_frame_count": int(stable_route_frame_count),
                "signature_counts": dict(route_counts.most_common()),
            },
            "state": {
                "phase_counts": dict(phase_counts),
                "event_counts": dict(event_counts),
                "events": event_rows,
                "timeline": timeline_rows,
            },
            "geometry": {
                **(report_geometry or {}),
                "minimum_visible_ball_center_margin_mm": min_center_margin,
                "outside_center_playable_samples": geometry_risks,
            },
            "tracking": {
                "loss_episode_count": len(loss_episodes),
                "loss_episodes": loss_episodes,
            },
            "pockets": {
                "observation_summary": {
                    str(index): {
                        **{key: int(value) for key, value in summary.items()},
                        "max_confidence": float(summary["max_confidence_milli"] / 1000.0),
                        "associated_track_ids": sorted(pocket_tracks[index]),
                    }
                    for index, summary in sorted(pocket_summary.items())
                },
                "judgment_event_count": int(sum(event_counts[name] for name in POCKET_EVENT_NAMES)),
            },
        }
        destination = Path(output_path) if output_path is not None else (
            Path(".tmp") / "overlay_video_diagnosis" / f"{video.stem}_{group}.json"
        )
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(_json_value(report), ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(destination)
        print(
            json.dumps(
                {
                    "verdict": report["verdict"],
                    "processed_frames": processed,
                    "failure_count": failure_count,
                    "semantic_failures": semantic_failures,
                    "pixel_failures": pixel_failures,
                    "route_frame_count": route_frame_count,
                    "event_counts": dict(event_counts),
                    "report_path": str(destination),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return report
    finally:
        if cache_writer is not None:
            cache_writer.close()
        if pipeline is not None:
            pipeline.close()
        capture_service_module.monotonic_ns = original_clock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对无画线录像执行真实 BAS 链路回放并检查球圈、路线和袋口隐患。")
    parser.add_argument("--video", type=Path, required=True, help="无画线 MP4 输入。")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="基础 YAML 配置。")
    parser.add_argument("--output", type=Path, default=None, help="JSON 报告路径；默认写入 .tmp。")
    parser.add_argument("--target-group", choices=("solid", "stripe", "black"), default="solid")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示处理到视频结尾。")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--write-detections", type=Path, default=None, help="逐帧写出可快速重放的检测 JSONL。")
    parser.add_argument("--read-detections", type=Path, default=None, help="读取已有检测 JSONL，跳过 YOLO。")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = diagnose_video(
        args.video,
        config_path=args.config,
        output_path=args.output,
        target_group=args.target_group,
        start_frame=max(0, int(args.start_frame)),
        max_frames=max(0, int(args.max_frames)),
        progress_every=max(0, int(args.progress_every)),
        write_detections=args.write_detections,
        read_detections=args.read_detections,
    )
    return 1 if report["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
