from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .calibration import CalibrationService, create_calibration_service
from .capture import CaptureService, create_capture_service
from .config import AppConfig
from .geometry import TableGeometry
from .geometry_runtime import RuntimeGeometryReloader
from .learning import LearningSampleRecorder
from .logging_config import configure_logging
from .perception import DetectService, create_detector
from .planning import GeometryPhysicsPlanner
from .projection import OverlayBuilder
from .projection.star_formula import StarFormulaConfig
from .replay import ReplayRecorder
from .schemas import DetectionsFrame, FramePacket, MatchStateFrame, ProjectionOverlay, ShotPlan, TracksFrame
from .state import MatchStateMachine
from .tracking import TemporalTracker

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    frame: FramePacket
    detections: DetectionsFrame
    tracks: TracksFrame
    state: MatchStateFrame
    plan: ShotPlan
    overlay: ProjectionOverlay


class RuntimePipeline:
    def __init__(self, config: AppConfig, star_formula: StarFormulaConfig | None = None):
        self.config = config
        self.capture: CaptureService = create_capture_service(config.camera)
        self.calibration: CalibrationService = create_calibration_service(
            config.calibration,
            frame_undistorted=self.capture.frame_distortion_corrected,
        )
        self.geometry_reloader = RuntimeGeometryReloader()
        self.geometry, _ = self.geometry_reloader.refresh(
            config.geometry.outline_path,
            config.geometry.inline_path,
            config.geometry.pocket_path,
        )
        self.detector = DetectService(
            create_detector(config.detector),
            detect_interval_frames=config.detector.detect_interval_frames,
            detect_fps_limit_hz=config.detector.detect_fps_limit_hz,
        )
        self.tracker = TemporalTracker(config.tracker)
        self.state_machine = MatchStateMachine(config.state)
        self.planner = GeometryPhysicsPlanner(config.planner, self.calibration, learning_config=config.learning)
        self.overlay_builder = OverlayBuilder(config.projection, self.calibration, star_formula=star_formula)
        self.recorder: Optional[ReplayRecorder] = ReplayRecorder(config.replay) if config.replay.enabled else None
        self.learning_recorder: Optional[LearningSampleRecorder] = LearningSampleRecorder(config.learning) if config.learning.collect_enabled else None
        self._last_tracks: Optional[TracksFrame] = None
        self._last_state: Optional[MatchStateFrame] = None
        self._last_plan: Optional[ShotPlan] = None
        self._last_overlay: Optional[ProjectionOverlay] = None
        self.last_timings_ms: dict[str, float] = {}
        self._processed_frames = 0
        self._cached_detection_frames = 0
        LOGGER.info("Capture opened: %s", self.capture.info())
        LOGGER.info("Calibration version: %s", self.calibration.calib_version)

    def step(self) -> Optional[PipelineOutput]:
        total_start = time.perf_counter()
        stage_start = total_start
        self._refresh_geometry_if_needed()
        frame = self.capture.read()
        capture_ms = (time.perf_counter() - stage_start) * 1000.0
        if frame is None:
            self.last_timings_ms = {
                "capture_ms": float(capture_ms),
                "total_ms": float((time.perf_counter() - total_start) * 1000.0),
            }
            return None
        frame.calib_version = self.calibration.calib_version
        stage_start = time.perf_counter()
        self._update_table_geometry_for_frame(frame)
        mask = self._camera_table_mask(frame)
        geometry_ms = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        detections = self.detector.process(frame, mask_polygon=mask)
        detect_ms = (time.perf_counter() - stage_start) * 1000.0
        cached = detections.detector_version.endswith(":cached")
        stage_start = time.perf_counter()
        if cached and self._last_tracks is not None:
            tracks = replace(self._last_tracks, frame_id=frame.frame_id, ts_cam_ns=frame.ts_cam_ns, latency_ms=0.0)
        else:
            tracks = self._enrich_tracks_with_table_units(self.tracker.update(detections))
            self._last_tracks = tracks
        track_ms = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        if cached and self._last_state is not None and self._last_plan is not None and self._last_overlay is not None:
            state = replace(self._last_state, frame_id=frame.frame_id, ts_cam_ns=frame.ts_cam_ns, events=[])
            plan = replace(self._last_plan, frame_id=frame.frame_id, ts_cam_ns=frame.ts_cam_ns)
            overlay = replace(self._last_overlay, frame_id=frame.frame_id)
        else:
            self.state_machine.set_table_context(
                inner_polygon_mm=self.calibration.table.inner_polygon_mm,
                pockets_mm=self.calibration.table.pockets_mm,
                ball_diameter_mm=self.calibration.table.ball_diameter_mm,
            )
            state = self.state_machine.update(tracks)
            plan = self.planner.plan(state, frame_bgr=frame.image)
            overlay = self.overlay_builder.from_plan(plan)
            self._last_state = state
            self._last_plan = plan
            self._last_overlay = overlay
        state_plan_overlay_ms = (time.perf_counter() - stage_start) * 1000.0
        out = PipelineOutput(frame=frame, detections=detections, tracks=tracks, state=state, plan=plan, overlay=overlay)
        stage_start = time.perf_counter()
        self._record(out)
        record_ms = (time.perf_counter() - stage_start) * 1000.0
        self._processed_frames += 1
        if cached:
            self._cached_detection_frames += 1
        total_ms = (time.perf_counter() - total_start) * 1000.0
        self.last_timings_ms = {
            "capture_ms": float(capture_ms),
            "geometry_ms": float(geometry_ms),
            "detect_ms": float(detect_ms),
            "track_ms": float(track_ms),
            "state_plan_overlay_ms": float(state_plan_overlay_ms),
            "record_ms": float(record_ms),
            "total_ms": float(total_ms),
            "detect_cached": 1.0 if cached else 0.0,
            "detect_cached_ratio": float(self._cached_detection_frames / max(1, self._processed_frames)),
        }
        return out

    def close(self) -> None:
        self.capture.release()
        if self.recorder is not None:
            self.recorder.close()
        if self.learning_recorder is not None:
            self.learning_recorder.close()

    def _refresh_geometry_if_needed(self) -> None:
        geometry, changed = self.geometry_reloader.refresh(
            self.config.geometry.outline_path,
            self.config.geometry.inline_path,
            self.config.geometry.pocket_path,
        )
        if not changed:
            return
        self.geometry = geometry
        self._last_state = None
        self._last_plan = None
        self._last_overlay = None
        LOGGER.info(
            "Geometry hot-reloaded: outline=%s inline=%s pocket=%s empty=%s",
            self.config.geometry.outline_path,
            self.config.geometry.inline_path,
            self.config.geometry.pocket_path,
            self.geometry.is_empty,
        )

    def _camera_table_mask(self, frame: FramePacket) -> Optional[np.ndarray]:
        if frame.image is not None and not self.geometry.is_empty:
            h, w = frame.image.shape[:2]
            outer, _, _ = self.geometry.scaled(w, h)
            if outer.shape[0] >= 3:
                return outer.astype(np.float32)
        poly = self.calibration.projection.table_polygon_cam
        if poly is not None and poly.shape[0] >= 3:
            return poly.astype(np.float32)
        return None

    def _update_table_geometry_for_frame(self, frame: FramePacket) -> None:
        if frame.image is None or self.geometry.is_empty:
            return
        h, w = frame.image.shape[:2]
        _, inner_px, pockets_px = self.geometry.scaled(w, h)
        if inner_px.shape[0] >= 3:
            inner_mm = self.calibration.camera_px_to_table_mm(inner_px)
            self.calibration.table.inner_polygon_mm = [(float(x), float(y)) for x, y in inner_mm]
        pocket_points = []
        for pocket in pockets_px:
            if pocket.shape[0] >= 2:
                center = np.mean(pocket, axis=0).reshape((1, 2)).astype(np.float32)
                center_mm = self.calibration.camera_px_to_table_mm(center)[0]
                pocket_points.append((float(center_mm[0]), float(center_mm[1])))
        if pocket_points:
            self.calibration.table.pockets_mm = pocket_points

    def _enrich_tracks_with_table_units(self, tracks: TracksFrame) -> TracksFrame:
        enriched = []
        dt = 0.1
        for track in tracks.tracks:
            try:
                center_px = np.asarray([track.center_px], dtype=np.float32)
                center_mm = self.calibration.camera_px_to_table_mm(center_px)[0]
                v_px = np.asarray(track.velocity_px_s, dtype=np.float32)
                edge_px = np.asarray([[track.center_px[0] + float(v_px[0]) * dt, track.center_px[1] + float(v_px[1]) * dt]], dtype=np.float32)
                edge_mm = self.calibration.camera_px_to_table_mm(edge_px)[0]
                velocity_mm = (edge_mm - center_mm) / dt
                radius_mm = self.calibration.pixel_radius_to_mm(track.center_px, track.radius_px)
                enriched.append(
                    replace(
                        track,
                        center_mm=(float(center_mm[0]), float(center_mm[1])),
                        velocity_mm_s=(float(velocity_mm[0]), float(velocity_mm[1])),
                        radius_mm=float(radius_mm),
                    )
                )
            except Exception:
                enriched.append(track)
        return replace(tracks, tracks=enriched)

    def _record(self, out: PipelineOutput) -> None:
        if self.recorder is None:
            if self.learning_recorder is not None:
                self.learning_recorder.observe(out.state, out.plan)
            return
        self.recorder.write_frame_packet(out.frame)
        self.recorder.write_detections(out.detections)
        self.recorder.write_tracks(out.tracks)
        self.recorder.write_state(out.state)
        self.recorder.write_plan(out.plan)
        self.recorder.write_overlay(out.overlay)
        if self.learning_recorder is not None:
            self.learning_recorder.observe(out.state, out.plan)


def run_headless(config: AppConfig, max_frames: int = 0) -> int:
    configure_logging(config.logging.directory, config.logging.level)
    pipeline = RuntimePipeline(config)
    count = 0
    start = time.perf_counter()
    try:
        while True:
            out = pipeline.step()
            if out is None:
                LOGGER.info("Capture ended.")
                break
            count += 1
            if count % 30 == 0:
                best = out.plan.best.candidate_id if out.plan.best else "none"
                LOGGER.info(
                    "frame=%s det=%s tracks=%s phase=%s best=%s",
                    out.frame.frame_id,
                    len(out.detections.detections),
                    len(out.tracks.tracks),
                    out.state.phase,
                    best,
                )
            if max_frames > 0 and count >= max_frames:
                break
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user.")
    finally:
        elapsed = max(1e-6, time.perf_counter() - start)
        LOGGER.info("Processed %s frames at %.2f FPS.", count, count / elapsed)
        pipeline.close()
    return 0


def run_qt(config: AppConfig) -> int:
    configure_logging(config.logging.directory, config.logging.level)
    from PyQt5 import QtCore, QtWidgets

    from .projection.window import ProjectionWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pipeline = RuntimePipeline(config)
    window = ProjectionWindow(config.projection)
    window.show_on_configured_screen()

    timer = QtCore.QTimer()

    def tick() -> None:
        out = pipeline.step()
        if out is None:
            timer.stop()
            pipeline.close()
            app.quit()
            return
        window.set_overlay(out.overlay)

    timer.timeout.connect(tick)
    timer.start(max(1, int(1000 / max(1, config.camera.fps))))
    try:
        return int(app.exec_())
    finally:
        timer.stop()
        pipeline.close()
