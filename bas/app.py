from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .calibration import CalibrationService, create_calibration_service
from .capture import CaptureService, create_capture_service
from .config import AppConfig
from .geometry import TableGeometry, TableGeometryLoader
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
        self.calibration: CalibrationService = create_calibration_service(config.calibration)
        self.geometry: TableGeometry = TableGeometryLoader.load_optional(
            config.geometry.outline_path,
            config.geometry.inline_path,
            config.geometry.pocket_path,
        )
        self.detector = DetectService(create_detector(config.detector))
        self.tracker = TemporalTracker(config.tracker)
        self.state_machine = MatchStateMachine(config.state)
        self.planner = GeometryPhysicsPlanner(config.planner, self.calibration)
        self.overlay_builder = OverlayBuilder(config.projection, self.calibration, star_formula=star_formula)
        self.recorder: Optional[ReplayRecorder] = ReplayRecorder(config.replay) if config.replay.enabled else None
        LOGGER.info("Capture opened: %s", self.capture.info())
        LOGGER.info("Calibration version: %s", self.calibration.calib_version)

    def step(self) -> Optional[PipelineOutput]:
        frame = self.capture.read()
        if frame is None:
            return None
        frame.calib_version = self.calibration.calib_version
        self._update_table_geometry_for_frame(frame)
        mask = self._camera_table_mask(frame)
        detections = self.detector.process(frame, mask_polygon=mask)
        tracks = self.tracker.update(detections)
        state = self.state_machine.update(tracks)
        plan = self.planner.plan(state)
        overlay = self.overlay_builder.from_plan(plan)
        out = PipelineOutput(frame=frame, detections=detections, tracks=tracks, state=state, plan=plan, overlay=overlay)
        self._record(out)
        return out

    def close(self) -> None:
        self.capture.release()
        if self.recorder is not None:
            self.recorder.close()

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

    def _record(self, out: PipelineOutput) -> None:
        if self.recorder is None:
            return
        self.recorder.write_frame_packet(out.frame)
        self.recorder.write_detections(out.detections)
        self.recorder.write_tracks(out.tracks)
        self.recorder.write_state(out.state)
        self.recorder.write_plan(out.plan)
        self.recorder.write_overlay(out.overlay)


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
