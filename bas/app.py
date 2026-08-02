from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .calibration import CalibrationService, create_setting_aware_calibration_service
from .capture import CaptureService, VideoTimelineState, create_capture_service
from .config import AppConfig
from .geometry import TableGeometry
from .geometry_runtime import RuntimeGeometryReloader
from .learning import LearningSampleRecorder
from .logging_config import configure_logging
from .operator_controls import RuntimeControlState
from .perception import (
    DetectionRegionPolicy,
    ModeAwareDetectService,
    PocketObserver,
    build_detection_region_policy,
    create_detector,
)
from .planning import GeometryPhysicsPlanner
from .projection import OverlayBuilder
from .projection.star_formula import StarFormulaConfig
from .replay import ReplayRecorder
from .schemas import DetectionsFrame, FramePacket, MatchStateFrame, ProjectionOverlay, ShotPlan, TracksFrame
from .secondary_correction import SecondaryCorrectionController
from .state import create_match_state_machine
from .table_boundaries import EdgeInsets, derive_table_boundaries
from .tracking import TemporalTracker
from .training import (
    RULES_MODE,
    TRAINING_MODE,
    NumberedBallTracker,
    TrainingOverlayBuilder,
    TrainingSession,
    TrainingStateFrame,
    normalize_operating_mode,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    frame: FramePacket
    detections: DetectionsFrame
    tracks: TracksFrame
    state: MatchStateFrame
    plan: ShotPlan
    overlay: ProjectionOverlay
    training: Optional[TrainingStateFrame] = None


class RuntimePipeline:
    def __init__(
        self,
        config: AppConfig,
        star_formula: StarFormulaConfig | None = None,
        control_state: RuntimeControlState | None = None,
    ):
        self.config = config
        self.control_state = control_state or RuntimeControlState()
        self.capture: CaptureService = create_capture_service(config.camera)
        capture_info = self.capture.info()
        self.calibration: CalibrationService = create_setting_aware_calibration_service(
            config.calibration,
            config.camera,
            frame_undistorted=self.capture.frame_distortion_corrected,
            detector_config=config.detector,
            projection_config=config.projection,
            actual_frame_size=(int(capture_info.width), int(capture_info.height)),
        )
        self.geometry_reloader = RuntimeGeometryReloader()
        self.geometry, _ = self.geometry_reloader.refresh(
            config.geometry.outline_path,
            config.geometry.inline_path,
            config.geometry.pocket_path,
        )
        self.operating_mode = normalize_operating_mode(config.training.operating_mode)
        self.detector = ModeAwareDetectService(
            config.detector,
            config.training_detector,
            initial_mode=self.operating_mode,
            detector_factory=create_detector,
        )
        self.rule_tracker = TemporalTracker(config.tracker)
        self.training_tracker = NumberedBallTracker(config.tracker)
        self.tracker = self.training_tracker if self.operating_mode == TRAINING_MODE else self.rule_tracker
        self.state_machine = create_match_state_machine(config.state)
        self.pocket_observer = PocketObserver(history_ms=int(config.state.pocket_entry_history_ms))
        self.planner = GeometryPhysicsPlanner(config.planner, self.calibration, learning_config=config.learning)
        self.secondary_correction = SecondaryCorrectionController(config.planner)
        self.overlay_builder = OverlayBuilder(config.projection, self.calibration, star_formula=star_formula)
        self.training_session = TrainingSession(
            config.training,
            state_config=config.state,
            ball_diameter_mm=float(config.calibration.ball_diameter_mm),
        )
        self.training_overlay_builder = TrainingOverlayBuilder(config.projection, self.calibration)
        self.recorder: Optional[ReplayRecorder] = ReplayRecorder(config.replay) if config.replay.enabled else None
        self.learning_recorder: Optional[LearningSampleRecorder] = LearningSampleRecorder(config.learning) if config.learning.collect_enabled else None
        self._last_tracks: Optional[TracksFrame] = None
        self._last_state: Optional[MatchStateFrame] = None
        self._last_plan: Optional[ShotPlan] = None
        self._last_overlay: Optional[ProjectionOverlay] = None
        self._last_pocket_curves_mm: list[list[tuple[float, float]]] = []
        self._last_table_edge_polygon_mm: list[tuple[float, float]] = list(
            self.calibration.table.inner_polygon_mm
        )
        self.last_timings_ms: dict[str, float] = {}
        self._processed_frames = 0
        self._cached_detection_frames = 0
        LOGGER.info("Capture opened: %s", self.capture.info())
        LOGGER.info("Calibration version: %s", self.calibration.calib_version)
        LOGGER.info("Operating mode: %s", self.operating_mode)

    def step(self) -> Optional[PipelineOutput]:
        total_start = time.perf_counter()
        stage_start = total_start
        self._refresh_geometry_if_needed()
        sync_ball_compensation = getattr(self.calibration, "sync_ball_center_compensation", None)
        if callable(sync_ball_compensation):
            sync_ball_compensation(self.config.calibration)
        projection_only_training = (
            getattr(self, "operating_mode", RULES_MODE) == TRAINING_MODE
            and self.training_session.scenario.projection_only
        )
        projection = getattr(self.calibration, "projection", None)
        projection_geometry_ready = bool(getattr(projection, "is_valid", True))
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
        if projection_only_training:
            detection_regions = None
        else:
            self._update_table_geometry_for_frame(frame)
            detection_regions = self._camera_detection_regions(frame)
        mask = detection_regions.global_polygon if detection_regions is not None else None
        geometry_ms = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        if projection_only_training:
            detections = DetectionsFrame(
                frame_id=frame.frame_id,
                ts_cam_ns=frame.ts_cam_ns,
                detections=[],
                detector_version="projection_only",
                latency_ms=0.0,
            )
        else:
            detections = self.detector.process(
                frame,
                mask_polygon=mask,
                detection_regions=detection_regions,
            )
        detect_ms = (time.perf_counter() - stage_start) * 1000.0
        cached = (
            not projection_only_training
            and detections.detector_version.endswith(":cached")
        )
        stage_start = time.perf_counter()
        if projection_only_training:
            tracks = TracksFrame(
                frame_id=frame.frame_id,
                ts_cam_ns=frame.ts_cam_ns,
                tracks=[],
                tracker_version="projection_only",
                latency_ms=0.0,
            )
        elif cached and self._last_tracks is not None:
            tracks = replace(self._last_tracks, frame_id=frame.frame_id, ts_cam_ns=frame.ts_cam_ns, latency_ms=0.0)
        else:
            tracks = self._enrich_tracks_with_table_units(self.tracker.update(detections))
            self._last_tracks = tracks
        track_ms = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        pocket_observations = None
        if (
            (
                getattr(self, "operating_mode", RULES_MODE) == TRAINING_MODE
                or bool(getattr(self.state_machine, "supports_pocket_observations", False))
            )
            and not projection_only_training
            and detection_regions is not None
            and detection_regions.ball_guard_regions
        ):
            observer_tracks = tracks
            if getattr(self, "operating_mode", RULES_MODE) == TRAINING_MODE:
                observer_tracks = self.training_session.pocket_observer_tracks(tracks)
            pocket_observations = self.pocket_observer.update(frame, observer_tracks, detection_regions)
        pocket_observer_ms = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        training_state: Optional[TrainingStateFrame] = None
        uncalibrated_preview_mode = (
            not projection_geometry_ready
            and not projection_only_training
        )
        if uncalibrated_preview_mode:
            state = MatchStateFrame(
                frame_id=frame.frame_id,
                ts_cam_ns=frame.ts_cam_ns,
                phase="CALIBRATION_REQUIRED",
                layout=list(tracks.tracks),
                confidence=1.0,
                state_version="uncalibrated_camera_preview_v1",
            )
            plan = ShotPlan(
                plan_id=f"uncalibrated_{frame.frame_id}",
                frame_id=frame.frame_id,
                ts_cam_ns=frame.ts_cam_ns,
                planner_version="calibration_required_v1",
            )
            overlay = ProjectionOverlay(
                overlay_id=f"uncalibrated_{frame.frame_id}",
                frame_id=frame.frame_id,
                projector_size=(
                    int(self.config.projection.projector_width),
                    int(self.config.projection.projector_height),
                ),
            )
        elif getattr(self, "operating_mode", RULES_MODE) == TRAINING_MODE:
            self.training_session.set_table_context(
                inner_polygon_mm=self.calibration.table.inner_polygon_mm,
                table_edge_polygon_mm=getattr(self, "_last_table_edge_polygon_mm", []),
                ball_center_reachable_polygon_mm=getattr(
                    self.calibration.table,
                    "center_playable_polygon_mm",
                    self.calibration.table.inner_polygon_mm,
                ),
                pockets_mm=self.calibration.table.pockets_mm,
                ball_diameter_mm=self.calibration.table.ball_diameter_mm,
                pocket_curves_mm=getattr(self, "_last_pocket_curves_mm", []),
            )
            training_state = self.training_session.update(tracks, pocket_observations)
            state = MatchStateFrame(
                frame_id=frame.frame_id,
                ts_cam_ns=frame.ts_cam_ns,
                phase=f"TRAINING_{training_state.phase.upper()}",
                events=list(training_state.events),
                layout=[] if projection_only_training else list(tracks.tracks),
                confidence=1.0 if training_state.setup_ready or training_state.phase in {"running", "passed", "failed"} else 0.5,
                state_version=self.training_session.version,
            )
            if projection_only_training:
                plan = ShotPlan(
                    plan_id=f"projection_training_{frame.frame_id}",
                    frame_id=frame.frame_id,
                    ts_cam_ns=frame.ts_cam_ns,
                    planner_version="projection_only_v1",
                )
                overlay = self.training_overlay_builder.build(
                    tracks,
                    training_state,
                    route_overlay=None,
                )
            else:
                expected_numbers = [int(number) for number in training_state.expected_numbers]
                explicit_target = len(expected_numbers) == 1
                target_groups = {
                    "solid" if 1 <= number <= 7 else "black" if number == 8 else "stripe" if 9 <= number <= 15 else "other"
                    for number in expected_numbers
                }
                forced_target_group = next(iter(target_groups)) if len(target_groups) == 1 else None
                plan = self.planner.plan(
                    state,
                    frame_bgr=frame.image,
                    forced_shot_mode="hook" if explicit_target else "rule",
                    forced_turn_target_group=forced_target_group,
                    forced_target_track_ids=expected_numbers,
                )
                route_overlay = self.overlay_builder.from_plan(plan)
                overlay = self.training_overlay_builder.build(
                    tracks,
                    training_state,
                    route_overlay=route_overlay,
                )
        else:
            # Cached detections still need fresh state transitions so sample collection
            # and turn resolution are not throttled down to detector refresh cadence.
            self.state_machine.set_table_context(
                inner_polygon_mm=self.calibration.table.inner_polygon_mm,
                table_edge_polygon_mm=getattr(self, "_last_table_edge_polygon_mm", []),
                ball_center_reachable_polygon_mm=getattr(
                    self.calibration.table,
                    "center_playable_polygon_mm",
                    self.calibration.table.inner_polygon_mm,
                ),
                pockets_mm=self.calibration.table.pockets_mm,
                ball_diameter_mm=self.calibration.table.ball_diameter_mm,
                pocket_curves_mm=getattr(self, "_last_pocket_curves_mm", []),
            )
            if bool(getattr(self.state_machine, "supports_pocket_observations", False)):
                state = self.state_machine.update(tracks, pocket_observations)
            else:
                state = self.state_machine.update(tracks)
            secondary_correction = getattr(self, "secondary_correction", None)
            updated_turn_group = (
                secondary_correction.advance_from_state(state, self.state_machine)
                if secondary_correction is not None
                else None
            )
            if updated_turn_group is not None:
                state = replace(state, turn_target_group=updated_turn_group)
            self.control_state.advance_from_events(state.events)
            effective_shot_mode = self.control_state.effective_shot_mode(self.config.planner.shot_mode)
            effective_turn_target_group = self.control_state.effective_turn_target_group(state.turn_target_group, effective_shot_mode)
            plan = self.planner.plan(
                state,
                frame_bgr=frame.image,
                forced_shot_mode=effective_shot_mode,
                forced_turn_target_group=effective_turn_target_group,
            )
            if secondary_correction is not None:
                secondary_correction.arm_from_plan(state, plan)
            overlay = self.overlay_builder.from_plan(plan)
        self._last_state = state
        self._last_plan = plan
        self._last_overlay = overlay
        state_plan_overlay_ms = (time.perf_counter() - stage_start) * 1000.0
        out = PipelineOutput(
            frame=frame,
            detections=detections,
            tracks=tracks,
            state=state,
            plan=plan,
            overlay=overlay,
            training=training_state,
        )
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
            "pocket_observer_ms": float(pocket_observer_ms),
            "state_plan_overlay_ms": float(state_plan_overlay_ms),
            "record_ms": float(record_ms),
            "total_ms": float(total_ms),
            "detect_cached": 1.0 if cached else 0.0,
            "detect_cached_ratio": float(self._cached_detection_frames / max(1, self._processed_frames)),
        }
        return out

    def set_operating_mode(self, mode: str) -> str:
        normalized = normalize_operating_mode(mode)
        if normalized == self.operating_mode:
            return normalized
        if not (
            normalized == TRAINING_MODE
            and self.training_session.scenario.projection_only
        ):
            self.detector.activate(normalized)
        self.operating_mode = normalized
        self.config.training.operating_mode = normalized
        self.tracker = self.training_tracker if normalized == TRAINING_MODE else self.rule_tracker
        self.tracker.reset()
        self._last_tracks = None
        self._last_state = None
        self._last_plan = None
        self._last_overlay = None
        self.pocket_observer.reset()
        if normalized == TRAINING_MODE:
            self.training_session.reset()
        else:
            reset_state = getattr(self.state_machine, "reset", None)
            if callable(reset_state):
                reset_state()
        LOGGER.info("Operating mode switched to: %s", normalized)
        return normalized

    def select_training_scenario(self, scenario_id: str) -> TrainingStateFrame:
        state = self.training_session.select_scenario(scenario_id)
        if (
            self.operating_mode == TRAINING_MODE
            and not self.training_session.scenario.projection_only
            and self.detector.mode != TRAINING_MODE
        ):
            self.detector.activate(TRAINING_MODE)
        self.pocket_observer.reset()
        return state

    def start_training(self) -> tuple[bool, TrainingStateFrame]:
        started, state = self.training_session.start()
        if started:
            self.pocket_observer.reset()
        return started, state

    def reset_training(self) -> TrainingStateFrame:
        state = self.training_session.reset()
        self.pocket_observer.reset()
        return state

    def video_timeline_state(self) -> Optional[VideoTimelineState]:
        return self.capture.video_timeline_state()

    def seek_video(self, frame_index: int) -> VideoTimelineState:
        state = self.capture.seek_video(frame_index)
        self._reset_temporal_processing_state()
        return state

    def _reset_temporal_processing_state(self) -> None:
        """Drop history that is invalid after a non-sequential video seek."""

        reset_cache = getattr(self.detector, "reset_cache", None)
        if callable(reset_cache):
            reset_cache()
        self.rule_tracker.reset()
        self.training_tracker.reset()
        self.pocket_observer.reset()
        reset_state = getattr(self.state_machine, "reset", None)
        if callable(reset_state):
            reset_state()
        self.training_session.reset()
        self.planner.target_lock.reset()
        self.planner.target_shot_mode.reset()
        self.planner.cue_sector.reset()
        self._last_tracks = None
        self._last_state = None
        self._last_plan = None
        self._last_overlay = None

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
        self._reset_temporal_processing_state()
        LOGGER.info(
            "Geometry hot-reloaded: outline=%s inline=%s pocket=%s empty=%s",
            self.config.geometry.outline_path,
            self.config.geometry.inline_path,
            self.config.geometry.pocket_path,
            self.geometry.is_empty,
        )

    def _camera_detection_regions(
        self,
        frame: FramePacket,
    ) -> DetectionRegionPolicy | None:
        if frame.image is None or frame.image.size == 0:
            return None
        geometry_reloader = getattr(self, "geometry_reloader", None)
        if geometry_reloader is not None and not bool(getattr(geometry_reloader, "is_ready", False)):
            return DetectionRegionPolicy(detection_enabled=False)
        fallback_polygon = self._camera_table_mask(frame)
        policy = build_detection_region_policy(
            frame.image.shape,
            self.geometry,
            fallback_polygon=fallback_polygon,
            ball_diameter_px_by_pocket=self._camera_ball_diameters_px(frame),
        )
        if policy.global_polygon is None and policy.ball_polygon is None and policy.cue_stick_polygon is None:
            return None
        return policy

    def _camera_ball_diameters_px(self, frame: FramePacket) -> list[float]:
        if frame.image is None or self.geometry.is_empty:
            return []
        height, width = frame.image.shape[:2]
        _, _, pocket_curves = self.geometry.scaled(width, height)
        diameter_mm = max(1.0, float(self.calibration.table.ball_diameter_mm))
        diameters: list[float] = []
        for curve in pocket_curves:
            points = np.asarray(curve, dtype=np.float32).reshape((-1, 2))
            if points.shape[0] < 2:
                continue
            center_px = np.mean(points, axis=0)
            try:
                center_mm = self.calibration.camera_px_to_table_mm(np.asarray([center_px], dtype=np.float32))[0]
                half = diameter_mm * 0.5
                samples_mm = np.asarray(
                    [
                        [center_mm[0] - half, center_mm[1]],
                        [center_mm[0] + half, center_mm[1]],
                        [center_mm[0], center_mm[1] - half],
                        [center_mm[0], center_mm[1] + half],
                    ],
                    dtype=np.float32,
                )
                samples_px = self.calibration.table_mm_to_camera_px(samples_mm)
                horizontal = float(np.linalg.norm(samples_px[1] - samples_px[0]))
                vertical = float(np.linalg.norm(samples_px[3] - samples_px[2]))
                diameter_px = float(np.median([horizontal, vertical]))
            except Exception:
                diameter_px = 0.0
            if not np.isfinite(diameter_px) or diameter_px <= 0.0:
                return []
            diameters.append(diameter_px)
        return diameters

    def _camera_table_mask(self, frame: FramePacket) -> Optional[np.ndarray]:
        if frame.image is not None and not self.geometry.is_empty:
            h, w = frame.image.shape[:2]
            outer, _, _ = self.geometry.scaled(w, h)
            if outer.shape[0] >= 3:
                return outer.astype(np.float32)
        poly = getattr(getattr(self.calibration, "projection", None), "table_polygon_cam", None)
        if poly is None:
            return None
        poly_np = np.asarray(poly, dtype=np.float32).reshape((-1, 2))
        if poly_np.shape[0] >= 3:
            return poly_np.astype(np.float32)
        return None

    def _update_table_geometry_for_frame(self, frame: FramePacket) -> None:
        if frame.image is None or self.geometry.is_empty:
            return
        projection = getattr(self.calibration, "projection", None)
        if projection is None or not bool(getattr(projection, "is_valid", False)):
            # Camera acquisition and pixel-domain detection must remain usable
            # before the first projection calibration is created.
            return
        h, w = frame.image.shape[:2]
        _, inner_px, pockets_px = self.geometry.scaled(w, h)
        if inner_px.shape[0] < 3:
            return
        visible_mm = self.calibration.camera_px_to_table_mm(inner_px)
        self._last_table_edge_polygon_mm = _points_to_tuples(visible_mm)
        pocket_curves_mm = [
            self.calibration.camera_px_to_table_mm(np.asarray(pocket, dtype=np.float32))
            for pocket in pockets_px
            if np.asarray(pocket, dtype=np.float32).reshape((-1, 2)).shape[0] >= 2
        ]
        self._last_pocket_curves_mm = [_points_to_tuples(curve) for curve in pocket_curves_mm]
        boundaries = derive_table_boundaries(
            visible_mm,
            pocket_curves_mm,
            table_width_mm=float(self.calibration.table.width_mm),
            table_height_mm=float(self.calibration.table.height_mm),
            ball_diameter_mm=float(self.calibration.table.ball_diameter_mm),
            projection_visible_insets=EdgeInsets(
                top_mm=float(self.config.calibration.projection_visible_inset_top_mm),
                right_mm=float(self.config.calibration.projection_visible_inset_right_mm),
                bottom_mm=float(self.config.calibration.projection_visible_inset_bottom_mm),
                left_mm=float(self.config.calibration.projection_visible_inset_left_mm),
            ),
            physical_rail_insets=EdgeInsets(
                top_mm=float(self.config.calibration.physical_rail_inset_top_mm),
                right_mm=float(self.config.calibration.physical_rail_inset_right_mm),
                bottom_mm=float(self.config.calibration.physical_rail_inset_bottom_mm),
                left_mm=float(self.config.calibration.physical_rail_inset_left_mm),
            ),
            physical_middle_pocket_relief_top_mm=float(self.config.calibration.physical_middle_pocket_relief_top_mm),
            physical_middle_pocket_relief_bottom_mm=float(self.config.calibration.physical_middle_pocket_relief_bottom_mm),
            center_reachable_extra_margin_mm=float(self.config.calibration.center_reachable_extra_margin_mm),
        )
        self.calibration.table.projection_visible_polygon_mm = _points_to_tuples(boundaries.projection_visible_polygon_mm)
        self.calibration.table.inner_polygon_mm = _points_to_tuples(boundaries.physical_rail_polygon_mm)
        self.calibration.table.center_playable_polygon_mm = _points_to_tuples(boundaries.center_playable_polygon_mm)
        if boundaries.projection_visible_pocket_points_mm:
            self.calibration.table.projection_visible_pockets_mm = list(boundaries.projection_visible_pocket_points_mm)
        if boundaries.physical_pocket_points_mm:
            self.calibration.table.pockets_mm = list(boundaries.physical_pocket_points_mm)

    def _enrich_tracks_with_table_units(self, tracks: TracksFrame) -> TracksFrame:
        enriched = []
        dt = 0.1
        for track in tracks.tracks:
            try:
                located = self.calibration.ball_geometry.locate(
                    track.center_px,
                    radius_px=track.radius_px,
                    geometry_quality=float(getattr(track, "geometry_quality", track.quality)),
                    geometry_method=str(getattr(track, "geometry_method", "unknown")),
                )
                center_mm = np.asarray(located.table_center_mm, dtype=np.float32)
                v_px = np.asarray(track.velocity_px_s, dtype=np.float32)
                edge_px = np.asarray([[track.center_px[0] + float(v_px[0]) * dt, track.center_px[1] + float(v_px[1]) * dt]], dtype=np.float32)
                edge_mm = self.calibration.ball_camera_px_to_table_mm(edge_px)[0]
                velocity_mm = (edge_mm - center_mm) / dt
                radius_mm = located.radius_mm
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
            if self.learning_recorder is not None and out.training is None:
                self.learning_recorder.observe(out.state, out.plan)
            return
        self.recorder.write_frame_packet(out.frame)
        self.recorder.write_detections(out.detections)
        self.recorder.write_tracks(out.tracks)
        self.recorder.write_state(out.state)
        self.recorder.write_plan(out.plan)
        self.recorder.write_overlay(out.overlay)
        if self.learning_recorder is not None and out.training is None:
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


def _points_to_tuples(points: np.ndarray) -> list[tuple[float, float]]:
    arr = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    return [(float(x), float(y)) for x, y in arr]
