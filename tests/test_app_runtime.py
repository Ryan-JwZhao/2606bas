from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bas.app import RuntimePipeline
from bas.operator_controls import RuntimeControlState
from bas.config import AppConfig, CalibrationConfig
from bas.schemas import DetectionsFrame, FramePacket, MatchStateFrame, ProjectionOverlay, ShotPlan, TracksFrame


class _Capture:
    def __init__(self, frames):
        self._frames = list(frames)

    def read(self):
        if not self._frames:
            return None
        return self._frames.pop(0)


class _Detector:
    def __init__(self):
        self._calls = 0

    def process(self, frame, mask_polygon=None, detection_regions=None):
        self._calls += 1
        version = "fake_detector" if self._calls == 1 else "fake_detector:cached"
        return DetectionsFrame(frame_id=frame.frame_id, ts_cam_ns=frame.ts_cam_ns, detections=[], detector_version=version)


class _StateMachine:
    def __init__(self):
        self.calls = 0

    def set_table_context(self, **kwargs):
        return None

    def update(self, tracks):
        self.calls += 1
        return MatchStateFrame(frame_id=tracks.frame_id, ts_cam_ns=tracks.ts_cam_ns, phase=f"CALL_{self.calls}")


class _Planner:
    def plan(self, state, frame_bgr=None, forced_shot_mode=None, forced_turn_target_group=None):
        return ShotPlan(plan_id=f"plan_{state.frame_id}", frame_id=state.frame_id, ts_cam_ns=state.ts_cam_ns)


class _OverlayBuilder:
    def from_plan(self, plan):
        return ProjectionOverlay(overlay_id=f"overlay_{plan.frame_id}", frame_id=plan.frame_id, projector_size=(1280, 800))


def test_runtime_pipeline_updates_state_on_cached_detection_frames() -> None:
    pipeline = RuntimePipeline.__new__(RuntimePipeline)
    pipeline.config = SimpleNamespace(calibration=SimpleNamespace(), planner=SimpleNamespace(shot_mode="rule"))
    pipeline.capture = _Capture(
        [
            FramePacket(frame_id=1, ts_cam_ns=1, camera_id="cam", image=np.zeros((4, 4, 3), dtype=np.uint8)),
            FramePacket(frame_id=2, ts_cam_ns=2, camera_id="cam", image=np.zeros((4, 4, 3), dtype=np.uint8)),
        ]
    )
    pipeline.calibration = SimpleNamespace(
        calib_version="test",
        table=SimpleNamespace(inner_polygon_mm=[], pockets_mm=[], ball_diameter_mm=57.15),
    )
    pipeline._geometry_version = "geometry-test"
    pipeline.geometry = SimpleNamespace(is_empty=True)
    pipeline.detector = _Detector()
    pipeline.tracker = SimpleNamespace(
        update=lambda detections: TracksFrame(frame_id=detections.frame_id, ts_cam_ns=detections.ts_cam_ns, tracks=[])
    )
    pipeline.state_machine = _StateMachine()
    pipeline.planner = _Planner()
    pipeline.overlay_builder = _OverlayBuilder()
    pipeline.control_state = RuntimeControlState()
    pipeline.recorder = None
    pipeline.learning_recorder = None
    pipeline._last_tracks = None
    pipeline._last_state = None
    pipeline._last_plan = None
    pipeline._last_overlay = None
    pipeline.last_timings_ms = {}
    pipeline._processed_frames = 0
    pipeline._cached_detection_frames = 0
    pipeline._refresh_geometry_if_needed = lambda: None
    pipeline._update_table_geometry_for_frame = lambda frame: None
    pipeline._camera_table_mask = lambda frame: None
    pipeline._enrich_tracks_with_table_units = lambda tracks: tracks
    pipeline._record = lambda out: None

    first = pipeline.step()
    second = pipeline.step()

    assert first is not None
    assert second is not None
    assert pipeline.state_machine.calls == 2
    assert second.state.phase == "CALL_2"


def test_uncalibrated_runtime_keeps_camera_acquisition_available() -> None:
    pipeline = RuntimePipeline.__new__(RuntimePipeline)
    pipeline.config = SimpleNamespace(calibration=CalibrationConfig())
    pipeline.geometry = SimpleNamespace(
        is_empty=False,
        scaled=lambda _width, _height: (
            np.asarray([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.float32),
            np.asarray([[5, 5], [94, 5], [94, 94], [5, 94]], dtype=np.float32),
            [],
        ),
    )
    pipeline.calibration = SimpleNamespace(
        projection=SimpleNamespace(is_valid=False),
        table=SimpleNamespace(
            width_mm=2540.0,
            height_mm=1270.0,
            ball_diameter_mm=57.15,
            projection_visible_polygon_mm=[],
            inner_polygon_mm=[],
            center_playable_polygon_mm=[],
            projection_visible_pockets_mm=[],
            pockets_mm=[],
        ),
        camera_px_to_table_mm=lambda _points: (_ for _ in ()).throw(
            RuntimeError("Projection calibration is unavailable or invalid; geometry transforms are disabled.")
        ),
    )
    pipeline._last_table_edge_polygon_mm = []
    pipeline._last_pocket_curves_mm = []
    frame = FramePacket(1, 1, "camera", image=np.zeros((100, 100, 3), dtype=np.uint8))

    pipeline._update_table_geometry_for_frame(frame)

    assert pipeline._last_table_edge_polygon_mm == []


def test_clearing_runtime_geometry_clears_all_derived_table_context() -> None:
    pipeline = RuntimePipeline.__new__(RuntimePipeline)
    from bas.geometry import TableGeometry

    nonempty_geometry = TableGeometry(
        inner_norm=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )
    pipeline.geometry = nonempty_geometry
    pipeline.geometry_reloader = SimpleNamespace(
        refresh=lambda *_paths: (TableGeometry(), True)
    )
    pipeline.config = SimpleNamespace(
        geometry=SimpleNamespace(outline_path=None, inline_path=None, pocket_path=None)
    )
    pipeline.calibration = SimpleNamespace(
        table=SimpleNamespace(
            projection_visible_polygon_mm=[(0.0, 0.0)],
            inner_polygon_mm=[(0.0, 0.0)],
            center_playable_polygon_mm=[(0.0, 0.0)],
            projection_visible_pockets_mm=[(0.0, 0.0)],
            pockets_mm=[(0.0, 0.0)],
        )
    )
    pipeline._last_table_edge_polygon_mm = [(0.0, 0.0)]
    pipeline._last_pocket_curves_mm = [[(0.0, 0.0)]]
    reset_calls: list[bool] = []
    pipeline._reset_temporal_processing_state = lambda: reset_calls.append(True)

    pipeline._refresh_geometry_if_needed()

    assert pipeline._last_table_edge_polygon_mm == []
    assert pipeline._last_pocket_curves_mm == []
    assert pipeline.calibration.table.projection_visible_polygon_mm == []
    assert pipeline.calibration.table.inner_polygon_mm == []
    assert pipeline.calibration.table.center_playable_polygon_mm == []
    assert pipeline.calibration.table.projection_visible_pockets_mm == []
    assert pipeline.calibration.table.pockets_mm == []
    assert reset_calls == [True]


def test_uncalibrated_pipeline_returns_camera_frame_with_dense_geometry() -> None:
    config = AppConfig()
    config.camera.backend = "synthetic"
    config.camera.width = 320
    config.camera.height = 180
    config.camera.distortion_correction_enabled = False
    config.detector.backend = "disabled"
    config.calibration.camera_file = None
    config.calibration.projection_file = None
    pipeline = RuntimePipeline(config)
    pipeline.geometry = SimpleNamespace(
        is_empty=False,
        scaled=lambda width, height: (
            np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32),
            np.asarray([[5, 5], [width - 6, 5], [width - 6, height - 6], [5, height - 6]], dtype=np.float32),
            [],
        ),
    )
    try:
        output = pipeline.step()
    finally:
        pipeline.close()

    assert output is not None
    assert output.frame.image is not None
    assert output.frame.image.shape == (180, 320, 3)
    assert output.state.phase == "CALIBRATION_REQUIRED"
    assert pipeline.calibration.projection.is_valid is False


def test_uncalibrated_training_pipeline_returns_calibration_required_preview() -> None:
    config = AppConfig()
    config.camera.backend = "synthetic"
    config.camera.width = 320
    config.camera.height = 180
    config.camera.distortion_correction_enabled = False
    config.detector.backend = "disabled"
    config.training_detector.backend = "disabled"
    config.training.operating_mode = "training"
    config.training.scenario_id = "ordered_line_1_7"
    config.calibration.camera_file = None
    config.calibration.projection_file = None
    pipeline = RuntimePipeline(config)
    pipeline.planner.plan = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("uncalibrated training must not call the geometry planner")
    )
    try:
        output = pipeline.step()
    finally:
        pipeline.close()

    assert output is not None
    assert output.frame.image is not None
    assert output.state.phase == "CALIBRATION_REQUIRED"
    assert output.training is None
