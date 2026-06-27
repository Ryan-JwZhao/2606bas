from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bas.app import RuntimePipeline
from bas.operator_controls import RuntimeControlState
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
