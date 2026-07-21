from __future__ import annotations

import numpy as np

from bas.app import RuntimePipeline
from bas.config import AppConfig, DetectorConfig, TrackerConfig, TrainingConfig
from bas.perception.detector import Detector
from bas.perception.mode_service import ModeAwareDetectService
from bas.perception.regions import DetectionRegionPolicy, PocketGuardRegion
from bas.schemas import Detection, FramePacket, PocketVisualObservationFrame, TrackObservation, TracksFrame
from bas.training import NumberedBallTracker, TrainingSession, get_training_scenario, list_training_scenarios


class _TaggedDetector(Detector):
    def __init__(self, tag: str):
        self.version = tag

    def detect(self, frame_bgr, mask_polygon=None):
        return [Detection(bbox=(1, 1, 5, 5), conf=0.9, cls_id=0, cls_name=self.version)]


def _track(
    number: int,
    x: float,
    y: float,
    *,
    visible: bool = True,
    vx: float = 0.0,
    vy: float = 0.0,
) -> TrackObservation:
    return TrackObservation(
        track_id=number,
        bbox=(x - 10, y - 10, x + 10, y + 10),
        center_px=(x, y),
        center_mm=(x, y),
        radius_px=10.0,
        radius_mm=28.575,
        velocity_px_s=(vx, vy),
        velocity_mm_s=(vx, vy),
        cls_name=str(number),
        group=(
            "cue"
            if number == 0
            else "solid"
            if number <= 7
            else "black"
            if number == 8
            else "stripe"
        ),
        confidence=0.95,
        visibility="visible" if visible else "occluded",
        lost_frames=0 if visible else 1,
    )


def _tracks(frame_id: int, numbers: list[int]) -> TracksFrame:
    observations = [_track(0, 200.0, 500.0)]
    observations.extend(_track(number, 300.0 + number * 80.0, 300.0) for number in numbers)
    return TracksFrame(frame_id=frame_id, ts_cam_ns=frame_id * 100_000_000, tracks=observations)


def test_training_catalog_contains_multiple_number_aware_drills() -> None:
    scenarios = list_training_scenarios()
    assert len(scenarios) >= 6
    assert get_training_scenario("ordered_line_1_7").ordered_numbers == tuple(range(1, 8))
    assert get_training_scenario("solids_then_black").stages[-1] == (8,)
    assert get_training_scenario("stripes_then_black").stages[-1] == (8,)


def test_ordered_line_session_passes_correct_ball_and_rejects_wrong_ball() -> None:
    session = TrainingSession(
        TrainingConfig(scenario_id="ordered_line_1_7", disappearance_confirm_frames=2),
        ball_diameter_mm=57.15,
    )
    ready = session.update(_tracks(1, list(range(1, 8))))
    assert ready.phase == "ready"
    ok, running = session.start()
    assert ok is True
    assert running.expected_numbers == [1]

    session.update(_tracks(2, list(range(2, 8))))
    accepted = session.update(_tracks(3, list(range(2, 8))))
    assert accepted.phase == "running"
    assert accepted.potted_numbers == [1]
    assert accepted.expected_numbers == [2]

    session.update(_tracks(4, [2, 4, 5, 6, 7]))
    failed = session.update(_tracks(5, [2, 4, 5, 6, 7]))
    assert failed.phase == "failed"
    assert failed.failure_reason == "wrong_ball"


def test_training_session_rejects_cue_ball_scratch() -> None:
    session = TrainingSession(TrainingConfig(scenario_id="finish_6_7_8", disappearance_confirm_frames=1))
    session.update(_tracks(1, [6, 7, 8]))
    assert session.start()[0] is True
    scratched = session.update(
        TracksFrame(
            frame_id=2,
            ts_cam_ns=200_000_000,
            tracks=[_track(6, 780, 300), _track(7, 860, 300), _track(8, 940, 300)],
        )
    )
    assert scratched.phase == "failed"
    assert scratched.failure_reason == "cue_ball_pocketed"


def test_training_session_rejects_disappearance_away_from_pocket() -> None:
    session = TrainingSession(TrainingConfig(scenario_id="finish_6_7_8", disappearance_confirm_frames=1))
    session.set_table_context(pockets_mm=[(0.0, 0.0)])
    session.update(_tracks(1, [6, 7, 8]))
    assert session.start()[0] is True
    lost_in_middle = session.update(_tracks(2, [7, 8]))
    assert lost_in_middle.phase == "failed"
    assert lost_in_middle.failure_reason == "ball_lost_away_from_pocket"


def test_training_session_accepts_rule_judged_projected_entry() -> None:
    session = TrainingSession(
        TrainingConfig(
            scenario_id="solids_then_black",
            disappearance_confirm_frames=1,
            pocket_proximity_mm=50.0,
        )
    )
    session.set_table_context(
        inner_polygon_mm=[(0.0, 0.0), (1_000.0, 0.0), (1_000.0, 500.0), (0.0, 500.0)],
        table_edge_polygon_mm=[(0.0, 0.0), (1_000.0, 0.0), (1_000.0, 500.0), (0.0, 500.0)],
        ball_center_reachable_polygon_mm=[
            (28.0, 28.0),
            (972.0, 28.0),
            (972.0, 472.0),
            (28.0, 472.0),
        ],
        pockets_mm=[(500.0, 0.0)],
        ball_diameter_mm=56.0,
    )
    setup_tracks = [_track(0, 200.0, 400.0)] + [
        _track(number, 250.0 + number * 60.0, 300.0)
        for number in range(1, 9)
    ]
    session.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=setup_tracks))
    assert session.start()[0] is True

    stationary = [
        _track(number, 250.0 + number * 60.0, 300.0)
        for number in range(2, 9)
    ]
    session.update(
        TracksFrame(
            frame_id=2,
            ts_cam_ns=1_100_000_000,
            tracks=[_track(0, 200.0, 400.0), _track(1, 515.0, 180.0, vy=-700.0), *stationary],
        )
    )
    session.update(
        TracksFrame(
            frame_id=3,
            ts_cam_ns=1_200_000_000,
            tracks=[_track(0, 200.0, 400.0), _track(1, 514.0, 72.0, vy=-1_080.0), *stationary],
        )
    )
    missing = [_track(0, 200.0, 400.0), *stationary]
    session.update(TracksFrame(frame_id=4, ts_cam_ns=1_300_000_000, tracks=missing))
    session.update(TracksFrame(frame_id=5, ts_cam_ns=1_600_000_000, tracks=missing))
    resolved = session.update(TracksFrame(frame_id=6, ts_cam_ns=2_100_000_000, tracks=missing))

    assert resolved.phase == "running"
    assert resolved.failure_reason is None
    assert resolved.potted_numbers == [1]
    potted = next(event for event in resolved.events if event.name == "TRAINING_BALL_POTTED")
    assert potted.payload["judgment"] == "rules_pocket_fsm"
    assert potted.payload["decision_id"] == "pocket:1"


def test_numbered_tracker_keeps_identity_by_ball_number() -> None:
    tracker = NumberedBallTracker(TrackerConfig(high_conf=0.4, low_conf=0.1, max_lost_frames=3))
    frame = FramePacket(frame_id=1, ts_cam_ns=100_000_000, camera_id="test", image=np.zeros((20, 20, 3), dtype=np.uint8))
    detections = [
        Detection(bbox=(1, 1, 5, 5), conf=0.8, cls_id=3, cls_name="3"),
        Detection(bbox=(2, 2, 6, 6), conf=0.7, cls_id=3, cls_name="3"),
        Detection(bbox=(10, 10, 15, 15), conf=0.9, cls_id=8, cls_name="8"),
    ]
    from bas.schemas import DetectionsFrame

    output = tracker.update(DetectionsFrame(frame_id=frame.frame_id, ts_cam_ns=frame.ts_cam_ns, detections=detections))
    assert [(track.track_id, track.cls_name, track.group) for track in output.tracks] == [
        (3, "3", "solid"),
        (8, "8", "black"),
    ]


def test_mode_aware_detector_lazily_builds_each_model_once() -> None:
    created: list[str] = []

    def factory(config: DetectorConfig) -> Detector:
        created.append(str(config.model_path))
        return _TaggedDetector(str(config.model_path))

    service = ModeAwareDetectService(
        DetectorConfig(model_path="rule-model"),
        DetectorConfig(model_path="training-model"),
        detector_factory=factory,
    )
    assert created == ["rule-model"]
    service.activate("training")
    service.activate("rules")
    service.activate("training")
    assert created == ["rule-model", "training-model"]


def test_runtime_pipeline_uses_training_branch_without_rule_planning() -> None:
    config = AppConfig()
    config.camera.backend = "synthetic"
    config.camera.width = 320
    config.camera.height = 180
    config.detector.backend = "disabled"
    config.training_detector.backend = "disabled"
    config.training.operating_mode = "training"
    config.replay.enabled = False
    pipeline = RuntimePipeline(config)
    try:
        observed_modes: list[str] = []
        guard = PocketGuardRegion(
            pocket_index=0,
            polygon=np.asarray([(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)], dtype=np.float32),
            center_px=(10.0, 10.0),
            ball_diameter_px=10.0,
        )
        pipeline._camera_detection_regions = lambda frame: DetectionRegionPolicy(ball_guard_regions=(guard,))
        pipeline.pocket_observer.update = lambda frame, tracks, regions: (
            observed_modes.append(pipeline.operating_mode)
            or PocketVisualObservationFrame(frame_id=frame.frame_id, ts_cam_ns=frame.ts_cam_ns)
        )

        output = pipeline.step()
        assert output is not None
        assert output.training is not None
        assert output.plan.shot_mode == "training"
        assert output.state.phase.startswith("TRAINING_")
        assert pipeline.detector.mode == "training"
        assert observed_modes == ["training"]

        assert pipeline.set_operating_mode("rules") == "rules"
        rule_output = pipeline.step()
        assert rule_output is not None
        assert rule_output.training is None
        assert rule_output.plan.shot_mode in {"rule", "hook", "target"}
    finally:
        pipeline.close()
