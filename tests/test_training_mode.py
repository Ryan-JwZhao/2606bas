from __future__ import annotations

import numpy as np

from bas.app import RuntimePipeline
from bas.config import AppConfig, DetectorConfig, TrackerConfig, TrainingConfig
from bas.perception.detector import Detector
from bas.perception.mode_service import ModeAwareDetectService
from bas.perception.regions import DetectionRegionPolicy, PocketGuardRegion
from bas.schemas import (
    Detection,
    FramePacket,
    PocketVisualObservation,
    PocketVisualObservationFrame,
    OverlayLine,
    ProjectionOverlay,
    TrackObservation,
    TracksFrame,
)
from bas.training import (
    NumberedBallTracker,
    TrainingSession,
    TrainingStateFrame,
    get_training_scenario,
    list_training_scenarios,
)


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
    lost_frames: int | None = None,
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
        lost_frames=(0 if visible else 1) if lost_frames is None else int(lost_frames),
    )


def _tracks(frame_id: int, numbers: list[int]) -> TracksFrame:
    observations = [_track(0, 200.0, 500.0)]
    observations.extend(_track(number, 300.0 + number * 80.0, 300.0) for number in numbers)
    return TracksFrame(frame_id=frame_id, ts_cam_ns=frame_id * 100_000_000, tracks=observations)


def _confirm_visual_pot(
    session: TrainingSession,
    *,
    ball: int,
    frame_id: int,
    ts_cam_ns: int,
    remaining_numbers: list[int],
) -> TrainingStateFrame:
    remaining_tracks = [
        track
        for track in _tracks(frame_id, remaining_numbers).tracks
        if int(track.track_id) != int(ball)
    ]
    group = _track(ball, 0.0, 0.0).group
    session.update(
        TracksFrame(frame_id=frame_id, ts_cam_ns=ts_cam_ns, tracks=remaining_tracks),
        PocketVisualObservationFrame(
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            observations=[
                PocketVisualObservation(
                    pocket_index=0,
                    inward_crossing=True,
                    group=group,
                    confidence=0.98,
                    associated_track_ids=[ball],
                    evidence_sources=["foreground_motion", "ball_sized_motion"],
                    motion_score=0.9,
                    foreground_score=0.8,
                    foreground_depth_diameters=0.5,
                )
            ],
        ),
    )
    resolved_ts_ns = ts_cam_ns + 1_300_000_000
    return session.update(
        TracksFrame(frame_id=frame_id + 1, ts_cam_ns=resolved_ts_ns, tracks=remaining_tracks),
        PocketVisualObservationFrame(
            frame_id=frame_id + 1,
            ts_cam_ns=resolved_ts_ns,
            observations=[PocketVisualObservation(pocket_index=0, clear=True)],
        ),
    )


def test_training_catalog_contains_multiple_number_aware_drills() -> None:
    scenarios = list_training_scenarios()
    assert len(scenarios) >= 6
    assert get_training_scenario("ordered_line_1_7").ordered_numbers == tuple(range(1, 8))
    assert get_training_scenario("solids_then_black").stages[-1] == (8,)
    assert get_training_scenario("stripes_then_black").stages[-1] == (8,)


def test_ordered_line_session_passes_correct_ball_and_rejects_wrong_ball() -> None:
    session = TrainingSession(
        TrainingConfig(scenario_id="ordered_line_1_7"),
        ball_diameter_mm=57.15,
    )
    session.set_table_context(pockets_mm=[(0.0, 0.0)])
    ready = session.update(_tracks(1, list(range(1, 8))))
    assert ready.phase == "ready"
    ok, running = session.start()
    assert ok is True
    assert running.expected_numbers == [1]

    accepted = _confirm_visual_pot(
        session,
        ball=1,
        frame_id=2,
        ts_cam_ns=200_000_000,
        remaining_numbers=list(range(2, 8)),
    )
    assert accepted.phase == "running"
    assert accepted.potted_numbers == [1]
    assert accepted.expected_numbers == [2]

    failed = _confirm_visual_pot(
        session,
        ball=3,
        frame_id=4,
        ts_cam_ns=1_600_000_000,
        remaining_numbers=[2, 4, 5, 6, 7],
    )
    assert failed.phase == "failed"
    assert failed.failure_reason == "wrong_ball"


def test_training_session_rejects_cue_ball_scratch() -> None:
    session = TrainingSession(TrainingConfig(scenario_id="finish_6_7_8"))
    session.set_table_context(pockets_mm=[(0.0, 0.0)])
    session.update(_tracks(1, [6, 7, 8]))
    assert session.start()[0] is True
    scratched = _confirm_visual_pot(
        session,
        ball=0,
        frame_id=2,
        ts_cam_ns=200_000_000,
        remaining_numbers=[6, 7, 8],
    )
    assert scratched.phase == "failed"
    assert scratched.failure_reason == "cue_ball_pocketed"


def test_training_requires_pocket_calibration_before_start() -> None:
    session = TrainingSession(TrainingConfig(scenario_id="finish_6_7_8"))

    setup = session.update(_tracks(1, [6, 7, 8]))
    started, state = session.start()

    assert setup.phase == "setup"
    assert setup.setup_ready is False
    assert "袋口标定" in setup.message
    assert started is False
    assert state.phase == "setup"


def test_training_rejects_early_black_eight_with_shared_pocket_judgment() -> None:
    session = TrainingSession(
        TrainingConfig(scenario_id="solids_then_black")
    )
    session.set_table_context(pockets_mm=[(0.0, 0.0)])
    session.update(_tracks(1, list(range(1, 9))))
    assert session.start()[0] is True

    failed = _confirm_visual_pot(
        session,
        ball=8,
        frame_id=2,
        ts_cam_ns=200_000_000,
        remaining_numbers=list(range(1, 8)),
    )

    assert failed.phase == "failed"
    assert failed.failure_reason == "wrong_ball"


def test_training_ignores_a_confirmed_potted_ball_reappearing() -> None:
    session = TrainingSession(
        TrainingConfig(scenario_id="solids_then_black")
    )
    session.set_table_context(pockets_mm=[(0.0, 0.0)])
    session.update(_tracks(1, list(range(1, 9))))
    assert session.start()[0] is True
    confirmed = _confirm_visual_pot(
        session,
        ball=1,
        frame_id=2,
        ts_cam_ns=200_000_000,
        remaining_numbers=list(range(2, 9)),
    )
    assert confirmed.potted_numbers == [1]

    continued = session.update(_tracks(4, list(range(1, 9))))

    assert continued.phase == "running"
    assert continued.failure_reason is None
    assert continued.potted_numbers == [1]


def test_training_restart_discards_visual_candidates_from_previous_attempt() -> None:
    session = TrainingSession(
        TrainingConfig(scenario_id="finish_6_7_8")
    )
    session.set_table_context(pockets_mm=[(0.0, 0.0)])
    session.update(_tracks(1, [6, 7, 8]))
    assert session.start()[0] is True
    all_balls = _tracks(2, [6, 7, 8])
    candidate = session.update(
        all_balls,
        PocketVisualObservationFrame(
            frame_id=2,
            ts_cam_ns=all_balls.ts_cam_ns,
            observations=[
                PocketVisualObservation(
                    pocket_index=0,
                    inward_crossing=True,
                    group="solid",
                    confidence=0.98,
                    associated_track_ids=[6],
                    evidence_sources=["foreground_motion", "ball_sized_motion"],
                    motion_score=0.9,
                    foreground_score=0.8,
                    foreground_depth_diameters=0.5,
                )
            ],
        ),
    )
    assert candidate.phase == "running"
    assert candidate.potted_numbers == []

    started, restarted = session.start()
    assert started is True
    assert restarted.potted_numbers == []

    resolved = session.update(
        TracksFrame(
            frame_id=3,
            ts_cam_ns=1_500_000_000,
            tracks=_tracks(3, [7, 8]).tracks,
        ),
        PocketVisualObservationFrame(
            frame_id=3,
            ts_cam_ns=1_500_000_000,
            observations=[PocketVisualObservation(pocket_index=0, clear=True)],
        ),
    )

    assert resolved.phase == "running"
    assert resolved.failure_reason is None
    assert resolved.potted_numbers == []


def test_training_pocket_observer_ignores_numbers_outside_the_scenario() -> None:
    session = TrainingSession(TrainingConfig(scenario_id="solids_then_black"))
    tracks = _tracks(1, list(range(1, 10)))

    filtered = session.pocket_observer_tracks(tracks)

    assert [int(track.track_id) for track in filtered.tracks] == list(range(0, 9))


def test_training_session_holds_when_cue_ball_remains_on_pocket_lip() -> None:
    session = TrainingSession(
        TrainingConfig(scenario_id="finish_6_7_8")
    )
    session.set_table_context(pockets_mm=[(500.0, 0.0)])
    object_balls = [_track(6, 780.0, 300.0), _track(7, 860.0, 300.0), _track(8, 940.0, 300.0)]
    cue_on_lip = _track(0, 500.0, 70.0)
    session.update(
        TracksFrame(
            frame_id=1,
            ts_cam_ns=1_000_000_000,
            tracks=[cue_on_lip, *object_balls],
        )
    )
    assert session.start()[0] is True

    state = session.update(
        TracksFrame(
            frame_id=2,
            ts_cam_ns=1_100_000_000,
            tracks=[cue_on_lip, *object_balls],
        ),
        PocketVisualObservationFrame(
            frame_id=2,
            ts_cam_ns=1_100_000_000,
            observations=[
                PocketVisualObservation(
                    pocket_index=0,
                    inward_crossing=True,
                    lip_occupied=True,
                    group="cue",
                    confidence=0.98,
                    associated_track_ids=[0],
                    evidence_sources=["foreground_motion", "ball_sized_motion"],
                    motion_score=0.9,
                    foreground_score=0.8,
                    foreground_depth_diameters=0.2,
                )
            ],
        ),
    )

    for offset in range(1, 13):
        ts_ns = 1_100_000_000 + offset * 100_000_000
        state = session.update(
            TracksFrame(
                frame_id=2 + offset,
                ts_cam_ns=ts_ns,
                tracks=[
                    _track(0, 500.0, 70.0, visible=False, lost_frames=offset),
                    *object_balls,
                ],
            ),
            PocketVisualObservationFrame(
                frame_id=2 + offset,
                ts_cam_ns=ts_ns,
                observations=[
                    PocketVisualObservation(
                        pocket_index=0,
                        lip_occupied=True,
                        group="cue",
                        associated_track_ids=[0],
                    )
                ],
            ),
        )

    for offset in range(13, 21):
        ts_ns = 1_100_000_000 + offset * 100_000_000
        state = session.update(
            TracksFrame(frame_id=2 + offset, ts_cam_ns=ts_ns, tracks=object_balls),
            PocketVisualObservationFrame(
                frame_id=2 + offset,
                ts_cam_ns=ts_ns,
                observations=[
                    PocketVisualObservation(
                        pocket_index=0,
                        lip_occupied=True,
                        group="cue",
                        associated_track_ids=[0],
                    )
                ],
            ),
        )

    assert state.phase == "running"
    assert state.failure_reason is None
    assert state.potted_numbers == []

    for offset in range(21, 29):
        ts_ns = 1_100_000_000 + offset * 100_000_000
        state = session.update(
            TracksFrame(frame_id=2 + offset, ts_cam_ns=ts_ns, tracks=object_balls),
            PocketVisualObservationFrame(
                frame_id=2 + offset,
                ts_cam_ns=ts_ns,
                observations=[PocketVisualObservation(pocket_index=0, clear=True)],
            ),
        )

    assert state.phase == "running"
    assert state.failure_reason is None


def test_training_holds_numbered_ball_when_lip_loses_track_association() -> None:
    session = TrainingSession(
        TrainingConfig(scenario_id="finish_6_7_8")
    )
    session.set_table_context(pockets_mm=[(500.0, 0.0)])
    cue = _track(0, 200.0, 500.0)
    ball_on_lip = _track(6, 500.0, 70.0)
    other_balls = [_track(7, 860.0, 300.0), _track(8, 940.0, 300.0)]
    session.update(
        TracksFrame(
            frame_id=1,
            ts_cam_ns=1_000_000_000,
            tracks=[cue, ball_on_lip, *other_balls],
        )
    )
    assert session.start()[0] is True
    session.update(
        TracksFrame(
            frame_id=2,
            ts_cam_ns=1_100_000_000,
            tracks=[cue, ball_on_lip, *other_balls],
        ),
        PocketVisualObservationFrame(
            frame_id=2,
            ts_cam_ns=1_100_000_000,
            observations=[
                PocketVisualObservation(
                    pocket_index=0,
                    lip_occupied=True,
                    group="solid",
                    associated_track_ids=[6],
                )
            ],
        ),
    )

    for offset in range(1, 7):
        ts_ns = 1_100_000_000 + offset * 100_000_000
        state = session.update(
            TracksFrame(
                frame_id=2 + offset,
                ts_cam_ns=ts_ns,
                tracks=[cue, *other_balls],
            ),
            PocketVisualObservationFrame(
                frame_id=2 + offset,
                ts_cam_ns=ts_ns,
                observations=[
                    PocketVisualObservation(
                        pocket_index=0,
                        lip_occupied=True,
                        group="solid",
                    )
                ],
            ),
        )

    assert state.phase == "running"
    assert state.failure_reason is None
    assert state.potted_numbers == []

    for offset in range(7, 11):
        ts_ns = 1_100_000_000 + offset * 100_000_000
        state = session.update(
            TracksFrame(
                frame_id=2 + offset,
                ts_cam_ns=ts_ns,
                tracks=[cue, *other_balls],
            ),
            PocketVisualObservationFrame(
                frame_id=2 + offset,
                ts_cam_ns=ts_ns,
                observations=[PocketVisualObservation(pocket_index=0, clear=True)],
            ),
        )

    assert state.phase == "running"
    assert state.failure_reason is None


def test_training_session_ignores_disappearance_away_from_pocket() -> None:
    session = TrainingSession(TrainingConfig(scenario_id="finish_6_7_8"))
    session.set_table_context(pockets_mm=[(0.0, 0.0)])
    session.update(_tracks(1, [6, 7, 8]))
    assert session.start()[0] is True
    lost_in_middle = session.update(_tracks(2, [7, 8]))
    assert lost_in_middle.phase == "running"
    assert lost_in_middle.failure_reason is None


def test_training_session_ignores_missing_ball_after_tracker_eviction() -> None:
    session = TrainingSession(TrainingConfig(scenario_id="solids_then_black"))
    session.set_table_context(pockets_mm=[(0.0, 0.0)])
    setup_tracks = [_track(0, 200.0, 500.0)] + [
        _track(number, 300.0 + number * 80.0, 300.0)
        for number in range(1, 9)
    ]
    session.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=setup_tracks))
    assert session.start()[0] is True

    state = session.state
    for offset in range(1, 9):
        lost_frames = 1 if offset <= 4 else 2
        tracks = [_track(0, 200.0, 500.0)] + [
            _track(
                number,
                300.0 + number * 80.0,
                300.0,
                visible=number != 1,
                lost_frames=lost_frames if number == 1 else 0,
            )
            for number in range(1, 9)
        ]
        state = session.update(
            TracksFrame(
                frame_id=1 + offset,
                ts_cam_ns=1_000_000_000 + offset * 16_700_000,
                tracks=tracks,
            )
        )

    assert state.phase == "running"
    assert state.failure_reason is None

    without_ball_one = [_track(0, 200.0, 500.0)] + [
        _track(number, 300.0 + number * 80.0, 300.0)
        for number in range(2, 9)
    ]
    for offset in range(9, 17):
        state = session.update(
            TracksFrame(
                frame_id=1 + offset,
                ts_cam_ns=1_000_000_000 + offset * 16_700_000,
                tracks=without_ball_one,
            )
        )

    assert state.phase == "running"
    assert state.failure_reason is None


def test_training_session_waits_for_late_visual_crossing_and_confirms_occluded_ball() -> None:
    session = TrainingSession(
        TrainingConfig(scenario_id="solids_then_black")
    )
    session.set_table_context(pockets_mm=[(0.0, 0.0)])
    setup_tracks = [_track(0, 200.0, 500.0)] + [
        _track(number, 300.0 + number * 80.0, 300.0)
        for number in range(1, 9)
    ]
    session.update(TracksFrame(frame_id=1, ts_cam_ns=1_000_000_000, tracks=setup_tracks))
    assert session.start()[0] is True

    state = session.state
    for offset in range(1, 101):
        tracker_lost_frames = min(18, (offset + 3) // 4)
        include_ball_one = offset <= 72
        tracks = [_track(0, 200.0, 500.0)] + [
            _track(number, 300.0 + number * 80.0, 300.0)
            for number in range(2, 9)
        ]
        if include_ball_one:
            tracks.append(
                _track(
                    1,
                    380.0,
                    300.0,
                    visible=False,
                    lost_frames=tracker_lost_frames,
                )
            )
        observations = []
        if offset == 12:
            observations.append(
                PocketVisualObservation(
                    pocket_index=0,
                    inward_crossing=True,
                    group="solid",
                    confidence=0.98,
                    associated_track_ids=[1],
                    evidence_sources=["foreground_motion", "ball_sized_motion"],
                    motion_score=0.9,
                    foreground_score=0.8,
                    foreground_depth_diameters=0.5,
                )
            )
        elif offset > 12:
            observations.append(PocketVisualObservation(pocket_index=0, clear=True))
        state = session.update(
            TracksFrame(
                frame_id=1 + offset,
                ts_cam_ns=1_000_000_000 + offset * 16_700_000,
                tracks=tracks,
            ),
            PocketVisualObservationFrame(
                frame_id=1 + offset,
                ts_cam_ns=1_000_000_000 + offset * 16_700_000,
                observations=observations,
            ),
        )
        if state.potted_numbers == [1]:
            break

    assert state.phase == "running"
    assert state.failure_reason is None
    assert state.potted_numbers == [1]
    potted = next(event for event in state.events if event.name == "TRAINING_BALL_POTTED")
    assert potted.payload["judgment"] == "rules_pocket_fsm"


def test_training_session_does_not_accept_projected_only_entry() -> None:
    session = TrainingSession(
        TrainingConfig(
            scenario_id="solids_then_black",
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
    assert resolved.potted_numbers == []
    assert not any(event.name == "TRAINING_BALL_POTTED" for event in resolved.events)


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


def test_runtime_pipeline_routes_training_scenarios_through_existing_planners() -> None:
    config = AppConfig()
    config.camera.backend = "synthetic"
    config.camera.width = 320
    config.camera.height = 180
    config.detector.backend = "disabled"
    config.training_detector.backend = "disabled"
    config.training.operating_mode = "training"
    config.replay.enabled = False
    pipeline = RuntimePipeline(config)
    pipeline.calibration.projection.homography = np.eye(3, dtype=np.float64)
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
        assert output.plan.shot_mode == "hook"
        assert output.state.phase.startswith("TRAINING_")
        assert pipeline.detector.mode == "training"
        assert observed_modes == ["training"]

        pipeline.select_training_scenario("solids_then_black")
        open_stage_output = pipeline.step()
        assert open_stage_output is not None
        assert open_stage_output.training is not None
        assert open_stage_output.training.expected_numbers == list(range(1, 8))
        assert open_stage_output.plan.shot_mode == "rule"

        assert pipeline.set_operating_mode("rules") == "rules"
        rule_output = pipeline.step()
        assert rule_output is not None
        assert rule_output.training is None
        assert rule_output.plan.shot_mode in {"rule", "hook", "target"}
    finally:
        pipeline.close()


def test_training_overlay_keeps_route_lines_and_adds_training_status() -> None:
    from bas.config import ProjectionConfig
    from bas.training.overlay import TrainingOverlayBuilder

    route_overlay = ProjectionOverlay(
        overlay_id="route",
        frame_id=1,
        projector_size=(1920, 1080),
        lines=[OverlayLine(points=[(10.0, 10.0), (20.0, 20.0)], label="aim")],
    )
    state = TrainingStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        scenario_id="ordered_line_1_7",
        scenario_title="ordered",
        expected_numbers=[1],
    )
    builder = TrainingOverlayBuilder(ProjectionConfig(), calibration=object())

    overlay = builder.build(TracksFrame(frame_id=1, ts_cam_ns=1), state, route_overlay=route_overlay)

    assert [line.label for line in overlay.lines] == ["aim"]
    assert any("TARGET 1" in text for _, text, _ in overlay.labels)
    assert len(overlay.texts) == 1
    assert "请完成摆球" in overlay.texts[0].text


def test_training_overlay_uses_shared_ball_center_geometry() -> None:
    from types import SimpleNamespace

    from bas.config import ProjectionConfig
    from bas.training.overlay import TrainingOverlayBuilder

    calls = []

    class _BallGeometry:
        @staticmethod
        def locate(center_px, *, radius_px, geometry_quality, geometry_method):
            calls.append((center_px, radius_px, geometry_quality, geometry_method))
            return SimpleNamespace(
                projector_ellipse=SimpleNamespace(
                    center_px=(321.0, 123.0),
                    radius_x_px=17.0,
                    radius_y_px=13.0,
                    rotation_deg=8.0,
                )
            )

    class _Calibration:
        ball_geometry = _BallGeometry()

        @staticmethod
        def ball_projector_ellipse(_center_px):
            raise AssertionError("training overlay must not bypass BallCenterGeometry")

    track = _track(1, 300.0, 200.0)
    track.radius_px = 11.5
    track.geometry_quality = 0.82
    track.geometry_method = "appearance_ellipse"
    state = TrainingStateFrame(
        frame_id=1,
        ts_cam_ns=1,
        scenario_id="ordered_line_1_7",
        scenario_title="ordered",
        phase="running",
        expected_numbers=[1],
    )

    overlay = TrainingOverlayBuilder(ProjectionConfig(), _Calibration()).build(
        TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[track]),
        state,
    )

    assert calls == [((300.0, 200.0), 11.5, 0.82, "appearance_ellipse")]
    assert overlay.circles[0].center == (321.0, 123.0)
    assert overlay.circles[0].radius == 21.25
    assert overlay.circles[0].radius_y == 16.25
    assert overlay.circles[0].rotation_deg == 8.0


def test_training_overlay_guides_every_configured_setup_zone() -> None:
    from bas.config import ProjectionConfig
    from bas.training.overlay import TrainingOverlayBuilder

    class _Calibration:
        @staticmethod
        def table_mm_to_projector_px(points):
            return np.asarray(points, dtype=np.float32)

    reachable_polygon = [
        (100.0, 50.0),
        (900.0, 50.0),
        (900.0, 450.0),
        (100.0, 450.0),
    ]
    zone_scenarios = [scenario for scenario in list_training_scenarios() if scenario.zones]
    assert zone_scenarios

    for scenario in zone_scenarios:
        session = TrainingSession(TrainingConfig(scenario_id=scenario.scenario_id))
        session.set_table_context(
            ball_center_reachable_polygon_mm=reachable_polygon,
            pockets_mm=[(100.0, 50.0)],
        )
        state = session.update(TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[]))

        overlay = TrainingOverlayBuilder(ProjectionConfig(), _Calibration()).build(
            TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[]),
            state,
        )

        guides = {
            line.label: line
            for line in overlay.lines
            if line.label.startswith("setup_target_zone_")
        }
        assert set(guides) == {
            f"setup_target_zone_{zone.ball}"
            for zone in scenario.zones
        }, scenario.scenario_id
        assert all(guide.points[0] == guide.points[-1] for guide in guides.values())
        assert {
            text.text
            for text in overlay.texts
            if text.text.startswith("摆放 ")
        } == {f"摆放 {zone.ball} 号球" for zone in scenario.zones}


def test_training_projection_prompt_uses_configured_final_canvas_position_and_size() -> None:
    from bas.config import ProjectionConfig
    from bas.training.overlay import TrainingOverlayBuilder

    config = ProjectionConfig(
        projector_width=1000,
        projector_height=500,
        training_prompt_enabled=True,
        training_prompt_x_pct=25.0,
        training_prompt_y_pct=80.0,
        training_prompt_font_size_px=44,
    )
    state = TrainingStateFrame(
        frame_id=2,
        ts_cam_ns=2,
        scenario_id="ordered_line_1_7",
        scenario_title="顺序训练",
        phase="running",
        message="当前目标：1 号球",
        expected_numbers=[1],
        progress_total=7,
    )

    overlay = TrainingOverlayBuilder(config, calibration=object()).build(
        TracksFrame(frame_id=2, ts_cam_ns=2),
        state,
    )

    assert overlay.texts[0].position == (250.0, 400.0)
    assert overlay.texts[0].font_size_px == 44.0
    assert "当前目标：1 号球" in overlay.texts[0].text


def test_training_projection_text_uses_local_calibrated_table_axis() -> None:
    from bas.config import ProjectionConfig
    from bas.training import SetupTargetZoneGuide
    from bas.training.overlay import TrainingOverlayBuilder

    angle_deg = -6.5
    angle_rad = np.deg2rad(angle_deg)
    matrix = np.asarray(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ],
        dtype=np.float32,
    )

    class _RotatedCalibration:
        @staticmethod
        def table_mm_to_projector_px(points):
            return np.asarray(points, dtype=np.float32) @ matrix.T

        @staticmethod
        def projector_px_to_table_mm(points):
            return np.asarray(points, dtype=np.float32) @ matrix

    state = TrainingStateFrame(
        frame_id=4,
        ts_cam_ns=4,
        scenario_id="BEGINNER_1_TO_3_POINTS",
        scenario_title="三点顺序",
        phase="setup",
        message="请按投影区域摆球",
        setup_target_zones=[
            SetupTargetZoneGuide(
                ball=1,
                polygon_mm=(
                    (100.0, 100.0),
                    (200.0, 100.0),
                    (200.0, 200.0),
                    (100.0, 200.0),
                    (100.0, 100.0),
                ),
            )
        ],
    )
    overlay = TrainingOverlayBuilder(
        ProjectionConfig(projector_width=1000, projector_height=500),
        _RotatedCalibration(),
    ).build(
        TracksFrame(frame_id=4, ts_cam_ns=4),
        state,
    )

    prompt = next(text for text in overlay.texts if text.text.startswith("请完成摆球"))
    placement = next(text for text in overlay.texts if text.text == "摆放 1 号球")
    assert abs(prompt.rotation_deg - angle_deg) <= 0.01
    assert abs(placement.rotation_deg - angle_deg) <= 0.01


def test_training_overlay_projects_cue_ball_goal_region_and_result_color() -> None:
    from bas.config import ProjectionConfig
    from bas.training.overlay import TrainingOverlayBuilder

    class _Calibration:
        @staticmethod
        def table_mm_to_projector_px(points):
            return np.asarray(points, dtype=np.float32) * 0.5

    state = TrainingStateFrame(
        frame_id=3,
        ts_cam_ns=3,
        scenario_id="CUE_CONTROL_STOP_ZONE",
        scenario_title="白球停球区",
        phase="running",
        cue_ball_goal_polygon_mm=[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 100.0)],
        cue_ball_goal_result="passed",
    )
    overlay = TrainingOverlayBuilder(ProjectionConfig(), _Calibration()).build(
        TracksFrame(frame_id=3, ts_cam_ns=3),
        state,
    )
    goal = next(line for line in overlay.lines if line.label == "cue_ball_goal")

    assert goal.points[0] == (50.0, 50.0)
    assert goal.points[-1] == goal.points[0]
    assert goal.style == "dashed"
    assert goal.color == (0, 220, 80)


def test_runtime_clears_pocket_observer_history_at_training_boundaries() -> None:
    config = AppConfig()
    config.camera.backend = "synthetic"
    config.detector.backend = "disabled"
    config.training_detector.backend = "disabled"
    config.training.operating_mode = "training"
    config.replay.enabled = False
    pipeline = RuntimePipeline(config)
    try:
        reset_reasons: list[str] = []
        pipeline.pocket_observer.reset = lambda: reset_reasons.append("reset")
        pipeline.training_session.start = lambda: (True, pipeline.training_session.state)

        assert pipeline.start_training()[0] is True
        pipeline.reset_training()
        pipeline.select_training_scenario("finish_6_7_8")

        assert reset_reasons == ["reset", "reset", "reset"]
    finally:
        pipeline.close()
