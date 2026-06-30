from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from bas.config import AppConfig
from bas import runtime_env
from bas.operator_controls import RuntimeControlState
from bas.schemas import DetectionsFrame, FramePacket, MatchStateFrame, ProjectionOverlay, ShotCandidate, ShotPlan, TracksFrame

runtime_env.preload_torch_for_backend = lambda backend: None

from bas.ui import main_window


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_operator_window_uses_clean_cue_sector_preview_label() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())

    assert window.cue_sector_preview_check.text() == "显示球杆矩形候选框"

    window.close()
    app.processEvents()


def test_start_pipeline_and_tick_use_same_thread(monkeypatch) -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    created: list[object] = []

    class _FakePipeline:
        def __init__(self, config, star_formula=None, control_state=None):
            self.creator_thread_id = threading.get_ident()
            self.step_thread_id: int | None = None
            self.state_machine = SimpleNamespace(turn_target_group=None, operator_hold=False)
            self.capture = SimpleNamespace(info=lambda: SimpleNamespace(backend="fake", width=1920, height=1080, fps=30.0))
            self.calibration = SimpleNamespace(
                projection=SimpleNamespace(is_valid=False),
                camera=SimpleNamespace(is_valid=False),
            )
            self.detector = SimpleNamespace(detector=SimpleNamespace(version="fake_detector"))
            self.tracker = SimpleNamespace(version="fake_tracker")
            self.planner = SimpleNamespace(version="fake_planner", learning_ranker=SimpleNamespace(version="fake_ranker"))
            self.recorder = None
            self.last_timings_ms = {}
            created.append(self)

        def step(self):
            self.step_thread_id = threading.get_ident()
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(main_window, "RuntimePipeline", _FakePipeline)
    monkeypatch.setattr(window, "_save_user_settings", lambda: None)

    window.start_pipeline()

    deadline = time.time() + 1.0
    while not created and time.time() < deadline:
        app.processEvents()
    assert created

    while window.pipeline is None and time.time() < deadline:
        app.processEvents()
    assert window.pipeline is created[0]

    window._tick()

    fake = created[0]
    assert fake.creator_thread_id == fake.step_thread_id
    window.close()


def test_tick_runtime_exception_stops_pipeline_instead_of_crashing(monkeypatch) -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    monkeypatch.setattr(main_window.QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: 0)
    monkeypatch.setattr(main_window.LOGGER, "exception", lambda *args, **kwargs: None)

    class _ExplodingPipeline:
        def __init__(self) -> None:
            self.closed = False
            self.state_machine = SimpleNamespace(turn_target_group=None)

        def step(self):
            raise RuntimeError("boom")

        def close(self) -> None:
            self.closed = True

    class _Timer:
        def stop(self) -> None:
            return None

        def start(self, *_args) -> None:
            return None

    pipeline = _ExplodingPipeline()
    window.timer = _Timer()
    window.pipeline = pipeline
    window._frame_busy = False
    window._append_log = lambda _message: None
    window._capture_timer_interval_ms = lambda *, elapsed_ms=0.0: 1

    def _stop_pipeline() -> None:
        pipeline.close()
        window.pipeline = None

    window.stop_pipeline = _stop_pipeline

    main_window.OperatorWindow._tick(window)

    assert pipeline.closed is True
    assert window.pipeline is None
    assert window._frame_busy is False


def test_settings_dialog_groups_controls_into_clear_tabs() -> None:
    app = _app()
    dialog = main_window.SettingsDialog(AppConfig(), main_window.StarFormulaConfig())

    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
        "相机采集",
        "检测状态",
        "标定几何",
        "投影输出",
        "路线策略",
        "学习数据",
        "投影修正",
        "颗星公式",
    ]
    assert dialog.tabs.usesScrollButtons() is True

    dialog.close()
    app.processEvents()


def test_draw_rule_plan_preview_uses_computed_stroke_style() -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    window.star_formula = None
    window.config = SimpleNamespace(projection=SimpleNamespace(projector_width=1280, projector_height=800))
    window.pipeline = SimpleNamespace(
        calibration=SimpleNamespace(
            table=SimpleNamespace(width_mm=2540.0, height_mm=1270.0, ball_diameter_mm=57.15),
        )
    )
    window._route_preview_stroke_style = lambda: SimpleNamespace(solid_line_width=6, dashed_line_width=4, circle_width=2)
    window._camera_radius_px = lambda point, radius_mm: 8.0
    window._inner_polygon_table = lambda: np.array(
        [[0.0, 0.0], [2540.0, 0.0], [2540.0, 1270.0], [0.0, 1270.0]],
        dtype=np.float32,
    )
    window._draw_segment_trimmed = lambda *args, **kwargs: None
    called = []
    window._draw_dashed_segment_trimmed = lambda *args, **kwargs: called.append((args, kwargs))
    window._table_mm_to_camera_px = lambda points: np.asarray(points, dtype=np.float32)

    candidate = ShotCandidate(
        candidate_id="c1",
        cue_track_id=1,
        target_track_id=2,
        target_group="solid",
        pocket_index=0,
        cue_ball=(100.0, 100.0),
        object_ball=(200.0, 130.0),
        ghost_ball=(180.0, 100.0),
        pocket_point=(300.0, 60.0),
        aim_line=[(100.0, 100.0), (180.0, 100.0)],
        object_line=[(200.0, 130.0), (300.0, 60.0)],
        cut_angle_deg=20.0,
        cue_distance_mm=80.0,
        object_distance_mm=120.0,
        score=1.0,
        risk=0.2,
    )

    main_window.OperatorWindow._draw_rule_plan_preview(window, np.zeros((32, 32, 3), dtype=np.uint8), candidate)

    assert len(called) >= 2
    assert called[1][0][4] == 4


def test_refresh_current_plan_uses_live_state_machine_turn_group() -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    window.config = AppConfig()
    window.control_state = RuntimeControlState()
    window._apply_route_display_filters = lambda out, *, force_raw=False: out
    window._update_plan = lambda _out: None
    window._update_preview = lambda _out: None
    window._refresh_projection = lambda: None

    seen: list[tuple[str | None, str | None]] = []

    class _Planner:
        def plan(self, state, frame_bgr=None, *, forced_shot_mode=None, forced_turn_target_group=None):
            seen.append((state.turn_target_group, forced_turn_target_group))
            return ShotPlan(plan_id="new", frame_id=state.frame_id, ts_cam_ns=state.ts_cam_ns)

    window.pipeline = SimpleNamespace(
        state_machine=SimpleNamespace(turn_target_group="stripe"),
        planner=_Planner(),
        overlay_builder=SimpleNamespace(
            from_plan=lambda plan: ProjectionOverlay(
                overlay_id="overlay",
                frame_id=plan.frame_id,
                projector_size=(1000, 500),
            )
        ),
        secondary_correction=None,
        _last_state=None,
        _last_plan=None,
        _last_overlay=None,
    )
    state = MatchStateFrame(frame_id=1, ts_cam_ns=1, phase="STABLE_IDLE", turn_target_group="solid")
    window.last_output = main_window.PipelineOutput(
        frame=FramePacket(frame_id=1, ts_cam_ns=1, camera_id="fake", image=np.zeros((8, 8, 3), dtype=np.uint8)),
        detections=DetectionsFrame(frame_id=1, ts_cam_ns=1),
        tracks=TracksFrame(frame_id=1, ts_cam_ns=1),
        state=state,
        plan=ShotPlan(plan_id="old", frame_id=1, ts_cam_ns=1),
        overlay=ProjectionOverlay(overlay_id="old", frame_id=1, projector_size=(1000, 500)),
    )

    main_window.OperatorWindow._refresh_current_plan(window)

    assert seen == [("stripe", "stripe")]
    assert window.last_output.state.turn_target_group == "stripe"


def test_refresh_projection_uses_composed_interaction_frame() -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    captured: list[np.ndarray] = []
    overlay = ProjectionOverlay(overlay_id="overlay", frame_id=3, projector_size=(6, 4))

    class _ProjectionWindow:
        def set_image(self, image: np.ndarray) -> None:
            captured.append(image.copy())

    window.projection_window = _ProjectionWindow()
    window._projection_calibration_mode = False
    window.pipeline = None
    window.last_output = main_window.PipelineOutput(
        frame=FramePacket(frame_id=3, ts_cam_ns=3, camera_id="fake"),
        detections=DetectionsFrame(frame_id=3, ts_cam_ns=3),
        tracks=TracksFrame(frame_id=3, ts_cam_ns=3),
        state=MatchStateFrame(frame_id=3, ts_cam_ns=3, phase="STABLE_IDLE"),
        plan=ShotPlan(plan_id="p3", frame_id=3, ts_cam_ns=3),
        overlay=overlay,
    )
    window.star_formula = main_window.StarFormulaConfig(enabled=False)
    window._projection_overlay_for_output = lambda out: overlay
    window._projection_interaction = SimpleNamespace(
        compose_frame=lambda overlay, *, star_formula, calibration=None: np.full((4, 6, 3), 77, dtype=np.uint8)
    )

    main_window.OperatorWindow._refresh_projection(window)

    assert len(captured) == 1
    assert captured[0].shape == (4, 6, 3)
    assert int(captured[0][0, 0, 0]) == 77
