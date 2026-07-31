from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from bas.config import AppConfig
from bas.capture import VideoTimelineState
from bas import runtime_env
from bas.operator_controls import RuntimeControlState
from bas.schemas import DetectionsFrame, Event, FramePacket, MatchStateFrame, ProjectionOverlay, ShotCandidate, ShotPlan, TrackObservation, TracksFrame

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


def test_operator_window_uses_scrollable_three_column_layout() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    window.resize(window.BASE_WIDTH, window.BASE_HEIGHT)
    window.show()
    app.processEvents()

    assert window.main_splitter.count() == 3
    assert window.left_scroll_area.widget() is window.sidebar
    assert window.right_scroll_area.widget() is window.right_panel
    assert window.outer_layout.indexOf(window.log_box) == -1
    assert window.right_panel.isAncestorOf(window.log_box) is True
    assert window.preview_label.parent() is window.preview_frame
    assert window.preview_caption.height() < 30
    assert abs((window.preview_label.width() / max(1, window.preview_label.height())) - (16.0 / 9.0)) < 0.05
    assert window.preview_panel.height() > window.plan_panel.height()
    assert window.pocket_notice_label.parent() is window.preview_frame
    assert window.pocket_notice_label.isHidden() is True

    window._show_desktop_pocket_notice([{"message": "sob进球"}])
    app.processEvents()

    assert window.pocket_notice_label.isVisible() is True
    assert "sob进球" in window.pocket_notice_label.text()

    window.close()
    app.processEvents()


def test_video_timeline_is_only_shown_for_video_capture_mode() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())

    assert window.video_timeline_widget.isHidden() is True

    window.backend_combo.setCurrentText("video")
    assert window.video_timeline_widget.isHidden() is False

    window.backend_combo.setCurrentText("opencv")
    assert window.video_timeline_widget.isHidden() is True

    window.close()
    app.processEvents()


def test_video_timeline_drag_seeks_pipeline_and_formats_time() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    seeks: list[int] = []

    class _VideoPipeline:
        def video_timeline_state(self) -> VideoTimelineState:
            return VideoTimelineState(current_frame=0, total_frames=3600, fps=30.0)

        def seek_video(self, frame_index: int) -> VideoTimelineState:
            seeks.append(frame_index)
            return VideoTimelineState(current_frame=frame_index, total_frames=3600, fps=30.0)

    window.backend_combo.setCurrentText("video")
    window.pipeline = _VideoPipeline()
    window._configure_video_timeline()

    assert window.video_timeline_slider.maximum() == 3599
    assert window.video_timeline_slider.isEnabled() is True
    assert window.video_timeline_label.text() == "00:00 / 02:00"

    window.video_timeline_slider.setSliderDown(True)
    window.video_timeline_slider.setSliderPosition(900)
    assert seeks == []
    assert window.video_timeline_label.text() == "00:30 / 02:00"
    window.video_timeline_slider.setSliderDown(False)

    assert seeks == [900]
    assert window.video_timeline_slider.value() == 900

    window.pipeline = None
    window.close()
    app.processEvents()


def test_video_timeline_pause_stops_ticks_and_resume_restarts_timer() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    window.backend_combo.setCurrentText("video")
    window.pipeline = SimpleNamespace()
    window._set_video_timeline_state(VideoTimelineState(current_frame=0, total_frames=300, fps=30.0))
    window.timer.start(1000)

    window.video_pause_btn.click()

    assert window._video_playback_paused is True
    assert window.timer.isActive() is False
    assert window.video_pause_btn.text() == "继续"

    window.video_pause_btn.click()

    assert window._video_playback_paused is False
    assert window.timer.isActive() is True
    assert window.video_pause_btn.text() == "暂停"

    window.pipeline = None
    window.close()
    app.processEvents()


def test_paused_video_seek_refreshes_one_frame_without_resuming() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    refresh_calls: list[bool] = []

    class _VideoPipeline:
        def video_timeline_state(self) -> VideoTimelineState:
            return VideoTimelineState(current_frame=0, total_frames=300, fps=30.0)

        def seek_video(self, frame_index: int) -> VideoTimelineState:
            return VideoTimelineState(current_frame=frame_index, total_frames=300, fps=30.0)

    window.backend_combo.setCurrentText("video")
    window.pipeline = _VideoPipeline()
    window._configure_video_timeline()
    window._set_video_playback_paused(True)
    window._tick = lambda *, allow_paused_frame=False: refresh_calls.append(allow_paused_frame)

    window._seek_video_timeline(150)

    assert refresh_calls == [True]
    assert window._video_playback_paused is True
    assert window.timer.isActive() is False

    window.pipeline = None
    window.close()
    app.processEvents()


def test_tick_does_not_read_another_frame_while_video_is_paused() -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    step_calls: list[bool] = []
    timer_starts: list[int] = []
    window._video_playback_paused = True
    window._frame_busy = False
    window.pipeline = SimpleNamespace(step=lambda: step_calls.append(True))
    window.timer = SimpleNamespace(start=lambda interval: timer_starts.append(interval))

    main_window.OperatorWindow._tick(window)

    assert step_calls == []
    assert timer_starts == []


def test_operator_window_preserves_preview_ratio_at_minimum_size() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    window.resize(1120, 720)
    window.show()
    app.processEvents()

    assert abs((window.preview_label.width() / max(1, window.preview_label.height())) - (16.0 / 9.0)) < 0.05
    assert window.left_scroll_area.verticalScrollBarPolicy() == main_window.QtCore.Qt.ScrollBarAsNeeded
    assert window.right_scroll_area.verticalScrollBarPolicy() == main_window.QtCore.Qt.ScrollBarAsNeeded
    assert window.candidates.maximumHeight() > 0

    window.close()
    app.processEvents()


def test_operator_window_preview_fills_available_viewport_without_top_gap() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    window.resize(1920, 1080)
    window.show()
    app.processEvents()

    margins = window.preview_layout.contentsMargins()
    available_width = window.preview_panel.width() - margins.left() - margins.right()
    caption_bottom = window.preview_caption.y() + window.preview_caption.height()
    caption_gap = window.preview_frame.y() - caption_bottom
    one_side_fills = (
        abs(window.preview_label.width() - window.preview_frame.width()) <= 2
        or abs(window.preview_label.height() - window.preview_frame.height()) <= 2
    )

    assert window.preview_frame.width() >= available_width - 4
    assert caption_gap <= window.preview_layout.spacing() + 2
    assert window.preview_label.y() <= 1
    assert one_side_fills is True
    assert abs((window.preview_label.width() / max(1, window.preview_label.height())) - (16.0 / 9.0)) < 0.02

    window.close()
    app.processEvents()


def test_refresh_preview_pixmap_scales_small_frame_to_preview_label() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    window.resize(1920, 1080)
    window.show()
    app.processEvents()

    window._set_preview_image(np.zeros((360, 640, 3), dtype=np.uint8))
    pixmap = window.preview_label.pixmap()

    assert pixmap is not None
    assert abs(pixmap.width() - window.preview_label.width()) <= 1
    assert abs(pixmap.height() - window.preview_label.height()) <= 1

    window.close()
    app.processEvents()


def test_preview_zoom_controls_work_for_shared_live_and_video_viewport() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    window.resize(1920, 1080)
    window.show()
    app.processEvents()

    fitted_size = window.preview_label.size()
    assert window.preview_zoom_label.text() == "100%"
    assert window.preview_fit_btn.isEnabled() is False

    window.preview_zoom_in_btn.click()
    app.processEvents()

    assert window.preview_zoom_label.text() == "125%"
    assert window.preview_label.width() > fitted_size.width()
    assert window.preview_label.height() > fitted_size.height()
    assert window.preview_fit_btn.isEnabled() is True

    window.backend_combo.setCurrentText("video")
    window.preview_zoom_out_btn.click()
    window.preview_zoom_out_btn.click()
    app.processEvents()

    assert window.preview_zoom_label.text() == "80%"
    assert window.preview_label.width() < fitted_size.width()
    assert window.preview_label.height() < fitted_size.height()

    window.preview_fit_btn.click()
    app.processEvents()

    assert window.preview_zoom_label.text() == "100%"
    assert (
        abs(window.preview_label.width() - window.preview_frame.width()) <= 2
        or abs(window.preview_label.height() - window.preview_frame.height()) <= 2
    )
    assert abs(
        (window.preview_label.width() / max(1, window.preview_label.height()))
        - (16.0 / 9.0)
    ) < 0.02
    assert window.preview_fit_btn.isEnabled() is False

    window.close()
    app.processEvents()


def test_operator_window_scene_capture_buttons_keep_tooltips_after_shortening_labels() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())

    assert window.raw_photo_btn.text() == window.RAW_PHOTO_LABEL
    assert window.raw_photo_btn.toolTip() == window.RAW_PHOTO_TOOLTIP
    assert window.instant_replay_export_btn.text() == window.INSTANT_REPLAY_LABEL
    assert window.instant_replay_export_btn.toolTip() == window.INSTANT_REPLAY_TOOLTIP
    assert window.raw_video_btn.text() == window.RAW_VIDEO_START_LABEL
    assert window.raw_video_btn.toolTip() == window.RAW_VIDEO_TOOLTIP
    assert window.route_video_btn.text() == window.ROUTE_VIDEO_START_LABEL
    assert window.route_video_btn.toolTip() == window.ROUTE_VIDEO_TOOLTIP

    window.close()
    app.processEvents()


def test_capture_raw_photo_saves_the_unmodified_camera_frame(monkeypatch, tmp_path) -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    raw_frame = np.full((6, 8, 3), 17, dtype=np.uint8)
    recording_frame = np.full((6, 8, 3), 99, dtype=np.uint8)
    saved_frames: list[np.ndarray] = []

    window.last_output = SimpleNamespace(frame=SimpleNamespace(frame_id=42))
    window._current_raw_frame = lambda: raw_frame
    window._current_recording_frame = lambda: recording_frame
    window._capture_output_dir = lambda: tmp_path
    window._append_log = lambda _message: None
    monkeypatch.setattr(
        main_window.cv2,
        "imwrite",
        lambda _path, frame, _params: saved_frames.append(frame.copy()) or True,
    )

    main_window.OperatorWindow.capture_raw_photo(window)

    assert len(saved_frames) == 1
    assert np.array_equal(saved_frames[0], raw_frame)
    assert np.array_equal(saved_frames[0], recording_frame) is False


def test_recording_base_frame_uses_the_authoritative_pipeline_frame_for_both_switch_states() -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    raw_frame = np.full((6, 8, 3), 23, dtype=np.uint8)
    window.config = SimpleNamespace(camera=SimpleNamespace(distortion_correction_enabled=False))
    window.pipeline = SimpleNamespace(capture=SimpleNamespace(frame_distortion_corrected=False))
    out = SimpleNamespace(frame=SimpleNamespace(image=raw_frame))

    disabled_frame = main_window.OperatorWindow._recording_base_frame(window, out)
    window.config.camera.distortion_correction_enabled = True
    enabled_frame = main_window.OperatorWindow._recording_base_frame(window, out)

    assert disabled_frame is raw_frame
    assert enabled_frame is raw_frame


def test_raw_and_route_video_recorders_share_the_authoritative_pipeline_frame() -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    base_frame = np.full((6, 8, 3), 31, dtype=np.uint8)
    route_frame = np.full((6, 8, 3), 47, dtype=np.uint8)
    writes: list[tuple[bool, np.ndarray]] = []
    window.config = SimpleNamespace(instant_replay=SimpleNamespace(enabled=False))
    window._recording_fps_estimator = SimpleNamespace(observe=lambda _ts: None)
    window._recording_base_frame = lambda _out: base_frame
    window._instant_replay_start_failed = False
    window._raw_video_recorder = object()
    window._route_video_recorder = object()
    window._write_video_frame = lambda *, route, frame: writes.append((route, frame))
    window._route_capture_frame = lambda _out, *, base_frame: route_frame
    window._flush_instant_replay_events = lambda: None
    out = SimpleNamespace(frame=SimpleNamespace(ts_cam_ns=123, image=base_frame))

    main_window.OperatorWindow._record_media_frames(window, out)

    assert writes == [(False, base_frame), (True, route_frame)]


def test_runtime_and_calibration_workflow_follow_current_correction_setting(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_create(_calibration_config, camera_config, **kwargs):
        calls.append(
            {
                **kwargs,
                "distortion_correction_enabled": bool(camera_config.distortion_correction_enabled),
            }
        )
        return SimpleNamespace()

    monkeypatch.setattr(main_window, "create_setting_aware_calibration_service", fake_create)
    config = AppConfig()
    config.camera.distortion_correction_enabled = False

    main_window._create_runtime_calibration_service(config)
    main_window._create_calibration_workflow_service(config)

    assert calls[0]["distortion_correction_enabled"] is False
    assert calls[1]["distortion_correction_enabled"] is False


def test_operator_window_exposes_explicit_review_controls() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())

    assert window.confirm_review_btn.text() == "确认复核"
    assert window.reject_review_btn.text() == "驳回复核"
    assert window.resolve_open_table_group_btn.text() == "确认开台花色"
    assert window.review_group_combo.itemData(0) == "solid"
    assert window.review_group_combo.itemData(1) == "stripe"

    window.close()
    app.processEvents()


def test_review_controls_follow_pending_review_payload() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    window.pipeline = SimpleNamespace(
        state_machine=SimpleNamespace(
            debug_snapshot=lambda: {
                "pending_review": {
                    "decision_ids": ["pocket:review"],
                    "review_reasons": ["visible_exceeds_ledger"],
                    "group_choice_required": True,
                    "review_pockets": [
                        {
                            "decision_id": "pocket:review",
                            "group": "solid",
                            "pocket_index": 0,
                        }
                    ],
                    "committed_pockets": [],
                }
            }
        )
    )

    window._refresh_review_controls()

    assert window.review_status_label.text().startswith("待复核")
    assert window.review_pending_list.count() == 1
    assert window.confirm_review_btn.isEnabled() is True
    assert window.reject_review_btn.isEnabled() is True
    assert window.resolve_open_table_group_btn.isEnabled() is True
    assert window.review_section.isExpanded() is True

    window.pipeline = None
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


def test_operator_window_buttons_remain_wired_after_layout_refactor(monkeypatch) -> None:
    app = _app()
    calls: list[str] = []

    def _record(name: str):
        def _handler(self, *args) -> None:
            suffix = f":{args[0]}" if name == "pocket" and args else ""
            calls.append(f"{name}{suffix}")

        return _handler

    patched_methods = [
        ("toggle_capture", "capture"),
        ("toggle_projection_window", "projection"),
        ("initialize_graphics_image_module", "init_module"),
        ("open_calibration_workbench", "projector_calib"),
        ("open_settings", "settings"),
        ("probe_camera_devices", "probe"),
        ("export_diagnostic_snapshot", "export_diag"),
        ("toggle_web_control", "web_control"),
        ("capture_raw_photo", "raw_photo"),
        ("trigger_instant_replay_export", "instant_replay"),
        ("toggle_raw_video_recording", "raw_video"),
        ("toggle_route_video_recording", "route_video"),
        ("_trigger_interaction_test_pocket", "pocket"),
        ("_trigger_interaction_test_victory", "victory"),
        ("force_state_phase", "force_phase"),
        ("toggle_state_hold", "hold_state"),
        ("toggle_deep_debug_mode", "deep_debug"),
        ("snapshot_stable_layout", "snapshot_state"),
        ("reset_state_machine", "reset_state"),
        ("confirm_state_review", "confirm_review"),
        ("reject_state_review", "reject_review"),
        ("resolve_state_open_table_group", "resolve_group"),
    ]
    for attr, name in patched_methods:
        monkeypatch.setattr(main_window.OperatorWindow, attr, _record(name))

    window = main_window.OperatorWindow(AppConfig())
    button_expectations = [
        (window.capture_btn, "capture"),
        (window.projection_btn, "projection"),
        (window.init_module_btn, "init_module"),
        (window.projector_calib_btn, "projector_calib"),
        (window.settings_btn, "settings"),
        (window.probe_btn, "probe"),
        (window.export_diag_btn, "export_diag"),
        (window.web_control_btn, "web_control"),
        (window.raw_photo_btn, "raw_photo"),
        (window.instant_replay_export_btn, "instant_replay"),
        (window.raw_video_btn, "raw_video"),
        (window.route_video_btn, "route_video"),
        (window.interaction_test_buttons["pocket0"], "pocket:0"),
        (window.interaction_test_buttons["pocket1"], "pocket:1"),
        (window.interaction_test_buttons["pocket2"], "pocket:2"),
        (window.interaction_test_buttons["pocket3"], "pocket:3"),
        (window.interaction_test_buttons["pocket4"], "pocket:4"),
        (window.interaction_test_buttons["pocket5"], "pocket:5"),
        (window.interaction_test_buttons["victory"], "victory"),
        (window.force_phase_btn, "force_phase"),
        (window.hold_state_btn, "hold_state"),
        (window.deep_debug_btn, "deep_debug"),
        (window.snapshot_state_btn, "snapshot_state"),
        (window.reset_state_btn, "reset_state"),
        (window.confirm_review_btn, "confirm_review"),
        (window.reject_review_btn, "reject_review"),
        (window.resolve_open_table_group_btn, "resolve_group"),
    ]
    for button, _expected in button_expectations:
        button.setEnabled(True)
        button.click()

    assert calls == [expected for _button, expected in button_expectations]

    window.close()
    app.processEvents()


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
        "Web 控制",
        "投影修正",
        "颗星公式",
    ]
    assert dialog.tabs.usesScrollButtons() is True

    dialog.close()
    app.processEvents()


def test_settings_dialog_applies_desktop_only_training_prompt_controls() -> None:
    app = _app()
    config = AppConfig()
    dialog = main_window.SettingsDialog(config, main_window.StarFormulaConfig())
    dialog.training_prompt_enabled.setChecked(False)
    dialog.training_prompt_x_pct.setValue(31.0)
    dialog.training_prompt_y_pct.setValue(79.0)
    dialog.training_prompt_font_size.setValue(48)

    dialog.apply_to_config(config)

    assert config.projection.training_prompt_enabled is False
    assert config.projection.training_prompt_x_pct == 31.0
    assert config.projection.training_prompt_y_pct == 79.0
    assert config.projection.training_prompt_font_size_px == 48
    dialog.close()
    app.processEvents()


def test_settings_dialog_applies_exposure_control_mode() -> None:
    app = _app()
    config = AppConfig()
    config.camera.exposure_control = "decxin"
    dialog = main_window.SettingsDialog(config, main_window.StarFormulaConfig())

    assert dialog.exposure_control.currentData() == "decxin"
    dialog.exposure_control.setCurrentIndex(dialog.exposure_control.findData("uvc"))
    dialog.apply_to_config(config)

    assert config.camera.exposure_control == "uvc"
    dialog.close()
    app.processEvents()


def test_set_running_updates_capture_action_states() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    window.config.instant_replay.enabled = True

    window._set_running(True)
    assert window.raw_photo_btn.isEnabled() is True
    assert window.instant_replay_export_btn.isEnabled() is True
    assert window.deep_debug_btn.isEnabled() is True

    window._set_running(False)
    assert window.raw_photo_btn.isEnabled() is False
    assert window.instant_replay_export_btn.isEnabled() is False
    assert window.raw_video_btn.text() == window.RAW_VIDEO_START_LABEL
    assert window.route_video_btn.text() == window.ROUTE_VIDEO_START_LABEL

    window.close()
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


def test_web_state_keeps_base_rule_separate_from_single_hook_shot_override() -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    window.config = AppConfig()
    window.config.planner.shot_mode = "rule"
    window.control_state = RuntimeControlState(hook_shot_active=True)
    window.last_output = None
    window.pipeline = None
    window.projection_window = None
    window.star_formula = SimpleNamespace(enabled=False)
    window._pending_turn_target_group = "solid"
    window._manual_web_target_id = None

    state = main_window.OperatorWindow._build_web_state(window)

    assert state["base_shot_mode"] == {"code": "rule", "name": "规则模式"}
    assert state["shot_mode"]["code"] == "hook"
    assert state["shot_mode"]["base_code"] == "rule"
    assert state["match"]["turn_group"] == "solid"
    assert state["shot_overrides"]["hook_shot_once"]["active"] is True


def test_web_state_serializes_an_available_route_for_control_responses() -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    window.config = AppConfig()
    window.control_state = RuntimeControlState()
    window.pipeline = None
    window.projection_window = None
    window.star_formula = SimpleNamespace(enabled=False)
    window._pending_turn_target_group = "solid"
    window._manual_web_target_id = None

    candidate = ShotCandidate(
        candidate_id="web-route",
        cue_track_id=1,
        target_track_id=2,
        target_group="solid",
        pocket_index=0,
        cue_ball=(100.0, 100.0),
        object_ball=(200.0, 100.0),
        ghost_ball=(180.0, 100.0),
        pocket_point=(300.0, 100.0),
        aim_line=[(100.0, 100.0), (180.0, 100.0)],
        object_line=[(200.0, 100.0), (300.0, 100.0)],
        cut_angle_deg=0.0,
        cue_distance_mm=80.0,
        object_distance_mm=100.0,
        score=0.9,
        risk=0.1,
    )
    window.last_output = main_window.PipelineOutput(
        frame=FramePacket(frame_id=7, ts_cam_ns=9, camera_id="test", image=np.zeros((9, 16, 3), dtype=np.uint8)),
        detections=DetectionsFrame(frame_id=7, ts_cam_ns=9),
        tracks=TracksFrame(frame_id=7, ts_cam_ns=9),
        state=MatchStateFrame(frame_id=7, ts_cam_ns=9, phase="STABLE_IDLE", turn_target_group="solid"),
        plan=ShotPlan(plan_id="web-plan", frame_id=7, ts_cam_ns=9, best=candidate),
        overlay=ProjectionOverlay(overlay_id="web-overlay", frame_id=7, projector_size=(16, 9)),
    )

    state = main_window.OperatorWindow._build_web_state(window)

    assert state["ok"] is True
    assert state["route"]["candidate_id"] == "web-route"
    assert state["route"]["shot_type"] == "rule"


def test_web_state_keeps_confirmed_pocket_notices_for_all_rule_ball_codes() -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    window.config = AppConfig()
    window.control_state = RuntimeControlState()
    window.last_output = None
    window.pipeline = None
    window.projection_window = None
    window.star_formula = SimpleNamespace(enabled=False)
    window._pending_turn_target_group = "solid"
    window._manual_web_target_id = None

    events = [
        Event("POCKET_CONFIRMED", 1, 1, payload={"decision_id": "pocket:cue", "group": "cue"}),
        Event("POCKET_CONFIRMED", 1, 1, payload={"decision_id": "pocket:black", "group": "black"}),
        Event("POCKET_CONFIRMED", 1, 1, payload={"decision_id": "pocket:solid", "group": "solid"}),
        Event("POCKET_CONFIRMED", 1, 1, payload={"decision_id": "pocket:stripe", "group": "stripe"}),
    ]
    main_window.OperatorWindow._observe_web_events(window, [*events, events[-1]])

    state = main_window.OperatorWindow._build_web_state(window)

    notices = state["pocket_notices"]
    assert [notice["ball_code"] for notice in notices] == ["wb", "bb", "sob", "stb"]
    assert [notice["message"] for notice in notices] == ["wb进球", "bb进球", "sob进球", "stb进球"]
    assert len({notice["sequence"] for notice in notices}) == 4


def _web_selection_window(track: TrackObservation) -> tuple[main_window.OperatorWindow, list[int], list[bool]]:
    selected_ids: list[int] = []
    refreshes: list[bool] = []
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    window.config = AppConfig()
    window.control_state = RuntimeControlState()
    window.pipeline = SimpleNamespace(
        planner=SimpleNamespace(set_manual_target=selected_ids.append),
        state_machine=SimpleNamespace(turn_target_group="solid"),
    )
    window.projection_window = None
    window.star_formula = SimpleNamespace(enabled=False)
    window._pending_turn_target_group = "solid"
    window._manual_web_target_id = None
    window._refresh_current_plan = lambda: refreshes.append(True)
    window._append_log = lambda _message: None
    window.last_output = main_window.PipelineOutput(
        frame=FramePacket(frame_id=1, ts_cam_ns=1, camera_id="test", image=np.zeros((200, 200, 3), dtype=np.uint8)),
        detections=DetectionsFrame(frame_id=1, ts_cam_ns=1),
        tracks=TracksFrame(frame_id=1, ts_cam_ns=1, tracks=[track]),
        state=MatchStateFrame(frame_id=1, ts_cam_ns=1, phase="STABLE_IDLE", layout=[track]),
        plan=ShotPlan(plan_id="p", frame_id=1, ts_cam_ns=1),
        overlay=ProjectionOverlay(overlay_id="o", frame_id=1, projector_size=(200, 200)),
    )
    return window, selected_ids, refreshes


def test_web_manual_selection_rejects_cue_ball() -> None:
    cue_track = TrackObservation(
        track_id=1,
        bbox=(85.0, 85.0, 115.0, 115.0),
        center_px=(100.0, 100.0),
        radius_px=15.0,
        cls_name="cue",
        group="cue",
        confidence=0.9,
    )
    window, selected_ids, refreshes = _web_selection_window(cue_track)

    result = main_window.OperatorWindow._select_web_target(window, {"x": 100, "y": 100})

    assert result["ok"] is False
    assert window._manual_web_target_id is None
    assert selected_ids == []
    assert refreshes == []


def test_web_manual_selection_accepts_visible_object_ball() -> None:
    object_track = TrackObservation(
        track_id=7,
        bbox=(85.0, 85.0, 115.0, 115.0),
        center_px=(100.0, 100.0),
        radius_px=15.0,
        cls_name="solid",
        group="solid",
        confidence=0.9,
    )
    window, selected_ids, refreshes = _web_selection_window(object_track)

    result = main_window.OperatorWindow._select_web_target(window, {"x": 100, "y": 100})

    assert result["ok"] is True
    assert result["target_id"] == 7
    assert window._manual_web_target_id == 7
    assert selected_ids == [7]
    assert refreshes == [True]


def test_toggle_turn_group_flips_known_group_and_does_not_guess_unknown_group() -> None:
    window = main_window.OperatorWindow.__new__(main_window.OperatorWindow)
    window.pipeline = None
    window._pending_turn_target_group = None
    messages: list[str] = []
    window._append_log = messages.append

    assert main_window.OperatorWindow._toggle_turn_target_group(window, source="test") is False
    assert window._pending_turn_target_group is None
    assert "尚未确定" in messages[-1]

    window._pending_turn_target_group = "solid"
    assert main_window.OperatorWindow._toggle_turn_target_group(window, source="test") is True
    assert window._pending_turn_target_group == "stripe"

    assert main_window.OperatorWindow._toggle_turn_target_group(window, source="test") is True
    assert window._pending_turn_target_group == "solid"


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


def test_web_target_state_is_cleared_after_shot_started() -> None:
    app = _app()
    window = main_window.OperatorWindow(AppConfig())
    window._manual_web_target_id = 7

    window._release_web_target_after_shot([Event(name="SHOT_STARTED", frame_id=2, ts_cam_ns=2)])

    assert window._manual_web_target_id is None
    window.close()
    app.processEvents()
