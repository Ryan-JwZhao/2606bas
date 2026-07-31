from __future__ import annotations

import copy
import logging
import time
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ..config import AppConfig, normalize_exposure_control
from ..runtime_env import prepare_runtime_environment, preload_torch_for_backend
from ..user_settings import UserSettings

try:
    _active_config = UserSettings.load().apply_to_config(AppConfig.load()).resolve_paths()
    prepare_runtime_environment()
    projection_only_training = (
        _active_config.training.operating_mode == "training"
        and _active_config.training.scenario_id == "OTHER_STROKE_CHECK"
    )
    active_detector_backend = (
        "disabled"
        if projection_only_training
        else _active_config.training_detector.backend
        if _active_config.training.operating_mode == "training"
        else _active_config.detector.backend
    )
    preload_torch_for_backend(active_detector_backend)
except Exception:
    prepare_runtime_environment()

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..app import PipelineOutput, RuntimePipeline
from ..calibration import (
    build_linked_patterns,
    create_setting_aware_calibration_service,
    match_linked_pattern_observation,
    projection_output_summary,
    solve_linked_projection_calibration,
)
from ..capture import (
    VideoTimelineState,
    capture_frames_are_distortion_corrected,
    create_capture_service,
    normalize_frame_rotation_degrees,
    probe_cameras,
)
from ..capture.nori_sdk import NoriProtocolController
from ..geometry import TableGeometryLoader
from ..geometry_contract import projection_calibration_context
from ..logging_config import configure_logging
from ..instant_replay import InstantReplayBuffer
from ..media_capture import FfmpegH264Recorder
from ..operator_controls import RuntimeControlState, normalize_shot_mode, toggled_object_group
from ..perception import create_detector
from ..paths import PROJECT_ROOT
from ..projection.frame_transform import normalize_projection_rotation_degrees
from ..projection.interaction import ProjectionInteractionController
from ..projection.overlay import projection_route_stroke_style
from ..projection.star_formula import StarFormulaConfig
from ..projection.window import ProjectionWindow
from ..recording_fps import RecordingFpsEstimator
from ..remote_control import RemoteCommand, RemoteCommandQueue
from ..route_freeze import MotionRouteFreezeController
from ..route_geometry import cue_alignment_start, estimate_route_end, rule_cue_separation_end
from ..schemas import MatchPhase, OverlayCircle, OverlayLine, ProjectionOverlay, to_jsonable
from ..state import normalize_state_machine_engine
from ..state_debug import StateDebugSession, StateDebugSessionResult
from ..training import (
    RULES_MODE,
    TRAINING_MODE,
    build_projection_drill_overlay,
    get_training_scenario,
    list_training_scenarios,
    normalize_operating_mode,
)
from ..utils import unit
from ..web_control import PocketNoticeTracker, WebControlServer
from .cue_sector_preview import draw_cue_sector_candidate_box
from .calibration_workbench import CalibrationWorkbenchDialog
from .engineered_ball_compensation_wizard import EngineeredBallCompensationWizardDialog
from .geometry_reference import draw_geometry_reference_lines
from .projection_debug import append_projected_ball_overlays, append_projected_boundary_overlays
from .web_control_bridge import WebControlOperatorMixin
from .widgets import AspectRatioPreviewFrame, CollapsibleSection, CompactButtonGrid

LOGGER = logging.getLogger(__name__)


COMMON_RESOLUTIONS = ["1920x1080", "3840x2160", "2560x1440", "1280x720", "1280x800", "640x480"]
COMMON_FPS = ["30", "60", "120", "164"]
TIMESTAMPED_PROJECTION_FILE_RE = re.compile(r"^(?P<base>.*?)(?:_\d{8}_\d{6})?$")
DEFAULT_PROJECTION_OUTPUT_DIR = PROJECT_ROOT / "local_settings" / "calibrations"


def format_video_time(seconds: float) -> str:
    total_seconds = max(0, int(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def projection_config_path_or_default(path_value: Optional[str]) -> Path:
    current = (path_value or "").strip()
    if current:
        return Path(current)
    return PROJECT_ROOT / "local_settings" / "projection_calibration_linked.json"


def projection_config_path_from_input(raw_value: str, current_value: Optional[str]) -> Path:
    candidate = Path((raw_value or "").strip())
    if candidate.is_absolute():
        return candidate
    base_dir = projection_config_path_or_default(current_value).parent
    return (base_dir / candidate).resolve()


def timestamped_projection_output_path(path_value: Optional[str]) -> Path:
    template = projection_config_path_or_default(path_value)
    stem_match = TIMESTAMPED_PROJECTION_FILE_RE.match(template.stem)
    base_stem = stem_match.group("base") if stem_match else template.stem
    safe_stem = (base_stem or "projection_calibration_linked").strip() or "projection_calibration_linked"
    suffix = template.suffix or ".json"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_PROJECTION_OUTPUT_DIR / f"{safe_stem}_{stamp}{suffix}"


def _configured_frame_undistorted(config: AppConfig, pipeline: Optional[RuntimePipeline] = None) -> bool:
    if pipeline is not None:
        return bool(getattr(pipeline.capture, "frame_distortion_corrected", False))
    return bool(capture_frames_are_distortion_corrected(config.camera))


def _capture_frame_size(pipeline: Optional[RuntimePipeline]) -> Optional[tuple[int, int]]:
    if pipeline is None:
        return None
    info = pipeline.capture.info()
    width, height = int(info.width), int(info.height)
    return (width, height) if width > 0 and height > 0 else None


def _create_runtime_calibration_service(
    config: AppConfig,
    pipeline: Optional[RuntimePipeline] = None,
):
    return create_setting_aware_calibration_service(
        config.calibration,
        config.camera,
        frame_undistorted=_configured_frame_undistorted(config, pipeline),
        detector_config=config.detector,
        projection_config=config.projection,
        actual_frame_size=_capture_frame_size(pipeline),
    )


def _create_calibration_workflow_service(
    config: AppConfig,
    pipeline: Optional[RuntimePipeline] = None,
    *,
    actual_frame_size: Optional[tuple[int, int]] = None,
):
    """Create calibration state in the same camera coordinate domain as runtime."""
    return create_setting_aware_calibration_service(
        config.calibration,
        config.camera,
        frame_undistorted=_configured_frame_undistorted(config, pipeline),
        detector_config=config.detector,
        projection_config=config.projection,
        actual_frame_size=actual_frame_size or _capture_frame_size(pipeline),
    )


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, config: AppConfig, star_formula: StarFormulaConfig, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.config = config
        self._operator = parent if hasattr(parent, "_apply_projection_tuning_preview") else None
        self._original_calibration = copy.deepcopy(config.calibration)
        self._preview_active = False
        self.setWindowTitle("设置")
        self.resize(820, 760)
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        tabs.setUsesScrollButtons(True)
        tabs.setElideMode(QtCore.Qt.ElideRight)
        self.tabs = tabs
        layout.addWidget(tabs, 1)
        self.model_path = self._path_row(config.detector.model_path, "模型文件 (*.pt *.onnx *.engine);;所有文件 (*.*)")
        self.class_file_path = self._path_row(config.detector.class_file_path, "类别文件 (*.txt *.json);;所有文件 (*.*)")
        self.training_model_path = self._path_row(
            config.training_detector.model_path,
            "编号球模型文件 (*.pt *.onnx *.engine);;所有文件 (*.*)",
        )
        self.learning_ranker_enabled = QtWidgets.QCheckBox("启用学习排序")
        self.learning_ranker_enabled.setChecked(bool(config.learning.ranker_enabled))
        self.learning_ranker_model_path = self._path_row(config.learning.ranker_model_path, "学习排序模型 (*.json);;所有文件 (*.*)")
        self.learning_score_blend = self._dspin(float(config.learning.score_blend), 0.0, 1.0, 0.1)
        learning_ranker_box = QtWidgets.QWidget()
        learning_ranker_layout = QtWidgets.QHBoxLayout(learning_ranker_box)
        learning_ranker_layout.setContentsMargins(0, 0, 0, 0)
        learning_ranker_layout.addWidget(self.learning_ranker_enabled)
        learning_ranker_layout.addWidget(self.learning_ranker_model_path, 1)
        learning_ranker_layout.addWidget(QtWidgets.QLabel("权重"))
        learning_ranker_layout.addWidget(self.learning_score_blend)
        self.learning_collect_enabled = QtWidgets.QCheckBox("启用学习样本采集")
        self.learning_collect_enabled.setChecked(bool(config.learning.collect_enabled))
        self.learning_samples_directory = self._dir_row(config.learning.samples_directory)
        learning_collect_box = QtWidgets.QWidget()
        learning_collect_layout = QtWidgets.QHBoxLayout(learning_collect_box)
        learning_collect_layout.setContentsMargins(0, 0, 0, 0)
        learning_collect_layout.addWidget(self.learning_collect_enabled)
        learning_collect_layout.addWidget(self.learning_samples_directory, 1)
        self.video_path = self._path_row(config.camera.video_path, "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)")
        self.nori_sdk_root = self._dir_row(config.camera.nori_sdk_root)
        self.frame_rotation_degrees = QtWidgets.QComboBox()
        for degrees in (0, 90, 180, 270):
            label = "不旋转" if degrees == 0 else f"顺时针 {degrees}°"
            self.frame_rotation_degrees.addItem(label, degrees)
        rotation_index = self.frame_rotation_degrees.findData(
            normalize_frame_rotation_degrees(config.camera.frame_rotation_degrees)
        )
        self.frame_rotation_degrees.setCurrentIndex(max(0, rotation_index))
        self.distortion_enabled = QtWidgets.QCheckBox("启用")
        self.distortion_enabled.setChecked(bool(config.camera.distortion_correction_enabled))
        self.distortion_file = self._path_row(config.camera.distortion_correction_file, "OpenCV 标定文件 (*.yaml *.yml *.xml);;所有文件 (*.*)")
        distortion_box = QtWidgets.QWidget()
        distortion_layout = QtWidgets.QHBoxLayout(distortion_box)
        distortion_layout.setContentsMargins(0, 0, 0, 0)
        distortion_layout.addWidget(self.distortion_enabled)
        distortion_layout.addWidget(self.distortion_file, 1)
        self.exposure_control = QtWidgets.QComboBox()
        self.exposure_control.addItem("自动选择", "auto")
        self.exposure_control.addItem("Decxin SDK", "decxin")
        self.exposure_control.addItem("UVC / DirectShow", "uvc")
        exposure_control_index = self.exposure_control.findData(
            normalize_exposure_control(config.camera.exposure_control)
        )
        self.exposure_control.setCurrentIndex(max(0, exposure_control_index))
        self.exposure_auto = QtWidgets.QCheckBox("自动曝光")
        self.exposure_auto.setChecked(bool(config.camera.exposure_auto))
        self.exposure_level = self._spin(int(config.camera.exposure_level if config.camera.exposure_level is not None else -5), -10, 0)
        self.exposure_level.setEnabled(not self.exposure_auto.isChecked())
        self.exposure_auto.toggled.connect(lambda checked: self.exposure_level.setEnabled(not checked))
        exposure_box = QtWidgets.QWidget()
        exposure_layout = QtWidgets.QHBoxLayout(exposure_box)
        exposure_layout.setContentsMargins(0, 0, 0, 0)
        exposure_layout.addWidget(self.exposure_control)
        exposure_layout.addWidget(self.exposure_auto)
        exposure_layout.addWidget(QtWidgets.QLabel("手动档位"))
        exposure_layout.addWidget(self.exposure_level)
        exposure_layout.addStretch(1)
        wb_min, wb_max, wb_step, wb_value = self._white_balance_control_spec(config)
        self.white_balance_auto = QtWidgets.QCheckBox("自动白平衡")
        self.white_balance_auto.setChecked(bool(config.camera.white_balance_auto))
        self.white_balance_value = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.white_balance_value.setRange(wb_min, wb_max)
        self.white_balance_value.setSingleStep(wb_step)
        self.white_balance_value.setPageStep(max(wb_step, min(1000, wb_step * 10)))
        self.white_balance_value.setTickInterval(max(wb_step, min(1000, wb_step * 5)))
        self.white_balance_value.setValue(wb_value)
        self.white_balance_value.setEnabled(not self.white_balance_auto.isChecked())
        self.white_balance_value_label = QtWidgets.QLabel()
        self.white_balance_value_label.setMinimumWidth(90)
        self.white_balance_value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._update_white_balance_value_label(self.white_balance_value.value())
        self.white_balance_auto.toggled.connect(lambda checked: self.white_balance_value.setEnabled(not checked))
        self.white_balance_value.valueChanged.connect(self._update_white_balance_value_label)
        white_balance_box = QtWidgets.QWidget()
        white_balance_layout = QtWidgets.QHBoxLayout(white_balance_box)
        white_balance_layout.setContentsMargins(0, 0, 0, 0)
        white_balance_layout.addWidget(self.white_balance_auto)
        white_balance_layout.addWidget(QtWidgets.QLabel("色温"))
        white_balance_layout.addWidget(self.white_balance_value, 1)
        white_balance_layout.addWidget(self.white_balance_value_label)
        self.outline_path = self._path_row(config.geometry.outline_path, "JSON (*.json);;所有文件 (*.*)")
        self.inline_path = self._path_row(config.geometry.inline_path, "JSON (*.json);;所有文件 (*.*)")
        self.pocket_path = self._path_row(config.geometry.pocket_path, "JSON (*.json);;所有文件 (*.*)")
        self.camera_calibration_file = self._path_row(config.calibration.camera_file, "OpenCV 标定文件 (*.yaml *.yml *.xml);;所有文件 (*.*)")
        self.projection_mode = QtWidgets.QComboBox()
        self.projection_mode.addItem("独立二维几何（新）", "engineered")
        selected_mode = "engineered"
        selected_index = self.projection_mode.findData(selected_mode)
        self.projection_mode.setCurrentIndex(max(0, selected_index))
        self._legacy_projection_file = config.calibration.legacy_projection_file
        self._engineered_projection_file = config.calibration.engineered_plane_projection_file
        if not self._legacy_projection_file and selected_mode == "legacy":
            self._legacy_projection_file = config.calibration.projection_file
        if not self._engineered_projection_file and selected_mode == "engineered":
            self._engineered_projection_file = config.calibration.projection_file
        self._projection_mode_for_path = selected_mode
        self.projection_calibration_file = self._path_row(
            self._active_projection_path_for_mode(selected_mode),
            "投影平面校正文件 (*.json);;所有文件 (*.*)",
        )
        self.engineered_ball_compensation_file = self._path_row(
            config.calibration.engineered_ball_compensation_file,
            "球体补偿文件 (*.json);;所有文件 (*.*)",
        )
        self.projection_mode.currentIndexChanged.connect(self._projection_mode_changed)
        self.detector_backend = QtWidgets.QComboBox()
        self.detector_backend.addItems(["disabled", "ultralytics", "debug_color"])
        self.detector_backend.setCurrentText(config.detector.backend)
        self.training_detector_backend = QtWidgets.QComboBox()
        self.training_detector_backend.addItems(["disabled", "ultralytics"])
        self.training_detector_backend.setCurrentText(config.training_detector.backend)
        self.state_machine_engine = QtWidgets.QComboBox()
        self.state_machine_engine.addItem("状态机（旧）", "legacy")
        self.state_machine_engine.addItem("状态机（新）", "modern")
        state_engine_index = self.state_machine_engine.findData(normalize_state_machine_engine(getattr(config.state, "engine", "legacy")))
        self.state_machine_engine.setCurrentIndex(max(0, state_engine_index))
        self.detect_interval = self._spin(int(config.detector.detect_interval_frames), 1, 12)
        self.detect_fps_limit = self._dspin(float(config.detector.detect_fps_limit_hz), 0.0, 30.0, 0.5)
        self.web_control_host = QtWidgets.QLineEdit(str(config.web_control.host or "0.0.0.0"))
        self.web_control_host.setPlaceholderText("0.0.0.0")
        self.web_control_port = self._spin(int(config.web_control.port), 1, 65535)
        web_control_note = QtWidgets.QLabel("0.0.0.0 表示允许局域网设备访问；保存后由主界面的“开启 Web 控制”按钮启动。")
        web_control_note.setWordWrap(True)
        self.proj_screen = QtWidgets.QComboBox()
        for idx, screen in enumerate(QtWidgets.QApplication.screens()):
            geo = screen.geometry()
            self.proj_screen.addItem(f"{idx}: {screen.name()} {geo.width()}x{geo.height()}", idx)
        if self.proj_screen.count() == 0:
            self.proj_screen.addItem("0", 0)
        self.proj_screen.setCurrentIndex(max(0, min(config.projection.screen_index, self.proj_screen.count() - 1)))
        self.proj_w = QtWidgets.QSpinBox()
        self.proj_w.setRange(320, 7680)
        self.proj_w.setValue(config.projection.projector_width)
        self.proj_h = QtWidgets.QSpinBox()
        self.proj_h.setRange(240, 4320)
        self.proj_h.setValue(config.projection.projector_height)
        self.projection_calibration_rotation = QtWidgets.QComboBox()
        self.projection_output_rotation = QtWidgets.QComboBox()
        for degrees in (0, 90, 180, 270):
            label = "不旋转" if degrees == 0 else f"顺时针 {degrees}°"
            self.projection_calibration_rotation.addItem(label, degrees)
            self.projection_output_rotation.addItem(label, degrees)
        calibration_rotation_index = self.projection_calibration_rotation.findData(
            normalize_projection_rotation_degrees(
                config.projection.calibration_rotation_degrees
            )
        )
        output_rotation_index = self.projection_output_rotation.findData(
            normalize_projection_rotation_degrees(config.projection.output_rotation_degrees)
        )
        self.projection_calibration_rotation.setCurrentIndex(max(0, calibration_rotation_index))
        self.projection_output_rotation.setCurrentIndex(max(0, output_rotation_index))
        self.projection_calibration_rotation.setToolTip(
            "仅用于联合校准图样、球心补偿目标和校准结果；修改后需重新校准。"
        )
        self.projection_output_rotation.setToolTip(
            "仅旋转正常运行时的最终投影帧，不改变相机坐标、几何标注或校准细调角。"
        )
        proj_size = QtWidgets.QHBoxLayout()
        proj_size.addWidget(self.proj_w)
        proj_size.addWidget(QtWidgets.QLabel("x"))
        proj_size.addWidget(self.proj_h)
        self.auto_pocket_animation_enabled = QtWidgets.QCheckBox("开启击球特效")
        self.auto_pocket_animation_enabled.setChecked(bool(config.projection.auto_pocket_animation_enabled))
        self.auto_victory_animation_enabled = QtWidgets.QCheckBox("开启胜利动画")
        self.auto_victory_animation_enabled.setChecked(bool(config.projection.auto_victory_animation_enabled))
        projection_effects_box = QtWidgets.QWidget()
        projection_effects_layout = QtWidgets.QVBoxLayout(projection_effects_box)
        projection_effects_layout.setContentsMargins(0, 0, 0, 0)
        projection_effects_layout.addWidget(self.auto_pocket_animation_enabled)
        projection_effects_layout.addWidget(self.auto_victory_animation_enabled)
        projection_effects_layout.addStretch(1)
        self.training_prompt_enabled = QtWidgets.QCheckBox("显示训练中文提示")
        self.training_prompt_enabled.setChecked(bool(config.projection.training_prompt_enabled))
        self.training_prompt_x_pct = self._dspin(
            float(config.projection.training_prompt_x_pct),
            0.0,
            100.0,
            1.0,
        )
        self.training_prompt_x_pct.setSuffix(" %")
        self.training_prompt_y_pct = self._dspin(
            float(config.projection.training_prompt_y_pct),
            0.0,
            100.0,
            1.0,
        )
        self.training_prompt_y_pct.setSuffix(" %")
        self.training_prompt_font_size = self._spin(
            int(config.projection.training_prompt_font_size_px),
            12,
            240,
        )
        self.training_prompt_font_size.setSuffix(" px")
        training_prompt_box = QtWidgets.QGroupBox("训练交互提示")
        training_prompt_grid = QtWidgets.QGridLayout(training_prompt_box)
        training_prompt_grid.addWidget(self.training_prompt_enabled, 0, 0, 1, 4)
        self._grid_pair(
            training_prompt_grid,
            1,
            "水平位置（左 0 / 右 100）",
            self.training_prompt_x_pct,
            "垂直位置（上 0 / 下 100）",
            self.training_prompt_y_pct,
        )
        training_prompt_grid.addWidget(QtWidgets.QLabel("字号"), 2, 0)
        training_prompt_grid.addWidget(self.training_prompt_font_size, 2, 1)
        training_prompt_note = QtWidgets.QLabel(
            "位置按最终投影画布百分比保存，默认位于中间偏下；文字在几何校准完成后绘制，不会随校准矩阵拉伸。"
        )
        training_prompt_note.setWordWrap(True)
        training_prompt_grid.addWidget(training_prompt_note, 2, 2, 1, 2)
        self.training_prompt_enabled.toggled.connect(self._sync_training_prompt_inputs)
        self._sync_training_prompt_inputs()
        self.route_freeze_enabled = QtWidgets.QCheckBox("运动时冻结路线")
        self.route_freeze_enabled.setChecked(bool(config.planner.route_freeze_enabled))
        self.route_freeze_enter_frames = self._spin(int(config.planner.route_freeze_enter_frames), 1, 30)
        self.route_freeze_release_frames = self._spin(int(config.planner.route_freeze_release_frames), 1, 90)
        self.route_freeze_same_route_refresh_mm = self._dspin(float(config.planner.route_freeze_same_route_refresh_mm), 0.0, 120.0, 1.0)
        self.route_freeze_same_route_refresh_score_delta = self._dspin(
            float(config.planner.route_freeze_same_route_refresh_score_delta),
            0.0,
            2.0,
            0.01,
            decimals=2,
        )
        self.route_freeze_switch_confirm_frames = self._spin(int(config.planner.route_freeze_switch_confirm_frames), 1, 30)
        self.route_freeze_switch_min_distance_mm = self._dspin(float(config.planner.route_freeze_switch_min_distance_mm), 0.0, 240.0, 1.0)
        self.route_freeze_switch_min_score_delta = self._dspin(
            float(config.planner.route_freeze_switch_min_score_delta),
            0.0,
            2.0,
            0.01,
            decimals=2,
        )
        self.cue_sector_enabled = QtWidgets.QCheckBox("启用球杆矩形走廊纠正")
        self.cue_sector_enabled.setChecked(bool(config.planner.cue_sector_correction_enabled))
        self.cue_sector_corridor_width = self._dspin(float(config.planner.cue_sector_corridor_width_px), 1.0, 600.0, 5.0)
        self.cue_sector_switch_confirm_frames = self._spin(int(config.planner.cue_sector_switch_confirm_frames), 1, 30)
        self.cue_sector_switch_min_score_delta = self._dspin(
            float(config.planner.cue_sector_switch_min_score_delta),
            0.0,
            2.0,
            0.01,
            decimals=2,
        )
        self.cue_sector_require_balls_stationary = QtWidgets.QCheckBox("只在所有球静止时生效")
        self.cue_sector_require_balls_stationary.setChecked(bool(config.planner.cue_sector_require_balls_stationary))
        self.cue_sector_stationary_speed_mm = self._dspin(float(config.planner.cue_sector_stationary_speed_mm_s), 0.0, 120.0, 0.5)
        self.cue_sector_stationary_speed_px = self._dspin(float(config.planner.cue_sector_stationary_speed_px_s), 0.0, 300.0, 1.0)
        self.target_lock_enabled = QtWidgets.QCheckBox("启用用户意图目标锁定")
        self.target_lock_enabled.setChecked(bool(config.planner.target_lock_enabled))
        self.target_lock_confirm_frames = self._spin(int(config.planner.target_lock_confirm_frames), 1, 60)
        self.target_lock_switch_confirm_frames = self._spin(int(config.planner.target_lock_switch_confirm_frames), 1, 120)
        self.target_shot_enabled = QtWidgets.QCheckBox("启用目标击球模式")
        self.target_shot_enabled.setChecked(bool(config.planner.target_shot_enabled))
        self.target_shot_activate_hold_seconds = self._dspin(
            float((config.planner.target_shot_activate_hold_ms or 1000) / 1000.0),
            0.1,
            30.0,
            0.1,
            decimals=2,
        )
        self.target_shot_switch_hold_seconds = self._dspin(
            float((config.planner.target_shot_switch_hold_ms or 1500) / 1000.0),
            0.1,
            30.0,
            0.1,
            decimals=2,
        )
        self.target_shot_miss_grace_seconds = self._dspin(
            float((config.planner.target_shot_miss_grace_ms or 200) / 1000.0),
            0.0,
            5.0,
            0.05,
            decimals=2,
        )
        self.target_shot_release_confirm_seconds = self._dspin(
            float((config.planner.target_shot_release_confirm_ms or 300) / 1000.0),
            0.0,
            5.0,
            0.05,
            decimals=2,
        )
        route_freeze_box = QtWidgets.QGroupBox("路线防闪烁")
        route_freeze_grid = QtWidgets.QGridLayout(route_freeze_box)
        route_freeze_grid.addWidget(self.route_freeze_enabled, 0, 0, 1, 4)
        self._grid_pair(route_freeze_grid, 1, "进入冻结连续帧", self.route_freeze_enter_frames, "解冻连续稳定帧", self.route_freeze_release_frames)
        self._grid_pair(
            route_freeze_grid,
            2,
            "同路线刷新位移(mm)",
            self.route_freeze_same_route_refresh_mm,
            "同路线刷新分差",
            self.route_freeze_same_route_refresh_score_delta,
        )
        self._grid_pair(
            route_freeze_grid,
            3,
            "切换确认连续帧",
            self.route_freeze_switch_confirm_frames,
            "切换最小位移(mm)",
            self.route_freeze_switch_min_distance_mm,
        )
        route_freeze_grid.addWidget(QtWidgets.QLabel("切换最小分差"), 4, 0)
        route_freeze_grid.addWidget(self.route_freeze_switch_min_score_delta, 4, 1)
        route_freeze_grid.addWidget(QtWidgets.QLabel("建议先从进入冻结 2、解冻 8、同路线 12mm、切换确认 3 开始调。"), 4, 2, 1, 2)
        cue_sector_box = QtWidgets.QGroupBox("球杆矩形走廊纠正")
        cue_sector_grid = QtWidgets.QGridLayout(cue_sector_box)
        cue_sector_grid.addWidget(self.cue_sector_enabled, 0, 0, 1, 4)
        cue_sector_grid.addWidget(self.cue_sector_require_balls_stationary, 1, 0, 1, 4)
        cue_sector_grid.addWidget(QtWidgets.QLabel("矩形总宽度(px)"), 2, 0)
        cue_sector_grid.addWidget(self.cue_sector_corridor_width, 2, 1)
        self._grid_pair(
            cue_sector_grid,
            3,
            "切换确认连续帧",
            self.cue_sector_switch_confirm_frames,
            "切换最小分差",
            self.cue_sector_switch_min_score_delta,
        )
        self._grid_pair(
            cue_sector_grid,
            4,
            "静止阈值(mm/s)",
            self.cue_sector_stationary_speed_mm,
            "静止阈值(px/s)",
            self.cue_sector_stationary_speed_px,
        )
        cue_sector_grid.addWidget(
            QtWidgets.QLabel("矩形总宽度按球杆方向的横向像素距离计算；前向长度不限制；静止判断只检查球，不要求球杆静止。"),
            5,
            0,
            1,
            4,
        )
        target_lock_box = QtWidgets.QGroupBox("用户意图目标锁定")
        target_lock_grid = QtWidgets.QGridLayout(target_lock_box)
        target_lock_grid.addWidget(self.target_lock_enabled, 0, 0, 1, 4)
        self._grid_pair(
            target_lock_grid,
            1,
            "首次锁定连续帧",
            self.target_lock_confirm_frames,
            "切换锁定连续帧",
            self.target_lock_switch_confirm_frames,
        )
        target_lock_grid.addWidget(
            QtWidgets.QLabel("锁定后，伸缩球杆、短暂无球杆和出杆运动阶段不会切换目标球；只有连续明确指向另一颗球才换锁。"),
            2,
            0,
            1,
            4,
        )
        target_shot_box = QtWidgets.QGroupBox("目标击球模式")
        target_shot_grid = QtWidgets.QGridLayout(target_shot_box)
        target_shot_grid.addWidget(self.target_shot_enabled, 0, 0, 1, 4)
        self._grid_pair(
            target_shot_grid,
            1,
            "首次激活保持(秒)",
            self.target_shot_activate_hold_seconds,
            "改锁保持(秒)",
            self.target_shot_switch_hold_seconds,
        )
        self._grid_pair(
            target_shot_grid,
            2,
            "短暂丢帧容忍(秒)",
            self.target_shot_miss_grace_seconds,
            "释放确认(秒)",
            self.target_shot_release_confirm_seconds,
        )
        target_shot_grid.addWidget(
            QtWidgets.QLabel("目标击球模式按真实时间累计，不再直接按帧计数；状态文本会显示毫秒进度，便于判断是未累计够还是中途被重置。"),
            3,
            0,
            1,
            4,
        )
        self._add_form_tab(
            tabs,
            "相机采集",
            [
                ("视频文件路径", self.video_path),
                ("Nori SDK 目录", self.nori_sdk_root),
                ("相机画面旋转（已有标注保持0°）", self.frame_rotation_degrees),
                ("工业相机畸变矫正", distortion_box),
                ("工业相机曝光", exposure_box),
                ("工业相机白平衡", white_balance_box),
            ],
        )
        self._add_form_tab(
            tabs,
            "检测状态",
            [
                ("检测后端", self.detector_backend),
                ("台球模型路径", self.model_path),
                ("类别文件路径", self.class_file_path),
                ("训练检测后端", self.training_detector_backend),
                ("0–15 编号球模型", self.training_model_path),
                ("检测间隔(帧)", self.detect_interval),
                ("检测频率上限(Hz)", self.detect_fps_limit),
                ("台球状态机", self.state_machine_engine),
            ],
        )
        self._add_form_tab(
            tabs,
            "标定几何",
            [
                ("outline.json", self.outline_path),
                ("inline.json", self.inline_path),
                ("pocket.json", self.pocket_path),
                ("相机标定文件", self.camera_calibration_file),
                ("联合平面校准文件（自动管理）", self.projection_calibration_file),
                ("球心补偿文件（自动管理）", self.engineered_ball_compensation_file),
            ],
        )
        self._add_form_tab(
            tabs,
            "投影输出",
            [
                ("默认投影设备", self.proj_screen),
                ("默认投影分辨率", proj_size),
                ("校准画面旋转", self.projection_calibration_rotation),
                ("运行输出旋转", self.projection_output_rotation),
                ("自动投影动效", projection_effects_box),
                ("训练中文提示", training_prompt_box),
            ],
        )
        self._add_widget_tab(tabs, "路线策略", [target_shot_box, target_lock_box, route_freeze_box, cue_sector_box])
        self._add_form_tab(
            tabs,
            "学习数据",
            [
                ("学习排序模型", learning_ranker_box),
                ("学习样本目录", learning_collect_box),
            ],
        )
        self._add_form_tab(
            tabs,
            "Web 控制",
            [
                ("监听地址", self.web_control_host),
                ("接口端口", self.web_control_port),
                ("说明", web_control_note),
            ],
        )

        tuning = QtWidgets.QWidget()
        tuning_layout = QtWidgets.QVBoxLayout(tuning)
        tuning_note = QtWidgets.QLabel(
            "这里的参数单位都是 mm。调节后会立即刷新当前投影预览，便于直接观察边界线和球圈变化。<br>"
            "如果直接关闭设置窗口而不保存，本页的修改不会落盘，也会恢复到打开设置前的状态。"
        )
        tuning_note.setWordWrap(True)
        tuning_layout.addWidget(tuning_note)

        boundary_box = QtWidgets.QGroupBox("边界修正")
        boundary_grid = QtWidgets.QGridLayout(boundary_box)
        self.visible_inset_top = self._dspin(float(config.calibration.projection_visible_inset_top_mm), 0.0, 120.0, 1.0)
        self.visible_inset_right = self._dspin(float(config.calibration.projection_visible_inset_right_mm), 0.0, 120.0, 1.0)
        self.visible_inset_bottom = self._dspin(float(config.calibration.projection_visible_inset_bottom_mm), 0.0, 120.0, 1.0)
        self.visible_inset_left = self._dspin(float(config.calibration.projection_visible_inset_left_mm), 0.0, 120.0, 1.0)
        self.physical_inset_top = self._dspin(float(config.calibration.physical_rail_inset_top_mm), 0.0, 120.0, 1.0)
        self.physical_inset_right = self._dspin(float(config.calibration.physical_rail_inset_right_mm), 0.0, 120.0, 1.0)
        self.physical_inset_bottom = self._dspin(float(config.calibration.physical_rail_inset_bottom_mm), 0.0, 120.0, 1.0)
        self.physical_inset_left = self._dspin(float(config.calibration.physical_rail_inset_left_mm), 0.0, 120.0, 1.0)
        self.middle_relief_top = self._dspin(float(config.calibration.physical_middle_pocket_relief_top_mm), 0.0, 120.0, 1.0)
        self.middle_relief_bottom = self._dspin(float(config.calibration.physical_middle_pocket_relief_bottom_mm), 0.0, 120.0, 1.0)
        self.center_margin = self._dspin(float(config.calibration.center_reachable_extra_margin_mm), 0.0, 80.0, 0.5)
        row = 0
        self._grid_pair(boundary_grid, row, "可见边界-上 (mm)", self.visible_inset_top, "可见边界-右 (mm)", self.visible_inset_right)
        row += 1
        self._grid_pair(boundary_grid, row, "可见边界-下 (mm)", self.visible_inset_bottom, "可见边界-左 (mm)", self.visible_inset_left)
        row += 1
        self._grid_pair(boundary_grid, row, "物理库边-上 (mm)", self.physical_inset_top, "物理库边-右 (mm)", self.physical_inset_right)
        row += 1
        self._grid_pair(boundary_grid, row, "物理库边-下 (mm)", self.physical_inset_bottom, "物理库边-左 (mm)", self.physical_inset_left)
        row += 1
        self._grid_pair(boundary_grid, row, "上中袋 relief (mm)", self.middle_relief_top, "下中袋 relief (mm)", self.middle_relief_bottom)
        row += 1
        boundary_grid.addWidget(QtWidgets.QLabel("球心额外安全边 (mm)"), row, 0)
        boundary_grid.addWidget(self.center_margin, row, 1)
        boundary_grid.addWidget(QtWidgets.QLabel("说明"), row, 2)
        boundary_grid.addWidget(QtWidgets.QLabel("relief 用于把上中袋、下中袋附近的物理库边往可见边界方向拉回。"), row, 3)
        tuning_layout.addWidget(boundary_box)

        ball_box = QtWidgets.QGroupBox("球心投影修正")
        ball_grid = QtWidgets.QGridLayout(ball_box)
        self.ball_comp_enabled = QtWidgets.QCheckBox("启用球心补偿")
        self.ball_comp_enabled.setChecked(bool(config.calibration.ball_center_compensation_enabled))
        self.ball_comp_auto_ref = QtWidgets.QCheckBox("使用画面中心作为参考点")
        self.ball_comp_auto_ref.setChecked(bool(config.calibration.ball_center_compensation_auto_reference))
        ref_x_value = float(config.calibration.ball_center_compensation_ref_x_px)
        ref_y_value = float(config.calibration.ball_center_compensation_ref_y_px)
        if self.ball_comp_auto_ref.isChecked() and abs(ref_x_value) < 1e-6 and abs(ref_y_value) < 1e-6:
            ref_x_value = 0.5 * float(max(0, int(config.camera.width)))
            ref_y_value = 0.5 * float(max(0, int(config.camera.height)))
        self.ball_comp_ref_x = self._dspin(ref_x_value, -7680.0, 7680.0, 1.0)
        self.ball_comp_ref_y = self._dspin(ref_y_value, -4320.0, 4320.0, 1.0)
        self.ball_comp_scale_x = self._dspin(float(config.calibration.ball_center_compensation_scale_x_pct), -20.0, 20.0, 0.1)
        self.ball_comp_scale_y = self._dspin(float(config.calibration.ball_center_compensation_scale_y_pct), -20.0, 20.0, 0.1)
        ball_grid.addWidget(self.ball_comp_enabled, 0, 0, 1, 4)
        ball_grid.addWidget(self.ball_comp_auto_ref, 1, 0, 1, 4)
        self._grid_pair(ball_grid, 2, "参考点 X (px)", self.ball_comp_ref_x, "参考点 Y (px)", self.ball_comp_ref_y)
        self._grid_pair(ball_grid, 3, "X 拉回比例 (%)", self.ball_comp_scale_x, "Y 拉回比例 (%)", self.ball_comp_scale_y)
        ball_grid.addWidget(QtWidgets.QLabel("正数表示把球心向参考点拉回。默认会以当前相机画面中心作为参考点；取消勾选后可手动输入，支持负值，适合处理投影仪中心落在画面外的情况。"), 4, 0, 1, 4)
        self.ball_comp_auto_ref.toggled.connect(self._sync_ball_reference_inputs)
        self._sync_ball_reference_inputs()
        tuning_layout.addWidget(ball_box)
        tuning_layout.addStretch(1)
        tabs.addTab(tuning, "投影修正")
        self._bind_projection_tuning_preview()

        star = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(star)
        self.star_enabled = QtWidgets.QCheckBox("启用颗星公式")
        self.star_enabled.setChecked(star_formula.enabled)
        self.star_left = self._dspin(star_formula.inset_left_pct, -20, 40, 0.2)
        self.star_right = self._dspin(star_formula.inset_right_pct, -20, 40, 0.2)
        self.star_top = self._dspin(star_formula.inset_top_pct, -20, 40, 0.2)
        self.star_bottom = self._dspin(star_formula.inset_bottom_pct, -20, 40, 0.2)
        self.star_label_tb = self._dspin(star_formula.label_offset_tb_pct, 0, 30, 0.2)
        self.star_label_lr = self._dspin(star_formula.label_offset_lr_pct, 0, 30, 0.2)
        self.star_offset_x = self._spin(int(star_formula.offset_x_px), -7680, 7680)
        self.star_offset_y = self._spin(int(star_formula.offset_y_px), -4320, 4320)
        self.star_move_step = self._spin(10, 1, 500)
        self.star_scale_x = self._dspin(star_formula.scale_x_pct, 10, 300, 0.5)
        self.star_scale_y = self._dspin(star_formula.scale_y_pct, 10, 300, 0.5)
        self.star_scale_step = self._dspin(1.0, 0.1, 50, 0.1)
        self.star_angle = self._dspin(star_formula.angle_deg, -180, 180, 0.5)
        self.star_angle_step = self._dspin(0.5, 0.1, 45, 0.1)
        row = 0
        grid.addWidget(self.star_enabled, row, 0, 1, 4)
        row += 1
        self._grid_pair(grid, row, "Left Inset (%)", self.star_left, "Right Inset (%)", self.star_right)
        row += 1
        self._grid_pair(grid, row, "Top Inset (%)", self.star_top, "Bottom Inset (%)", self.star_bottom)
        row += 1
        self._grid_pair(grid, row, "Top/Bottom Label (%)", self.star_label_tb, "Left/Right Label (%)", self.star_label_lr)
        row += 1
        self._grid_pair(grid, row, "Offset X", self.star_offset_x, "Offset Y", self.star_offset_y)
        row += 1
        self._grid_pair(grid, row, "Scale X (%)", self.star_scale_x, "Scale Y (%)", self.star_scale_y)
        row += 1
        self._grid_pair(grid, row, "Rotation", self.star_angle, "Move Step", self.star_move_step)
        row += 1
        self._grid_pair(grid, row, "Scale Step (%)", self.star_scale_step, "Rotation Step", self.star_angle_step)
        row += 1
        for text, dx, dy in [("左移", -1, 0), ("右移", 1, 0), ("上移", 0, -1), ("下移", 0, 1)]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(lambda _=False, x=dx, y=dy: self._move_star(x, y))
            grid.addWidget(btn, row, len([c for c in range(4) if grid.itemAtPosition(row, c) is not None]))
        row += 1
        for col, (text, fn) in enumerate(
            [
                ("X-", lambda: self.star_scale_x.setValue(self.star_scale_x.value() - self.star_scale_step.value())),
                ("X+", lambda: self.star_scale_x.setValue(self.star_scale_x.value() + self.star_scale_step.value())),
                ("Y-", lambda: self.star_scale_y.setValue(self.star_scale_y.value() - self.star_scale_step.value())),
                ("Y+", lambda: self.star_scale_y.setValue(self.star_scale_y.value() + self.star_scale_step.value())),
            ]
        ):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(fn)
            grid.addWidget(btn, row, col)
        row += 1
        rot_minus = QtWidgets.QPushButton("角度-")
        rot_plus = QtWidgets.QPushButton("角度+")
        rot_minus.clicked.connect(lambda: self.star_angle.setValue(self.star_angle.value() - self.star_angle_step.value()))
        rot_plus.clicked.connect(lambda: self.star_angle.setValue(self.star_angle.value() + self.star_angle_step.value()))
        grid.addWidget(rot_minus, row, 2)
        grid.addWidget(rot_plus, row, 3)
        tabs.addTab(star, "颗星公式")

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        save_button = buttons.button(QtWidgets.QDialogButtonBox.Save)
        cancel_button = buttons.button(QtWidgets.QDialogButtonBox.Cancel)
        if save_button is not None:
            save_button.setText("保存")
        if cancel_button is not None:
            cancel_button.setText("取消")
        layout.addWidget(buttons)

    def _add_form_tab(self, tabs: QtWidgets.QTabWidget, title: str, rows: list[tuple[str, object]]) -> None:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        for label, field in rows:
            form.addRow(label, field)
        tabs.addTab(self._scroll_page(page), title)

    def _add_widget_tab(self, tabs: QtWidgets.QTabWidget, title: str, widgets: list[QtWidgets.QWidget]) -> None:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        for widget in widgets:
            page_layout.addWidget(widget)
        page_layout.addStretch(1)
        tabs.addTab(self._scroll_page(page), title)

    def _scroll_page(self, page: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(page)
        return scroll

    def apply_to_config(self, config: AppConfig) -> None:
        config.web_control.host = self.web_control_host.text().strip() or "0.0.0.0"
        config.web_control.port = int(self.web_control_port.value())
        config.detector.model_path = self.model_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.detector.class_file_path = self.class_file_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.training_detector.model_path = self.training_model_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.learning.ranker_enabled = self.learning_ranker_enabled.isChecked()
        config.learning.ranker_model_path = self.learning_ranker_model_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.learning.score_blend = float(self.learning_score_blend.value())
        config.learning.collect_enabled = self.learning_collect_enabled.isChecked()
        config.learning.samples_directory = self.learning_samples_directory.line_edit.text().strip() or "rl/data/samples"  # type: ignore[attr-defined]
        config.camera.video_path = self.video_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.camera.nori_sdk_root = self.nori_sdk_root.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.camera.frame_rotation_degrees = normalize_frame_rotation_degrees(
            self.frame_rotation_degrees.currentData()
        )
        config.camera.distortion_correction_enabled = self.distortion_enabled.isChecked()
        config.camera.distortion_correction_file = self.distortion_file.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.camera.exposure_control = normalize_exposure_control(self.exposure_control.currentData())
        config.camera.exposure_auto = self.exposure_auto.isChecked()
        config.camera.exposure_level = int(self.exposure_level.value())
        config.camera.white_balance_auto = self.white_balance_auto.isChecked()
        config.camera.white_balance_value = int(self.white_balance_value.value())
        config.detector.backend = self.detector_backend.currentText()
        config.training_detector.backend = self.training_detector_backend.currentText()
        config.state.engine = normalize_state_machine_engine(self.state_machine_engine.currentData())
        config.detector.detect_interval_frames = int(self.detect_interval.value())
        config.detector.detect_fps_limit_hz = float(self.detect_fps_limit.value())
        config.geometry.outline_path = self.outline_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.geometry.inline_path = self.inline_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.geometry.pocket_path = self.pocket_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.calibration.camera_file = self.camera_calibration_file.line_edit.text().strip() or None  # type: ignore[attr-defined]
        self._store_projection_path_for_current_mode()
        config.calibration.legacy_projection_file = self._legacy_projection_file
        config.calibration.engineered_plane_projection_file = self._engineered_projection_file
        config.calibration.engineered_ball_compensation_file = (
            self.engineered_ball_compensation_file.line_edit.text().strip() or None  # type: ignore[attr-defined]
        )
        config.calibration.set_projection_mode(str(self.projection_mode.currentData() or "legacy"))
        config.calibration.sync_projection_file_alias()
        config.projection.screen_index = int(self.proj_screen.currentData() or 0)
        config.projection.projector_width = int(self.proj_w.value())
        config.projection.projector_height = int(self.proj_h.value())
        config.projection.calibration_rotation_degrees = normalize_projection_rotation_degrees(
            self.projection_calibration_rotation.currentData()
        )
        config.projection.output_rotation_degrees = normalize_projection_rotation_degrees(
            self.projection_output_rotation.currentData()
        )
        config.projection.auto_pocket_animation_enabled = self.auto_pocket_animation_enabled.isChecked()
        config.projection.auto_victory_animation_enabled = self.auto_victory_animation_enabled.isChecked()
        config.projection.training_prompt_enabled = self.training_prompt_enabled.isChecked()
        config.projection.training_prompt_x_pct = float(self.training_prompt_x_pct.value())
        config.projection.training_prompt_y_pct = float(self.training_prompt_y_pct.value())
        config.projection.training_prompt_font_size_px = int(self.training_prompt_font_size.value())
        config.planner.route_freeze_enabled = self.route_freeze_enabled.isChecked()
        config.planner.route_freeze_enter_frames = int(self.route_freeze_enter_frames.value())
        config.planner.route_freeze_release_frames = int(self.route_freeze_release_frames.value())
        config.planner.route_freeze_same_route_refresh_mm = float(self.route_freeze_same_route_refresh_mm.value())
        config.planner.route_freeze_same_route_refresh_score_delta = float(self.route_freeze_same_route_refresh_score_delta.value())
        config.planner.route_freeze_switch_confirm_frames = int(self.route_freeze_switch_confirm_frames.value())
        config.planner.route_freeze_switch_min_distance_mm = float(self.route_freeze_switch_min_distance_mm.value())
        config.planner.route_freeze_switch_min_score_delta = float(self.route_freeze_switch_min_score_delta.value())
        config.planner.cue_sector_correction_enabled = self.cue_sector_enabled.isChecked()
        config.planner.cue_sector_corridor_width_px = float(self.cue_sector_corridor_width.value())
        config.planner.cue_sector_switch_confirm_frames = int(self.cue_sector_switch_confirm_frames.value())
        config.planner.cue_sector_switch_min_score_delta = float(self.cue_sector_switch_min_score_delta.value())
        config.planner.cue_sector_require_balls_stationary = self.cue_sector_require_balls_stationary.isChecked()
        config.planner.cue_sector_stationary_speed_mm_s = float(self.cue_sector_stationary_speed_mm.value())
        config.planner.cue_sector_stationary_speed_px_s = float(self.cue_sector_stationary_speed_px.value())
        config.planner.target_lock_enabled = self.target_lock_enabled.isChecked()
        config.planner.target_lock_confirm_frames = int(self.target_lock_confirm_frames.value())
        config.planner.target_lock_switch_confirm_frames = int(self.target_lock_switch_confirm_frames.value())
        config.planner.target_shot_enabled = self.target_shot_enabled.isChecked()
        config.planner.target_shot_trigger_frames = None
        config.planner.target_shot_activate_hold_ms = max(1, int(round(float(self.target_shot_activate_hold_seconds.value()) * 1000.0)))
        config.planner.target_shot_switch_hold_ms = max(1, int(round(float(self.target_shot_switch_hold_seconds.value()) * 1000.0)))
        config.planner.target_shot_miss_grace_ms = max(0, int(round(float(self.target_shot_miss_grace_seconds.value()) * 1000.0)))
        config.planner.target_shot_release_confirm_ms = max(0, int(round(float(self.target_shot_release_confirm_seconds.value()) * 1000.0)))
        config.normalize_compat_settings()
        self._apply_projection_tuning_to_config(config)

    def _apply_projection_tuning_to_config(self, config: AppConfig) -> None:
        config.calibration.projection_visible_inset_top_mm = float(self.visible_inset_top.value())
        config.calibration.projection_visible_inset_right_mm = float(self.visible_inset_right.value())
        config.calibration.projection_visible_inset_bottom_mm = float(self.visible_inset_bottom.value())
        config.calibration.projection_visible_inset_left_mm = float(self.visible_inset_left.value())
        config.calibration.physical_rail_inset_top_mm = float(self.physical_inset_top.value())
        config.calibration.physical_rail_inset_right_mm = float(self.physical_inset_right.value())
        config.calibration.physical_rail_inset_bottom_mm = float(self.physical_inset_bottom.value())
        config.calibration.physical_rail_inset_left_mm = float(self.physical_inset_left.value())
        config.calibration.physical_middle_pocket_relief_top_mm = float(self.middle_relief_top.value())
        config.calibration.physical_middle_pocket_relief_bottom_mm = float(self.middle_relief_bottom.value())
        config.calibration.center_reachable_extra_margin_mm = float(self.center_margin.value())
        config.calibration.ball_center_compensation_enabled = self.ball_comp_enabled.isChecked()
        config.calibration.ball_center_compensation_auto_reference = self.ball_comp_auto_ref.isChecked()
        config.calibration.ball_center_compensation_ref_x_px = float(self.ball_comp_ref_x.value())
        config.calibration.ball_center_compensation_ref_y_px = float(self.ball_comp_ref_y.value())
        config.calibration.ball_center_compensation_scale_x_pct = float(self.ball_comp_scale_x.value())
        config.calibration.ball_center_compensation_scale_y_pct = float(self.ball_comp_scale_y.value())

    def _sync_ball_reference_inputs(self, *_args) -> None:
        manual = not self.ball_comp_auto_ref.isChecked()
        self.ball_comp_ref_x.setEnabled(manual)
        self.ball_comp_ref_y.setEnabled(manual)

    def _sync_training_prompt_inputs(self, *_args) -> None:
        enabled = self.training_prompt_enabled.isChecked()
        self.training_prompt_x_pct.setEnabled(enabled)
        self.training_prompt_y_pct.setEnabled(enabled)
        self.training_prompt_font_size.setEnabled(enabled)

    def _active_projection_path_for_mode(self, mode: str) -> Optional[str]:
        normalized = "engineered" if str(mode).strip().lower() == "engineered" else "legacy"
        if normalized == "engineered":
            return self._engineered_projection_file or self._legacy_projection_file
        return self._legacy_projection_file or self._engineered_projection_file

    def _store_projection_path_for_current_mode(self) -> None:
        value = self.projection_calibration_file.line_edit.text().strip() or None  # type: ignore[attr-defined]
        if self._projection_mode_for_path == "engineered":
            self._engineered_projection_file = value
        else:
            self._legacy_projection_file = value

    def _projection_mode_changed(self, _index: int = 0) -> None:
        self._store_projection_path_for_current_mode()
        self._projection_mode_for_path = str(self.projection_mode.currentData() or "legacy")
        next_path = self._active_projection_path_for_mode(self._projection_mode_for_path) or ""
        self.projection_calibration_file.line_edit.setText(next_path)  # type: ignore[attr-defined]

    def _bind_projection_tuning_preview(self) -> None:
        widgets = [
            self.visible_inset_top,
            self.visible_inset_right,
            self.visible_inset_bottom,
            self.visible_inset_left,
            self.physical_inset_top,
            self.physical_inset_right,
            self.physical_inset_bottom,
            self.physical_inset_left,
            self.middle_relief_top,
            self.middle_relief_bottom,
            self.center_margin,
            self.ball_comp_ref_x,
            self.ball_comp_ref_y,
            self.ball_comp_scale_x,
            self.ball_comp_scale_y,
        ]
        for widget in widgets:
            widget.valueChanged.connect(self._preview_projection_tuning)
        self.ball_comp_enabled.toggled.connect(self._preview_projection_tuning)
        self.ball_comp_auto_ref.toggled.connect(self._preview_projection_tuning)

    def _preview_projection_tuning(self, *_args) -> None:
        self._apply_projection_tuning_to_config(self.config)
        self._preview_active = True
        if self._operator is None:
            return
        try:
            self._operator._apply_projection_tuning_preview()
        except Exception as exc:
            self._operator._append_log(f"投影修正预览失败: {exc}")

    def _restore_projection_tuning_preview(self) -> None:
        self.config.calibration = copy.deepcopy(self._original_calibration)
        if self._operator is not None:
            self._operator._apply_projection_tuning_preview()
        self._preview_active = False

    def star_formula_config(self) -> StarFormulaConfig:
        return StarFormulaConfig(
            enabled=self.star_enabled.isChecked(),
            inset_left_pct=self.star_left.value(),
            inset_right_pct=self.star_right.value(),
            inset_top_pct=self.star_top.value(),
            inset_bottom_pct=self.star_bottom.value(),
            label_offset_tb_pct=self.star_label_tb.value(),
            label_offset_lr_pct=self.star_label_lr.value(),
            offset_x_px=float(self.star_offset_x.value()),
            offset_y_px=float(self.star_offset_y.value()),
            scale_x_pct=self.star_scale_x.value(),
            scale_y_pct=self.star_scale_y.value(),
            angle_deg=self.star_angle.value(),
        ).clamp_for_projection(int(self.proj_w.value()), int(self.proj_h.value()))

    def accept(self) -> None:
        self._preview_active = False
        super().accept()

    def reject(self) -> None:
        if self._preview_active:
            self._restore_projection_tuning_preview()
        super().reject()

    def _path_row(self, value: Optional[str], file_filter: str) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit(value or "")
        btn = QtWidgets.QPushButton("浏览")
        btn.clicked.connect(lambda: self._browse_file(edit, file_filter))
        layout.addWidget(edit, 1)
        layout.addWidget(btn)
        box.line_edit = edit  # type: ignore[attr-defined]
        return box

    def _dir_row(self, value: Optional[str]) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit(value or "")
        btn = QtWidgets.QPushButton("浏览")
        btn.clicked.connect(lambda: self._browse_dir(edit))
        layout.addWidget(edit, 1)
        layout.addWidget(btn)
        box.line_edit = edit  # type: ignore[attr-defined]
        return box

    def _browse_file(self, edit: QtWidgets.QLineEdit, file_filter: str) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择文件", edit.text() or str(Path.cwd()), file_filter)
        if path:
            edit.setText(path)

    def _browse_dir(self, edit: QtWidgets.QLineEdit) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择目录", edit.text() or str(Path.cwd()))
        if path:
            edit.setText(path)

    def _dspin(self, value: float, lo: float, hi: float, step: float, *, decimals: int = 1) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(int(max(0, decimals)))
        spin.setValue(float(value))
        return spin

    def _spin(self, value: int, lo: int, hi: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(int(value))
        return spin

    def _white_balance_control_spec(self, config: AppConfig) -> tuple[int, int, int, int]:
        fallback_min = 1000
        fallback_max = 15000
        fallback_step = 100
        fallback_value = int(config.camera.white_balance_value if config.camera.white_balance_value is not None else 4600)
        if str(config.camera.backend or "auto").lower() not in {"auto", "nori"}:
            return fallback_min, fallback_max, fallback_step, max(fallback_min, min(fallback_max, fallback_value))
        controller: Optional[NoriProtocolController] = None
        try:
            controller = NoriProtocolController(sdk_root=config.camera.nori_sdk_root)
            device_id = int(config.camera.nori_device_id) if config.camera.nori_device_id is not None else int(config.camera.device_index)
            value, _, step, min_v, max_v, def_v, _ = controller.get_white_balance_control(device_id)
            min_v = int(min_v) if int(min_v) < int(max_v) else fallback_min
            max_v = int(max_v) if int(max_v) > int(min_v) else fallback_max
            step = max(1, int(step))
            desired = fallback_value if config.camera.white_balance_value is not None else int(value or def_v or fallback_value)
            desired = max(min_v, min(max_v, desired))
            if step > 1:
                desired = min_v + round((desired - min_v) / step) * step
                desired = max(min_v, min(max_v, desired))
            return min_v, max_v, step, desired
        except Exception:
            return fallback_min, fallback_max, fallback_step, max(fallback_min, min(fallback_max, fallback_value))
        finally:
            if controller is not None:
                controller.shutdown()

    def _update_white_balance_value_label(self, value: int) -> None:
        self.white_balance_value_label.setText(f"{int(value)} K")

    def _grid_pair(self, grid: QtWidgets.QGridLayout, row: int, label_a: str, widget_a: QtWidgets.QWidget, label_b: str, widget_b: QtWidgets.QWidget) -> None:
        grid.addWidget(QtWidgets.QLabel(label_a), row, 0)
        grid.addWidget(widget_a, row, 1)
        grid.addWidget(QtWidgets.QLabel(label_b), row, 2)
        grid.addWidget(widget_b, row, 3)

    def _move_star(self, dx: int, dy: int) -> None:
        step = int(self.star_move_step.value())
        self.star_offset_x.setValue(self.star_offset_x.value() + dx * step)
        self.star_offset_y.setValue(self.star_offset_y.value() + dy * step)


class JointCalibrationWizardDialog(QtWidgets.QDialog):
    def __init__(self, operator: "OperatorWindow", parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent or operator)
        self.operator = operator
        self._busy = False
        self._result = None
        self._saved_path: Optional[Path] = None
        self.setWindowTitle("一键联动校正")
        self.resize(920, 760)

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "该流程会先基于工业相机畸变校正采集画面，再自动播放总览网格、全域 ChArUco、中心细化和六个袋口重点图样，"
            "完成投影仪-相机-程序的一键联动校正。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.preview = QtWidgets.QLabel("等待采集画面")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(280)
        self.preview.setStyleSheet("background:#111; border:1px solid #333;")
        layout.addWidget(self.preview)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.summary = QtWidgets.QLabel("尚未开始联动校正。")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box, 1)

        button_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("开始一键联动校正")
        self.start_btn.clicked.connect(self.run_calibration)
        button_row.addWidget(self.start_btn)
        self.show_btn = QtWidgets.QPushButton("投影显示结果")
        self.show_btn.setEnabled(False)
        self.show_btn.clicked.connect(self.show_result)
        button_row.addWidget(self.show_btn)
        button_row.addStretch(1)
        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

    def _append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{stamp}] {text}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        self.operator._append_log(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.start_btn.setEnabled(not busy)
        self.show_btn.setEnabled((self._result is not None) and (not busy))

    def _pump_ui(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()

    def _set_preview_image(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None or frame_bgr.size == 0:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        self.preview.setPixmap(pix.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def _projection_output_path(self) -> Path:
        return timestamped_projection_output_path(self.operator.config.calibration.active_projection_file())

    @QtCore.pyqtSlot()
    def run_calibration(self) -> None:
        if self._busy:
            return
        capture = None
        self._set_busy(True)
        self.progress.setValue(0)
        self._append_log("联动校正开始，准备同步当前配置。")
        try:
            self.operator._sync_config_from_controls()
            if (
                self.operator.config.camera.distortion_correction_enabled
                and not self.operator.config.calibration.camera_file
                and self.operator.config.camera.distortion_correction_file
            ):
                self.operator.config.calibration.camera_file = self.operator.config.camera.distortion_correction_file
                self._append_log("已将工业相机畸变标定文件同步为联动校正的相机标定输入。")
            if not self.operator.config.camera.distortion_correction_enabled:
                self._append_log("工业相机畸变校正未开启；联动校正将直接使用当前原始画面与原始像素坐标。")
            self.operator._save_user_settings()
            if self.operator.pipeline is not None:
                self._append_log("检测到实时采集正在运行，先自动停止以释放相机。")
                self.operator.stop_pipeline()

            geometry = TableGeometryLoader.load_optional(
                self.operator.config.geometry.outline_path,
                self.operator.config.geometry.inline_path,
                self.operator.config.geometry.pocket_path,
            )
            capture = create_capture_service(self.operator.config.camera)
            capture_info = capture.info()
            actual_frame_size = (int(capture_info.width), int(capture_info.height))
            calibration = _create_calibration_workflow_service(
                self.operator.config,
                actual_frame_size=actual_frame_size,
            )
            patterns = build_linked_patterns(
                geometry,
                (int(self.operator.config.projection.projector_width), int(self.operator.config.projection.projector_height)),
                prior_projection=calibration.projection if calibration.projection.is_valid else None,
            )
            self.operator.ensure_projection_window_for_operator()
            self._append_log(f"已生成 {len(patterns)} 个联动图样，开始自动采集与识别。")

            observations = []
            first_frame_shape: Optional[tuple[int, int]] = None
            undistort = None
            if calibration.camera.is_valid and not capture.frame_distortion_corrected:
                undistort = calibration.camera.undistort_points
            total = max(1, len(patterns))
            for idx, pattern in enumerate(patterns):
                percent = int(round(idx / total * 100.0))
                self.progress.setValue(percent)
                self.summary.setText(f"正在处理图样 {idx + 1}/{len(patterns)}: {pattern.title}")
                if self.operator.projection_window is not None:
                    self.operator.projection_window.set_image(pattern.image)
                self._append_log(f"切换图样: {pattern.title}")
                self._pump_ui()
                time.sleep(0.25)

                best = None
                for _ in range(6):
                    packet = capture.read()
                    if packet is None or packet.image is None:
                        continue
                    if first_frame_shape is None:
                        first_frame_shape = packet.image.shape[:2]
                    self._set_preview_image(packet.image)
                    self._pump_ui()
                    observation = match_linked_pattern_observation(pattern, packet.image, undistort_points=undistort)
                    if observation is not None and (best is None or observation.matched_count > best.matched_count):
                        best = observation
                    time.sleep(0.05)

                if not pattern.collect_for_solver:
                    continue
                if best is None:
                    self._append_log(f"{pattern.title} 未检测到足够的 ChArUco 角点，跳过该图样。")
                    continue
                observations.append(best)
                self._append_log(f"{pattern.title} 匹配角点 {best.matched_count} 个，重点区域 {best.emphasis_zone}。")

            if first_frame_shape is None:
                raise RuntimeError("未能从相机读取任何画面，联动校正中止。")
            table_polygon_cam = None
            if not geometry.is_empty:
                frame_h, frame_w = first_frame_shape
                outer_px, _, _ = geometry.scaled(frame_w, frame_h)
                if outer_px.shape[0] >= 3:
                    table_polygon_cam = outer_px.astype(np.float32)
                    if undistort is not None:
                        table_polygon_cam = undistort(table_polygon_cam)
            self._result = solve_linked_projection_calibration(
                observations,
                (int(self.operator.config.projection.projector_width), int(self.operator.config.projection.projector_height)),
                table_polygon_cam=table_polygon_cam,
            )
            frame_h, frame_w = first_frame_shape
            coordinate_domain = (
                "undistorted"
                if calibration.distortion_correction_enabled and calibration.camera.is_valid
                else "raw"
            )
            self._result.projection.calibration_context = projection_calibration_context(
                frame_width=int(frame_w),
                frame_height=int(frame_h),
                frame_rotation_degrees=int(self.operator.config.camera.frame_rotation_degrees),
                camera_coordinate_domain=coordinate_domain,
                distortion_file=(
                    self.operator.config.camera.distortion_correction_file
                    if coordinate_domain == "undistorted"
                    else None
                ),
                projector_width=int(self.operator.config.projection.projector_width),
                projector_height=int(self.operator.config.projection.projector_height),
                projection_calibration_rotation_degrees=int(
                    self.operator.config.projection.calibration_rotation_degrees
                ),
            )
            self._saved_path = self._projection_output_path()
            self._result.projection.save(self._saved_path)
            calibration = self._load_projection_from_path(
                self._saved_path,
                actual_frame_size=(int(frame_w), int(frame_h)),
            )
            self.progress.setValue(100)
            self.summary.setText(
                "联动校正完成。\n"
                f"{projection_output_summary(self._result)}\n"
                f"保存路径: {self._saved_path}"
            )
            self._append_log(f"联动校正完成，结果已保存并加载: {self._saved_path}")
            self.show_result(calibration)
        except Exception as exc:
            self.summary.setText(f"联动校正失败: {exc}")
            self._append_log(f"联动校正失败: {exc}")
            QtWidgets.QMessageBox.critical(self, "联动校正失败", str(exc))
        finally:
            if capture is not None:
                capture.release()
            self._set_busy(False)

    @QtCore.pyqtSlot()
    def show_result(self, calibration=None) -> None:
        calibration = calibration or _create_calibration_workflow_service(
            self.operator.config,
            self.operator.pipeline,
        )
        self.operator.show_projector_calibration_result(calibration)
        self.operator.show_projector_residual_overlay(calibration)
        self._append_log("已在投影窗口显示联动校正结果与残差箭头。")

    def _load_projection_from_path(
        self,
        path: Path,
        *,
        actual_frame_size: Optional[tuple[int, int]] = None,
    ):
        self.operator.config.calibration.set_active_projection_file(str(path))
        self.operator.config.calibration.sync_projection_file_alias()
        self.operator._sync_controls_from_config()
        self.operator._save_user_settings()
        return _create_calibration_workflow_service(
            self.operator.config,
            self.operator.pipeline,
            actual_frame_size=actual_frame_size,
        )


class OperatorWindow(WebControlOperatorMixin, QtWidgets.QMainWindow):
    BASE_WIDTH = 1420
    BASE_HEIGHT = 860
    RAW_PHOTO_LABEL = "抓拍原图"
    RAW_PHOTO_TOOLTIP = "抓取当前相机原始画面，不含画线且不做额外畸变校正"
    INSTANT_REPLAY_LABEL = "导出纯净回放"
    INSTANT_REPLAY_TOOLTIP = "导出前 60 秒纯净视频回放"
    RAW_VIDEO_START_LABEL = "开始原始录制"
    RAW_VIDEO_STOP_LABEL = "停止原始录制"
    RAW_VIDEO_TOOLTIP = "录制无画线视频，画面统一遵循工业相机畸变校正开关"
    ROUTE_VIDEO_START_LABEL = "开始路线录制"
    ROUTE_VIDEO_STOP_LABEL = "停止路线录制"
    ROUTE_VIDEO_TOOLTIP = "录制进洞路线画线视频，底图统一遵循工业相机畸变校正开关"

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.user_settings = UserSettings.load()
        self.star_formula = StarFormulaConfig.from_mapping(self.user_settings.star_formula)
        self.pipeline: Optional[RuntimePipeline] = None
        self._pipeline_start_pending = False
        self.projection_window: Optional[ProjectionWindow] = None
        self.last_output: Optional[PipelineOutput] = None
        self._last_preview_bgr: Optional[np.ndarray] = None
        self._projection_calibration_mode = False
        self._ui_scale = 1.0
        self.frame_count = 0
        self.started_at = 0.0
        self._last_preview_update_ts = 0.0
        self._preview_fps_limit = 15.0
        self._last_metrics_update_ts = 0.0
        self._metrics_fps_limit = 5.0
        self._last_heavy_ui_update_ts = 0.0
        self._heavy_ui_fps_limit = 3.0
        self._last_perf_log_ts = 0.0
        self._video_timeline_state: Optional[VideoTimelineState] = None
        self._video_timeline_dragging = False
        self._video_timeline_internal_update = False
        self._video_timeline_last_seek_value: Optional[int] = None
        self._video_playback_paused = False
        self._raw_video_recorder: Optional[FfmpegH264Recorder] = None
        self._route_video_recorder: Optional[FfmpegH264Recorder] = None
        self._raw_video_path: Optional[Path] = None
        self._route_video_path: Optional[Path] = None
        self._deep_debug_session: Optional[StateDebugSession] = None
        self._deep_debug_started_raw_video = False
        self._deep_debug_started_route_video = False
        self._deep_debug_last_result: Optional[StateDebugSessionResult] = None
        self._instant_replay = InstantReplayBuffer(self.config.instant_replay)
        self._instant_replay_start_failed = False
        self._projection_debug_enabled = False
        self._cue_sector_preview_enabled = False
        self._projection_interaction = ProjectionInteractionController(
            auto_pocket_animation_enabled=bool(self.config.projection.auto_pocket_animation_enabled),
            auto_victory_animation_enabled=bool(self.config.projection.auto_victory_animation_enabled),
        )
        self.control_state = RuntimeControlState()
        self._route_freeze = MotionRouteFreezeController(self.config.planner)
        self._recording_fps_estimator = RecordingFpsEstimator()
        self._pending_turn_target_group: Optional[str] = None
        self._remote_command_queue = RemoteCommandQueue()
        self.web_control = WebControlServer()
        self._manual_web_target_id: Optional[int] = None
        self._web_pocket_notice_tracker = PocketNoticeTracker()

        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self._frame_busy = False
        self._remote_timer = QtCore.QTimer(self)
        self._remote_timer.setInterval(180)
        self._remote_timer.timeout.connect(self._process_remote_commands)
        self._remote_timer.start()
        self._projection_effect_timer = QtCore.QTimer(self)
        self._projection_effect_timer.setInterval(50)
        self._projection_effect_timer.timeout.connect(self._advance_projection_effects)

        self.setWindowTitle("BAS Control Console")
        self.resize(self.BASE_WIDTH, self.BASE_HEIGHT)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self._sync_controls_from_config()
        self._apply_scale(force=True)
        self._set_running(False)
        self._update_module_status()
        self._append_log("就绪")

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        self.outer_layout = QtWidgets.QVBoxLayout(root)
        self._compact_grids: list[CompactButtonGrid] = []
        self._form_layouts: list[QtWidgets.QFormLayout] = []
        self._section_content_layouts: list[QtWidgets.QVBoxLayout] = []
        self._field_layouts: list[QtWidgets.QVBoxLayout] = []

        self._build_header()

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.outer_layout.addWidget(self.main_splitter, 1)

        self._build_left_sidebar()
        self._build_center_panel()
        self._build_right_sidebar()

        self.main_splitter.addWidget(self.left_scroll_area)
        self.main_splitter.addWidget(self.center_panel)
        self.main_splitter.addWidget(self.right_scroll_area)

    def _build_header(self) -> None:
        self.header_widget = QtWidgets.QWidget()
        self.header_layout = QtWidgets.QVBoxLayout(self.header_widget)
        self.header_top_row = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        self.title = QtWidgets.QLabel("BAS 台球智能辅助")
        self.title.setObjectName("title")
        self.subtitle = QtWidgets.QLabel("实时视觉 / 状态估计 / 路线投影")
        self.subtitle.setObjectName("muted")
        title_box.addWidget(self.title)
        title_box.addWidget(self.subtitle)
        self.header_top_row.addLayout(title_box)
        self.header_top_row.addStretch(1)
        self.status_label = QtWidgets.QLabel("离线")
        self.status_label.setObjectName("status")
        self.header_top_row.addWidget(self.status_label)
        self.header_layout.addLayout(self.header_top_row)

        self.header_metrics_layout = QtWidgets.QHBoxLayout()
        self.fps_metric = self._metric("FPS", "--")
        self.det_metric = self._metric("检测", "--")
        self.track_metric = self._metric("跟踪", "--")
        self.phase_metric = self._metric("状态", "--")
        for item in [self.fps_metric, self.det_metric, self.track_metric, self.phase_metric]:
            self.header_metrics_layout.addWidget(item)
        self.header_metrics_layout.addStretch(1)
        self.header_layout.addLayout(self.header_metrics_layout)
        self.outer_layout.addWidget(self.header_widget)

    def _build_left_sidebar(self) -> None:
        self.left_scroll_area = self._scroll_area()
        self.sidebar = self._panel()
        self.side_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.left_scroll_area.setWidget(self.sidebar)

        control_section = self._section("运行控制")
        control_layout = control_section.contentLayout()
        self.capture_btn = self._button("开始采集", variant="primary")
        self.projection_btn = self._button("开始投影", variant="primary")
        self.init_module_btn = self._button("初始化图形图像模块", variant="compact")
        self.projector_calib_btn = self._button("校准工作台", variant="compact")
        self.settings_btn = self._button("设置", variant="compact")
        self.probe_btn = self._button("探测相机", variant="compact")
        self.export_diag_btn = self._button("导出诊断快照", variant="compact")
        self.web_control_btn = self._button("开启 Web 控制", variant="compact")
        self.capture_btn.clicked.connect(self.toggle_capture)
        self.projection_btn.clicked.connect(self.toggle_projection_window)
        self.init_module_btn.clicked.connect(self.initialize_graphics_image_module)
        self.projector_calib_btn.clicked.connect(self.open_calibration_workbench)
        self.settings_btn.clicked.connect(self.open_settings)
        self.probe_btn.clicked.connect(self.probe_camera_devices)
        self.export_diag_btn.clicked.connect(self.export_diagnostic_snapshot)
        self.web_control_btn.clicked.connect(self.toggle_web_control)
        control_layout.addWidget(self.capture_btn)
        control_layout.addWidget(self.projection_btn)
        control_grid = CompactButtonGrid(columns=2)
        self._compact_grids.append(control_grid)
        for button in [self.init_module_btn, self.projector_calib_btn, self.settings_btn, self.probe_btn]:
            control_grid.addButton(button)
        control_grid.addButton(self.export_diag_btn, column_span=2)
        control_grid.addButton(self.web_control_btn, column_span=2)
        control_layout.addWidget(control_grid)
        self.side_layout.addWidget(control_section)

        capture_settings_section = self._section("采集设置")
        capture_settings_layout = capture_settings_section.contentLayout()
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(["auto", "nori", "opencv", "video", "synthetic"])
        self.backend_combo.currentTextChanged.connect(self._update_video_timeline_visibility)
        self.nori_device_combo = QtWidgets.QComboBox()
        self.nori_device_combo.addItem("默认", None)
        self.device_spin = QtWidgets.QSpinBox()
        self.device_spin.setRange(0, 32)
        self.resolution_combo = QtWidgets.QComboBox()
        self.resolution_combo.addItems(COMMON_RESOLUTIONS)
        self.resolution_combo.setEditable(True)
        self.fps_combo = QtWidgets.QComboBox()
        self.fps_combo.addItems(COMMON_FPS)
        self.fps_combo.setEditable(True)
        capture_form_box, capture_form = self._form_panel()
        self._add_form_row(capture_form, "输入类型", self.backend_combo)
        self._add_form_row(capture_form, "工业相机", self.nori_device_combo)
        self._add_form_row(capture_form, "OpenCV 设备", self.device_spin)
        self._add_form_row(capture_form, "分辨率", self.resolution_combo)
        self._add_form_row(capture_form, "帧率", self.fps_combo)
        capture_settings_layout.addWidget(capture_form_box)
        self.side_layout.addWidget(capture_settings_section)

        display_section = self._section("显示与模式")
        display_layout = display_section.contentLayout()
        self.replay_check = QtWidgets.QCheckBox("记录回放")
        self.replay_check.setChecked(self.config.replay.enabled)
        self.geometry_reference_check = QtWidgets.QCheckBox("显示几何参考线")
        self.geometry_reference_check.setChecked(bool(self.config.projection.geometry_reference_enabled))
        self.geometry_reference_check.toggled.connect(self._geometry_reference_toggled)
        self.projection_debug_check = QtWidgets.QCheckBox("投影调试模式")
        self.projection_debug_check.setChecked(False)
        self.projection_debug_check.toggled.connect(self._projection_debug_toggled)
        self.cue_sector_preview_check = QtWidgets.QCheckBox()
        self.cue_sector_preview_check.setChecked(False)
        self.cue_sector_preview_check.setText("\u663e\u793a\u7403\u6746\u77e9\u5f62\u5019\u9009\u6846")
        self.cue_sector_preview_check.toggled.connect(self._cue_sector_preview_toggled)
        for widget in [self.replay_check, self.geometry_reference_check, self.projection_debug_check, self.cue_sector_preview_check]:
            display_layout.addWidget(widget)
        self.runtime_mode_combo = QtWidgets.QComboBox()
        self.runtime_mode_combo.addItem("规则比赛", RULES_MODE)
        self.runtime_mode_combo.addItem("编号球训练", TRAINING_MODE)
        self.runtime_mode_combo.currentIndexChanged.connect(self._operating_mode_changed)
        self.shot_mode_combo = QtWidgets.QComboBox()
        self.shot_mode_combo.addItem("规则模式", "rule")
        self.shot_mode_combo.addItem("勾球模式", "hook")
        self.shot_mode_combo.currentIndexChanged.connect(self._shot_mode_changed)
        self.state_machine_combo = QtWidgets.QComboBox()
        self.state_machine_combo.addItem("状态机（旧）", "legacy")
        self.state_machine_combo.addItem("状态机（新）", "modern")
        self.state_machine_combo.currentIndexChanged.connect(self._state_machine_changed)
        display_form_box, display_form = self._form_panel()
        self._add_form_row(display_form, "运行模式", self.runtime_mode_combo)
        self._add_form_row(display_form, "画线模式", self.shot_mode_combo)
        self._add_form_row(display_form, "状态机", self.state_machine_combo)
        display_layout.addWidget(display_form_box)
        self.side_layout.addWidget(display_section)

        self.training_section = self._section("编号球训练")
        training_layout = self.training_section.contentLayout()
        self.training_scenario_combo = QtWidgets.QComboBox()
        current_group = None
        for scenario in list_training_scenarios():
            if scenario.group_title != current_group:
                current_group = scenario.group_title
                self.training_scenario_combo.addItem(f"── {current_group} ──", None)
                header_item = self.training_scenario_combo.model().item(self.training_scenario_combo.count() - 1)
                if header_item is not None:
                    header_item.setEnabled(False)
            self.training_scenario_combo.addItem(scenario.display_title, scenario.scenario_id)
        self.training_scenario_combo.currentIndexChanged.connect(self._training_scenario_changed)
        self.training_instruction_label = QtWidgets.QLabel()
        self.training_instruction_label.setObjectName("muted")
        self.training_instruction_label.setWordWrap(True)
        self.training_status_label = QtWidgets.QLabel("切换到编号球训练模式后开始")
        self.training_status_label.setWordWrap(True)
        self.training_start_btn = self._button("开始验证", variant="primary")
        self.training_reset_btn = self._button("重置本次", variant="compact")
        self.training_start_btn.clicked.connect(self.start_training_validation)
        self.training_reset_btn.clicked.connect(self.reset_training_validation)
        training_form_box, training_form = self._form_panel()
        self._add_form_row(training_form, "训练项目", self.training_scenario_combo)
        training_layout.addWidget(training_form_box)
        training_layout.addWidget(self.training_instruction_label)
        training_layout.addWidget(self.training_status_label)
        training_buttons = CompactButtonGrid(columns=2)
        self._compact_grids.append(training_buttons)
        training_buttons.addButton(self.training_start_btn)
        training_buttons.addButton(self.training_reset_btn)
        training_layout.addWidget(training_buttons)
        self.side_layout.addWidget(self.training_section)

        capture_actions_section = self._section("现场抓取")
        capture_actions_layout = capture_actions_section.contentLayout()
        self.raw_photo_btn = self._button(self.RAW_PHOTO_LABEL, variant="compact")
        self.instant_replay_export_btn = self._button(self.INSTANT_REPLAY_LABEL, variant="compact")
        self.raw_video_btn = self._button(self.RAW_VIDEO_START_LABEL, variant="compact")
        self.route_video_btn = self._button(self.ROUTE_VIDEO_START_LABEL, variant="compact")
        self._configure_capture_action_buttons()
        self.raw_photo_btn.clicked.connect(self.capture_raw_photo)
        self.instant_replay_export_btn.clicked.connect(self.trigger_instant_replay_export)
        self.raw_video_btn.clicked.connect(self.toggle_raw_video_recording)
        self.route_video_btn.clicked.connect(self.toggle_route_video_recording)
        capture_action_grid = CompactButtonGrid(columns=2)
        self._compact_grids.append(capture_action_grid)
        for button in [self.raw_photo_btn, self.instant_replay_export_btn, self.raw_video_btn, self.route_video_btn]:
            capture_action_grid.addButton(button)
        capture_actions_layout.addWidget(capture_action_grid)
        self.side_layout.addWidget(capture_actions_section)

        self.interaction_section = self._section("交互调试", expanded=False)
        interaction_layout = self.interaction_section.contentLayout()
        interaction_grid = CompactButtonGrid(columns=3)
        self._compact_grids.append(interaction_grid)
        self.interaction_test_buttons: dict[str, QtWidgets.QPushButton] = {}
        pocket_layout = (
            ("pocket0", "P0", 0, 0),
            ("pocket1", "P1", 0, 1),
            ("pocket2", "P2", 0, 2),
            ("pocket5", "P5", 2, 0),
            ("pocket4", "P4", 2, 1),
            ("pocket3", "P3", 2, 2),
        )
        for key, text, row, col in pocket_layout:
            btn = self._button(text, variant="compact")
            pocket_index = int(key.replace("pocket", ""))
            btn.clicked.connect(lambda _checked=False, idx=pocket_index: self._trigger_interaction_test_pocket(idx))
            interaction_grid.addButton(btn, row=row, column=col)
            self.interaction_test_buttons[key] = btn
        victory_btn = self._button("结算", variant="compact")
        victory_btn.clicked.connect(self._trigger_interaction_test_victory)
        interaction_grid.addButton(victory_btn, row=1, column=1)
        self.interaction_test_buttons["victory"] = victory_btn
        interaction_layout.addWidget(interaction_grid)
        self.side_layout.addWidget(self.interaction_section)

        self.side_layout.addStretch(1)
        self.config_label = QtWidgets.QLabel("设置保存在 local_settings/user_settings.json")
        self.config_label.setObjectName("muted")
        self.config_label.setWordWrap(True)
        self.side_layout.addWidget(self.config_label)

    def _build_center_panel(self) -> None:
        self.center_panel = QtWidgets.QWidget()
        self.center_layout = QtWidgets.QVBoxLayout(self.center_panel)

        self.preview_panel = self._panel()
        self.preview_panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.preview_layout = QtWidgets.QVBoxLayout(self.preview_panel)
        self.preview_header_layout = QtWidgets.QHBoxLayout()
        self.preview_header_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_caption = QtWidgets.QLabel("实时预览")
        self.preview_caption.setObjectName("previewCaption")
        self.preview_caption.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.preview_caption.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.preview_header_layout.addWidget(self.preview_caption)
        self.preview_header_layout.addStretch(1)
        self.preview_zoom_out_btn = self._button("缩小", variant="preview")
        self.preview_fit_btn = self._button("适应", variant="preview")
        self.preview_zoom_in_btn = self._button("放大", variant="preview")
        self.preview_zoom_label = QtWidgets.QLabel("100%")
        self.preview_zoom_label.setObjectName("previewZoomLabel")
        self.preview_zoom_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_zoom_out_btn.setToolTip("缩小播放画面")
        self.preview_fit_btn.setToolTip("恢复为当前预览区域内不裁切的最大画面")
        self.preview_zoom_in_btn.setToolTip("放大播放画面")
        self.preview_header_layout.addWidget(self.preview_zoom_out_btn)
        self.preview_header_layout.addWidget(self.preview_fit_btn)
        self.preview_header_layout.addWidget(self.preview_zoom_in_btn)
        self.preview_header_layout.addWidget(self.preview_zoom_label)
        self.preview_layout.addLayout(self.preview_header_layout)
        self.preview_frame = AspectRatioPreviewFrame()
        self.preview_label = self.preview_frame.label()
        self.preview_zoom_out_btn.clicked.connect(self.preview_frame.zoomOut)
        self.preview_fit_btn.clicked.connect(self.preview_frame.fitToViewport)
        self.preview_zoom_in_btn.clicked.connect(self.preview_frame.zoomIn)
        self.preview_frame.zoomChanged.connect(self._update_preview_zoom_controls)
        self._update_preview_zoom_controls(self.preview_frame.zoomFactor())
        self.preview_frame.viewportChanged.connect(self._refresh_preview_pixmap)
        self.preview_frame.viewportChanged.connect(self._position_desktop_pocket_notice)
        self.pocket_notice_label = QtWidgets.QLabel(self.preview_frame)
        self.pocket_notice_label.setObjectName("pocketNotice")
        self.pocket_notice_label.setAlignment(QtCore.Qt.AlignCenter)
        self.pocket_notice_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.pocket_notice_label.setStyleSheet(
            "QLabel#pocketNotice {"
            "background: rgba(8, 18, 14, 235);"
            "border: 3px solid #5cde97;"
            "border-radius: 14px;"
            "color: #f5fff9;"
            "font-size: 30px;"
            "font-weight: 800;"
            "padding: 14px 26px;"
            "}"
        )
        self.pocket_notice_label.hide()
        self._desktop_pocket_notice_timer = QtCore.QTimer(self)
        self._desktop_pocket_notice_timer.setSingleShot(True)
        self._desktop_pocket_notice_timer.timeout.connect(self.pocket_notice_label.hide)
        self.preview_layout.addWidget(self.preview_frame, 1)
        self.video_timeline_widget = QtWidgets.QWidget()
        self.video_timeline_widget.setObjectName("videoTimeline")
        self.video_timeline_layout = QtWidgets.QHBoxLayout(self.video_timeline_widget)
        self.video_timeline_layout.setContentsMargins(0, 0, 0, 0)
        self.video_pause_btn = self._button("暂停", variant="compact")
        self.video_pause_btn.setEnabled(False)
        self.video_pause_btn.setToolTip("暂停或继续视频检测")
        self.video_pause_btn.clicked.connect(self.toggle_video_pause)
        self.video_timeline_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.video_timeline_slider.setObjectName("videoTimelineSlider")
        self.video_timeline_slider.setRange(0, 0)
        self.video_timeline_slider.setTracking(False)
        self.video_timeline_slider.setEnabled(False)
        self.video_timeline_slider.setToolTip("拖动后松开，可快速定位视频检测位置")
        self.video_timeline_label = QtWidgets.QLabel("00:00 / --:--")
        self.video_timeline_label.setObjectName("videoTimelineLabel")
        self.video_timeline_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.video_timeline_slider.sliderPressed.connect(self._video_timeline_slider_pressed)
        self.video_timeline_slider.sliderMoved.connect(self._video_timeline_slider_moved)
        self.video_timeline_slider.sliderReleased.connect(self._video_timeline_slider_released)
        self.video_timeline_slider.valueChanged.connect(self._video_timeline_value_changed)
        self.video_timeline_layout.addWidget(self.video_pause_btn)
        self.video_timeline_layout.addWidget(self.video_timeline_slider, 1)
        self.video_timeline_layout.addWidget(self.video_timeline_label)
        self.preview_layout.addWidget(self.video_timeline_widget)
        self._update_video_timeline_visibility()
        self.center_layout.addWidget(self.preview_panel, 3)

        self.plan_panel = self._panel()
        self.plan_panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        self.plan_layout = QtWidgets.QVBoxLayout(self.plan_panel)
        self.plan_layout.addWidget(self._section_label("最佳路线"))
        self.best_label = QtWidgets.QLabel("无")
        self.best_label.setObjectName("bestShot")
        self.best_label.setWordWrap(True)
        self.plan_layout.addWidget(self.best_label)
        self.plan_layout.addWidget(self._section_label("候选路线"))
        self.candidates = QtWidgets.QTableWidget(0, 4)
        self.candidates.setHorizontalHeaderLabels(["ID", "袋口", "评分", "风险"])
        self.candidates.horizontalHeader().setStretchLastSection(True)
        self.candidates.verticalHeader().setVisible(False)
        self.candidates.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.candidates.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.plan_layout.addWidget(self.candidates, 1)
        self.center_layout.addWidget(self.plan_panel, 0)

    def _update_preview_zoom_controls(self, factor: float) -> None:
        self.preview_zoom_label.setText(f"{int(round(float(factor) * 100.0))}%")
        self.preview_zoom_out_btn.setEnabled(
            factor > self.preview_frame.MIN_ZOOM_FACTOR + 1e-6
        )
        self.preview_zoom_in_btn.setEnabled(
            factor < self.preview_frame.MAX_ZOOM_FACTOR - 1e-6
        )
        self.preview_fit_btn.setEnabled(abs(factor - 1.0) > 1e-6)

    def _show_desktop_pocket_notice(self, notices: list[dict[str, object]]) -> None:
        messages = [str(notice.get("message") or "").strip() for notice in notices]
        messages = [message for message in messages if message]
        if not messages:
            return
        self.pocket_notice_label.setText(f"进球！\n{' · '.join(messages)}")
        self.pocket_notice_label.adjustSize()
        self._position_desktop_pocket_notice()
        self.pocket_notice_label.show()
        self.pocket_notice_label.raise_()
        self._desktop_pocket_notice_timer.start(4500)

    def _position_desktop_pocket_notice(self) -> None:
        label = getattr(self, "pocket_notice_label", None)
        frame = getattr(self, "preview_frame", None)
        if label is None or frame is None:
            return
        label.adjustSize()
        x = max(12, int(round((frame.width() - label.width()) * 0.5)))
        y = max(12, int(round(frame.height() * 0.08)))
        label.move(x, y)

    def _clear_pocket_notices(self) -> None:
        self._pocket_notice_tracker().reset()
        label = getattr(self, "pocket_notice_label", None)
        if label is not None:
            label.hide()
        timer = getattr(self, "_desktop_pocket_notice_timer", None)
        if timer is not None:
            timer.stop()

    def _build_right_sidebar(self) -> None:
        self.right_scroll_area = self._scroll_area()
        self.right_panel = self._panel()
        self.right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        self.right_scroll_area.setWidget(self.right_panel)

        module_section = self._section("模块状态")
        module_layout = module_section.contentLayout()
        self.module_status = QtWidgets.QTableWidget(0, 3)
        self.module_status.setHorizontalHeaderLabels(["模块", "状态", "细节"])
        self.module_status.horizontalHeader().setStretchLastSection(True)
        self.module_status.verticalHeader().setVisible(False)
        self.module_status.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.module_status.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        module_layout.addWidget(self.module_status)
        self._module_rows = {}
        for name in ["采集", "检测", "跟踪", "状态机", "规划", "投影", "回放", "抓取", "标定"]:
            self._add_module_row(name)
        self.right_layout.addWidget(module_section)

        manual_section = self._section("状态机人工介入")
        manual_layout = manual_section.contentLayout()
        self.manual_phase_combo = QtWidgets.QComboBox()
        self.manual_phase_combo.addItems([phase.value for phase in MatchPhase])
        self.force_phase_btn = self._button("强制状态", variant="compact")
        self.hold_state_btn = self._button("冻结状态机", variant="compact")
        self.deep_debug_btn = self._button("开始深入调试", variant="compact")
        self.snapshot_state_btn = self._button("确认当前布局稳定", variant="compact")
        self.reset_state_btn = self._button("重置状态机", variant="compact")
        self.force_phase_btn.clicked.connect(self.force_state_phase)
        self.hold_state_btn.clicked.connect(self.toggle_state_hold)
        self.deep_debug_btn.clicked.connect(self.toggle_deep_debug_mode)
        self.snapshot_state_btn.clicked.connect(self.snapshot_stable_layout)
        self.reset_state_btn.clicked.connect(self.reset_state_machine)
        state_target_box = QtWidgets.QWidget()
        state_target_layout = QtWidgets.QVBoxLayout(state_target_box)
        state_target_layout.setContentsMargins(0, 0, 0, 0)
        state_target_layout.addWidget(self.manual_phase_combo)
        state_target_layout.addWidget(self.deep_debug_btn)
        manual_layout.addWidget(self._field("目标状态", state_target_box))
        manual_grid = CompactButtonGrid(columns=2)
        self._compact_grids.append(manual_grid)
        for button in [self.force_phase_btn, self.hold_state_btn, self.snapshot_state_btn, self.reset_state_btn]:
            manual_grid.addButton(button)
        manual_layout.addWidget(manual_grid)
        self.right_layout.addWidget(manual_section)

        self.review_section = self._section("复核", expanded=False)
        review_layout = self.review_section.contentLayout()
        self.confirm_review_btn = self._button("确认复核", variant="compact")
        self.reject_review_btn = self._button("驳回复核", variant="compact")
        self.review_group_combo = QtWidgets.QComboBox()
        self.review_group_combo.addItem("实色", "solid")
        self.review_group_combo.addItem("花色", "stripe")
        self.resolve_open_table_group_btn = self._button("确认开台花色", variant="compact")
        self.review_status_label = QtWidgets.QLabel("暂无待复核")
        self.review_status_label.setObjectName("muted")
        self.review_pending_list = QtWidgets.QListWidget()
        self.review_pending_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.confirm_review_btn.clicked.connect(self.confirm_state_review)
        self.reject_review_btn.clicked.connect(self.reject_state_review)
        self.resolve_open_table_group_btn.clicked.connect(self.resolve_state_open_table_group)
        review_layout.addWidget(self.review_status_label)
        review_layout.addWidget(self.review_pending_list)
        review_actions = CompactButtonGrid(columns=2)
        self._compact_grids.append(review_actions)
        review_actions.addButton(self.confirm_review_btn)
        review_actions.addButton(self.reject_review_btn)
        review_layout.addWidget(review_actions)
        review_group_box = QtWidgets.QWidget()
        review_group_layout = QtWidgets.QHBoxLayout(review_group_box)
        review_group_layout.setContentsMargins(0, 0, 0, 0)
        review_group_layout.addWidget(self.review_group_combo, 1)
        review_group_layout.addWidget(self.resolve_open_table_group_btn)
        review_layout.addWidget(self._field("开台花色复核", review_group_box))
        self.right_layout.addWidget(self.review_section)

        events_log_section = self._section("事件 / 日志")
        self.events_log_layout = events_log_section.contentLayout()
        self.event_list = QtWidgets.QListWidget()
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setObjectName("logBox")
        self.log_box.setReadOnly(True)
        self.events_log_layout.addWidget(self._field("事件", self.event_list))
        self.events_log_layout.addWidget(self._field("日志", self.log_box))
        self.right_layout.addWidget(events_log_section, 1)
        self.right_layout.addStretch(1)

    def _panel(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("panel")
        return frame

    def _scroll_area(self) -> QtWidgets.QScrollArea:
        area = QtWidgets.QScrollArea()
        area.setObjectName("panelScroll")
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        return area

    def _section(self, title: str, *, expanded: bool = True) -> CollapsibleSection:
        section = CollapsibleSection(title, expanded=expanded)
        content_layout = section.contentLayout()
        content_layout.setContentsMargins(0, 8, 0, 0)
        content_layout.setSpacing(8)
        self._section_content_layouts.append(content_layout)
        return section

    def _section_label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("section")
        return label

    def _button(self, text: str, *, variant: str = "default") -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setCursor(QtCore.Qt.PointingHandCursor)
        self._set_button_variant(button, variant)
        return button

    @staticmethod
    def _set_button_variant(button: QtWidgets.QPushButton, variant: str) -> None:
        button.setProperty("variant", variant)
        style = button.style()
        style.unpolish(button)
        style.polish(button)

    def _field(self, label_text: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QtWidgets.QLabel(label_text)
        label.setObjectName("muted")
        layout.addWidget(label)
        layout.addWidget(widget)
        self._field_layouts.append(layout)
        return box

    def _metric(self, name: str, value: str) -> QtWidgets.QFrame:
        box = QtWidgets.QFrame()
        box.setObjectName("metric")
        layout = QtWidgets.QVBoxLayout(box)
        name_label = QtWidgets.QLabel(name)
        name_label.setObjectName("metricName")
        value_label = QtWidgets.QLabel(value)
        value_label.setObjectName("metricValue")
        layout.addWidget(name_label)
        layout.addWidget(value_label)
        box.value_label = value_label  # type: ignore[attr-defined]
        box.metric_layout = layout  # type: ignore[attr-defined]
        return box

    def _form_panel(self) -> tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self._form_layouts.append(layout)
        return box, layout

    def _add_form_row(self, form: QtWidgets.QFormLayout, label_text: str, widget: QtWidgets.QWidget) -> None:
        label = QtWidgets.QLabel(label_text)
        label.setObjectName("muted")
        form.addRow(label, widget)

    def _configure_capture_action_buttons(self) -> None:
        self.raw_photo_btn.setToolTip(self.RAW_PHOTO_TOOLTIP)
        self.instant_replay_export_btn.setToolTip(self.INSTANT_REPLAY_TOOLTIP)
        self.raw_video_btn.setToolTip(self.RAW_VIDEO_TOOLTIP)
        self.route_video_btn.setToolTip(self.ROUTE_VIDEO_TOOLTIP)

    def _add_module_row(self, name: str) -> None:
        row = self.module_status.rowCount()
        self.module_status.insertRow(row)
        self._module_rows[name] = row
        for col, value in enumerate([name, "待机", "--"]):
            item = QtWidgets.QTableWidgetItem(value)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.module_status.setItem(row, col, item)

    def _set_module_status(self, name: str, status: str, detail: str = "") -> None:
        row = self._module_rows.get(name)
        if row is None:
            return
        values = [name, status, detail or "--"]
        for col, value in enumerate(values):
            item = self.module_status.item(row, col)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                self.module_status.setItem(row, col, item)
            item.setText(value)

    def _scale_for_size(self) -> float:
        return float(max(0.92, min(1.55, min(self.width() / self.BASE_WIDTH, self.height() / self.BASE_HEIGHT))))

    def _apply_scale(self, force: bool = False) -> None:
        scale = self._scale_for_size()
        if not force and abs(scale - self._ui_scale) < 0.025:
            return
        self._ui_scale = scale

        def px(value: float) -> int:
            return max(1, int(round(value * scale)))

        self.outer_layout.setContentsMargins(px(14), px(14), px(14), px(14))
        self.outer_layout.setSpacing(px(12))
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        self.header_layout.setSpacing(px(8))
        self.header_top_row.setSpacing(px(12))
        self.header_metrics_layout.setSpacing(px(10))
        self.side_layout.setContentsMargins(px(14), px(14), px(14), px(14))
        self.side_layout.setSpacing(px(10))
        self.left_scroll_area.setMinimumWidth(px(304))
        self.left_scroll_area.setMaximumWidth(px(304))
        self.center_layout.setSpacing(px(12))
        self.preview_layout.setContentsMargins(px(4), px(3), px(4), px(4))
        self.preview_layout.setSpacing(px(3))
        self.preview_header_layout.setSpacing(px(6))
        self.preview_zoom_label.setMinimumWidth(px(44))
        self.video_timeline_layout.setSpacing(px(10))
        self.video_timeline_label.setMinimumWidth(px(104))
        self.plan_layout.setContentsMargins(px(12), px(12), px(12), px(12))
        self.plan_layout.setSpacing(px(8))
        self.right_layout.setContentsMargins(px(14), px(14), px(14), px(14))
        self.right_layout.setSpacing(px(10))
        self.right_scroll_area.setMinimumWidth(px(372))
        self.right_scroll_area.setMaximumWidth(px(372))
        self.main_splitter.setHandleWidth(px(6))
        if force:
            self.main_splitter.setSizes([px(304), px(700), px(372)])
        self.preview_frame.setPreferredSize(px(680), px(382))
        self.best_label.setMaximumHeight(px(72))
        self.plan_panel.setMaximumHeight(px(236))
        for layout in self._section_content_layouts:
            layout.setContentsMargins(0, px(8), 0, 0)
            layout.setSpacing(px(8))
        for layout in self._field_layouts:
            layout.setSpacing(px(4))
        for form in self._form_layouts:
            form.setHorizontalSpacing(px(8))
            form.setVerticalSpacing(px(6))
        for grid in self._compact_grids:
            grid_layout = grid.gridLayout()
            grid_layout.setHorizontalSpacing(px(6))
            grid_layout.setVerticalSpacing(px(6))
        self.module_status.setMinimumHeight(px(170))
        self.module_status.setMaximumHeight(px(230))
        self.module_status.verticalHeader().setDefaultSectionSize(px(22))
        self.candidates.verticalHeader().setDefaultSectionSize(px(28))
        self.candidates.setMinimumHeight(px(96))
        self.candidates.setMaximumHeight(px(136))
        self.review_pending_list.setMinimumHeight(px(86))
        self.review_pending_list.setMaximumHeight(px(132))
        self.event_list.setMinimumHeight(px(150))
        self.log_box.setMinimumHeight(px(120))
        self.log_box.setMaximumHeight(px(180))
        for metric in [self.fps_metric, self.det_metric, self.track_metric, self.phase_metric]:
            metric.metric_layout.setContentsMargins(px(12), px(9), px(12), px(9))  # type: ignore[attr-defined]
            metric.metric_layout.setSpacing(px(2))  # type: ignore[attr-defined]
        self.setStyleSheet(self._stylesheet(scale))
        self._refresh_preview_pixmap()

    def _stylesheet(self, scale: float) -> str:
        def px(value: float) -> int:
            return max(1, int(round(value * scale)))

        checked_icon = (Path(__file__).resolve().parent / "assets" / "checkbox_checked.svg").as_posix()

        return f"""
        QMainWindow, QWidget {{
            background: #101010;
            color: #f2f2f2;
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: {px(13)}px;
        }}
        QLabel#title {{ font-size: {px(22)}px; font-weight: 700; color: #ffffff; }}
        QLabel#muted {{ color: #b8b8b8; font-size: {px(12)}px; }}
        QLabel#status {{
            padding: {px(5)}px {px(10)}px;
            border: 1px solid #444444;
            background: #1b1b1b;
            color: #ffffff;
            font-weight: 600;
        }}
        QScrollArea#panelScroll {{
            background: transparent;
            border: none;
        }}
        QFrame#panel, QFrame#metric {{
            background: #181818;
            border: 1px solid #343434;
            border-radius: {px(2)}px;
        }}
        QFrame#collapsibleSection {{
            background: transparent;
            border: none;
        }}
        QToolButton#sectionToggle {{
            background: #151515;
            border: 1px solid #343434;
            border-radius: {px(2)}px;
            color: #ffffff;
            font-weight: 700;
            padding: {px(7)}px {px(10)}px;
            text-align: left;
        }}
        QToolButton#sectionToggle:hover {{ background: #1d1d1d; border-color: #4c4c4c; }}
        QFrame#previewViewport {{
            background: transparent;
            border: none;
        }}
        QLabel#preview {{
            background: #050505;
            border: 1px solid #3a3a3a;
            color: #cfcfcf;
            font-size: {px(16)}px;
        }}
        QLabel#previewCaption {{
            color: #b8b8b8;
            font-size: {px(11)}px;
            font-weight: 600;
            padding: 0;
            min-height: {px(14)}px;
            max-height: {px(18)}px;
        }}
        QLabel#section {{ color: #ffffff; font-weight: 700; padding-top: {px(5)}px; }}
        QLabel#metricName {{ color: #b8b8b8; font-size: {px(12)}px; }}
        QLabel#metricValue {{ color: #ffffff; font-size: {px(20)}px; font-weight: 700; }}
        QLabel#bestShot {{ background: #111111; border: 1px solid #3a3a3a; padding: {px(10)}px; color: #ffffff; }}
        QPushButton {{
            background: #202020;
            border: 1px solid #4a4a4a;
            border-radius: {px(2)}px;
            padding: {px(9)}px {px(10)}px;
            color: #ffffff;
            font-weight: 600;
            text-align: center;
        }}
        QPushButton:hover {{ background: #2b2b2b; border-color: #666666; }}
        QPushButton:pressed {{ background: #171717; }}
        QPushButton:disabled {{ color: #777777; background: #181818; border-color: #303030; }}
        QPushButton[variant="primary"] {{
            background: #e9e9e9;
            border-color: #ffffff;
            color: #121212;
            padding: {px(10)}px {px(12)}px;
        }}
        QPushButton[variant="primary"]:hover {{ background: #ffffff; border-color: #ffffff; }}
        QPushButton[variant="primary"]:pressed {{ background: #d8d8d8; }}
        QPushButton[variant="compact"] {{
            padding: {px(6)}px {px(8)}px;
            min-height: {px(30)}px;
            font-weight: 600;
        }}
        QPushButton[variant="preview"] {{
            padding: 0 {px(6)}px;
            min-height: {px(14)}px;
            max-height: {px(18)}px;
            font-size: {px(11)}px;
            font-weight: 600;
        }}
        QComboBox, QSpinBox, QDoubleSpinBox {{
            background: #111111;
            border: 1px solid #444444;
            border-radius: {px(2)}px;
            padding: {px(6)}px;
            min-height: {px(25)}px;
            color: #ffffff;
        }}
        QCheckBox {{ color: #ffffff; spacing: {px(8)}px; }}
        QCheckBox::indicator {{
            width: {px(15)}px;
            height: {px(15)}px;
            border: 1px solid #9a9a9a;
            border-radius: {px(2)}px;
            background: #d8d8d8;
        }}
        QCheckBox::indicator:hover {{
            background: #eeeeee;
            border-color: #c8c8c8;
        }}
        QCheckBox::indicator:checked {{
            background: #f0f0f0;
            border-color: #ffffff;
            image: url({checked_icon});
        }}
        QCheckBox::indicator:disabled {{
            background: #777777;
            border-color: #555555;
        }}
        QTableWidget, QListWidget, QPlainTextEdit {{
            background: #111111;
            border: 1px solid #343434;
            color: #ffffff;
            gridline-color: #303030;
            selection-background-color: #3a3a3a;
            selection-color: #ffffff;
        }}
        QHeaderView::section {{ background: #202020; color: #ffffff; border: 0; padding: {px(6)}px; font-weight: 600; }}
        """

    def _sync_controls_from_config(self) -> None:
        self.backend_combo.setCurrentText(str(self.config.camera.backend))
        self.device_spin.setValue(int(self.config.camera.device_index))
        res = f"{int(self.config.camera.width)}x{int(self.config.camera.height)}"
        if self.resolution_combo.findText(res) < 0:
            self.resolution_combo.addItem(res)
        self.resolution_combo.setCurrentText(res)
        self.fps_combo.setCurrentText(str(int(self.config.camera.fps)))
        operating_mode = normalize_operating_mode(self.config.training.operating_mode)
        operating_index = self.runtime_mode_combo.findData(operating_mode)
        self.runtime_mode_combo.blockSignals(True)
        self.runtime_mode_combo.setCurrentIndex(max(0, operating_index))
        self.runtime_mode_combo.blockSignals(False)
        scenario_index = self.training_scenario_combo.findData(self.config.training.scenario_id)
        if scenario_index < 0:
            scenario_index = self.training_scenario_combo.findData("ordered_line_1_7")
            self.config.training.scenario_id = "ordered_line_1_7"
        self.training_scenario_combo.blockSignals(True)
        self.training_scenario_combo.setCurrentIndex(scenario_index)
        self.training_scenario_combo.blockSignals(False)
        mode = str(getattr(self.config.planner, "shot_mode", "rule") or "rule").strip().lower()
        idx = self.shot_mode_combo.findData("hook" if mode in {"hook", "hook_shot", "free", "free_shot"} else "rule")
        self.shot_mode_combo.blockSignals(True)
        self.shot_mode_combo.setCurrentIndex(max(0, idx))
        self.shot_mode_combo.blockSignals(False)
        engine = normalize_state_machine_engine(getattr(self.config.state, "engine", "legacy"))
        engine_idx = self.state_machine_combo.findData(engine)
        self.state_machine_combo.blockSignals(True)
        self.state_machine_combo.setCurrentIndex(max(0, engine_idx))
        self.state_machine_combo.blockSignals(False)
        self.geometry_reference_check.blockSignals(True)
        self.geometry_reference_check.setChecked(bool(self.config.projection.geometry_reference_enabled))
        self.geometry_reference_check.blockSignals(False)
        self.cue_sector_preview_check.blockSignals(True)
        self.cue_sector_preview_check.setChecked(bool(self._cue_sector_preview_enabled))
        self.cue_sector_preview_check.blockSignals(False)
        self.projection_debug_check.blockSignals(True)
        self.projection_debug_check.setChecked(bool(self._projection_debug_enabled))
        self.projection_debug_check.blockSignals(False)
        self._sync_projection_interaction_settings()
        self._sync_mode_dependent_controls()
        self._refresh_training_controls()

    def _sync_config_from_controls(self) -> None:
        self.config.camera.backend = self.backend_combo.currentText()
        self.config.camera.device_index = int(self.device_spin.value())
        nori_id = self.nori_device_combo.currentData()
        self.config.camera.nori_device_id = int(nori_id) if nori_id is not None else None
        try:
            w, h = self.resolution_combo.currentText().lower().split("x", maxsplit=1)
            self.config.camera.width = int(w)
            self.config.camera.height = int(h)
        except Exception:
            pass
        try:
            self.config.camera.fps = int(float(self.fps_combo.currentText()))
        except Exception:
            pass
        self.config.replay.enabled = self.replay_check.isChecked()
        self.config.projection.geometry_reference_enabled = self.geometry_reference_check.isChecked()
        self.config.training.operating_mode = str(self.runtime_mode_combo.currentData() or RULES_MODE)
        self.config.training.scenario_id = str(self.training_scenario_combo.currentData() or "ordered_line_1_7")
        self.config.planner.shot_mode = str(self.shot_mode_combo.currentData() or "rule")
        self.config.state.engine = normalize_state_machine_engine(self.state_machine_combo.currentData())

    @QtCore.pyqtSlot(int)
    def _shot_mode_changed(self, _index: int = 0) -> None:
        self._set_base_shot_mode(str(self.shot_mode_combo.currentData() or "rule"), source="ui")

    @QtCore.pyqtSlot(int)
    def _operating_mode_changed(self, _index: int = 0) -> None:
        self._set_operating_mode(str(self.runtime_mode_combo.currentData() or RULES_MODE), source="ui")

    @QtCore.pyqtSlot(int)
    def _training_scenario_changed(self, _index: int = 0) -> None:
        self._set_training_scenario(
            str(self.training_scenario_combo.currentData() or "ordered_line_1_7"),
            source="ui",
        )

    @QtCore.pyqtSlot(int)
    def _state_machine_changed(self, _index: int = 0) -> None:
        self._set_state_machine_engine(str(self.state_machine_combo.currentData() or "legacy"), source="ui")

    @QtCore.pyqtSlot(bool)
    def _geometry_reference_toggled(self, checked: bool) -> None:
        self.config.projection.geometry_reference_enabled = bool(checked)
        self._save_user_settings()
        if self.last_output is not None:
            self._update_preview(self.last_output)
        checked_message = "\u51e0\u4f55\u53c2\u8003\u7ebf\u5df2\u5f00\u542f" if checked else "\u51e0\u4f55\u53c2\u8003\u7ebf\u5df2\u5173\u95ed"
        self._append_log(checked_message)
        return
        self._append_log("几何参考线已开启" if checked else "几何参考线已关闭")

    @QtCore.pyqtSlot(bool)
    def _projection_debug_toggled(self, checked: bool) -> None:
        self._projection_debug_enabled = bool(checked)
        self._refresh_projection()
        if checked and self.projection_window is None:
            self._append_log("投影调试模式已开启，打开投影窗口后生效")
        else:
            self._append_log("投影调试模式已开启" if checked else "投影调试模式已关闭")

    @QtCore.pyqtSlot(bool)
    def _cue_sector_preview_toggled(self, checked: bool) -> None:
        self._cue_sector_preview_enabled = bool(checked)
        if self.last_output is not None:
            self._update_preview(self.last_output)
        checked_message = "\u7403\u6746\u77e9\u5f62\u5019\u9009\u6846\u5df2\u5f00\u542f" if checked else "\u7403\u6746\u77e9\u5f62\u5019\u9009\u6846\u5df2\u5173\u95ed"
        self._append_log(checked_message)
        return
        self._append_log("鐞冩潌鐭╁舰鍊欓€夋宸插紑鍚?" if checked else "鐞冩潌鐭╁舰鍊欓€夋宸插叧闂?")

    def _apply_projection_tuning_preview(self) -> None:
        calibration = _create_runtime_calibration_service(self.config, self.pipeline)
        if self.pipeline is not None:
            self.pipeline.calibration = calibration
            self.pipeline.planner.calibration = calibration
            self.pipeline.planner.target_shot_planner.calibration = calibration
            self.pipeline.overlay_builder.calibration = calibration
            self.pipeline.training_overlay_builder.calibration = calibration
            if self.last_output is not None:
                self.pipeline._update_table_geometry_for_frame(self.last_output.frame)
                if self.last_output.training is not None:
                    plan = self.last_output.plan
                    overlay = self.pipeline.training_overlay_builder.build(self.last_output.tracks, self.last_output.training)
                else:
                    plan = self.pipeline.planner.plan(self.last_output.state, frame_bgr=self.last_output.frame.image)
                    overlay = self.pipeline.overlay_builder.from_plan(plan)
                self.last_output = self._apply_route_display_filters(
                    replace(self.last_output, plan=plan, overlay=overlay),
                    force_raw=True,
                )
                self.pipeline._last_state = self.last_output.state
                self.pipeline._last_plan = plan
                self.pipeline._last_overlay = overlay
                self._update_plan(self.last_output)
                self._update_module_status(self.last_output)
                self._update_preview(self.last_output)
            else:
                self.pipeline._last_plan = None
                self.pipeline._last_overlay = None
        self._refresh_projection()

    def _save_user_settings(self) -> None:
        UserSettings.from_config(self.config, self.star_formula.to_dict()).save()

    def _sync_projection_interaction_settings(self) -> None:
        self._projection_interaction.set_auto_triggers(
            pocket_enabled=bool(self.config.projection.auto_pocket_animation_enabled),
            victory_enabled=bool(self.config.projection.auto_victory_animation_enabled),
        )

    def _projection_effect_fps_hint(self) -> float:
        if self.pipeline is not None:
            try:
                info = self.pipeline.capture.info()
                fps = float(getattr(info, "fps", 0.0) or 0.0)
                if fps > 0.0:
                    return fps
            except Exception:
                pass
        return float(getattr(self.config.camera, "fps", 0) or 0.0)

    def _queue_projection_notice(self, text: str) -> None:
        if self._projection_interaction.notify(text):
            self._arm_projection_effect_timer()
            self._refresh_projection()

    def _arm_projection_effect_timer(self) -> None:
        if self.projection_window is None:
            return
        if not self._projection_interaction.has_active_content():
            return
        if not self._projection_effect_timer.isActive():
            self._projection_effect_timer.start()

    def _advance_projection_effects(self) -> None:
        if self.projection_window is None or self._projection_calibration_mode:
            self._projection_effect_timer.stop()
            return
        self._refresh_projection()
        if not self._projection_interaction.has_active_content():
            self._projection_effect_timer.stop()

    def _trigger_interaction_test_pocket(self, pocket_index: int) -> None:
        if self.projection_window is None:
            self._projection_calibration_mode = False
            self._ensure_projection_window()
            self.projection_btn.setText("停止投影")
        if self._projection_interaction.trigger_pocket_animation(pocket_index, fps_hint=self._projection_effect_fps_hint()):
            self._queue_projection_notice(f"测试袋口 P{int(pocket_index)}")

    def _trigger_interaction_test_victory(self) -> None:
        if self.projection_window is None:
            self._projection_calibration_mode = False
            self._ensure_projection_window()
            self.projection_btn.setText("停止投影")
        if self._projection_interaction.trigger_victory_animation(fps_hint=self._projection_effect_fps_hint()):
            self._queue_projection_notice("测试结算动画")

    def _current_turn_target_group(self) -> Optional[str]:
        if self.pipeline is not None:
            return self.pipeline.state_machine.turn_target_group
        return self._pending_turn_target_group

    def _apply_pending_turn_target_group(self) -> None:
        if self.pipeline is None or self._pending_turn_target_group is None:
            return
        frame_id, ts_cam_ns = self._last_frame_marker()
        self.pipeline.state_machine.set_turn_target_group(
            self._pending_turn_target_group,
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            reason="runtime_restore",
        )

    def _set_base_shot_mode(self, mode: str, *, source: str) -> None:
        normalized = normalize_shot_mode(mode)
        self.config.planner.shot_mode = normalized
        self.control_state.clear_turn_overrides()
        idx = self.shot_mode_combo.findData(normalized)
        self.shot_mode_combo.blockSignals(True)
        self.shot_mode_combo.setCurrentIndex(max(0, idx))
        self.shot_mode_combo.blockSignals(False)
        if self.pipeline is not None:
            self.pipeline.config.planner.shot_mode = normalized
            self.pipeline.planner.config.shot_mode = normalized
        self._save_user_settings()
        label = "勾球模式" if normalized == "hook" else "规则模式"
        self._append_log(f"画线模式已切换为 {label} ({source})")
        self._queue_projection_notice(label)
        self._refresh_current_plan()
        self._update_module_status(self.last_output)

    def _set_operating_mode(self, mode: str, *, source: str) -> bool:
        normalized = normalize_operating_mode(mode)
        previous = normalize_operating_mode(self.config.training.operating_mode)
        if self.pipeline is not None:
            try:
                self.pipeline.set_operating_mode(normalized)
            except Exception as exc:
                self.runtime_mode_combo.blockSignals(True)
                self.runtime_mode_combo.setCurrentIndex(max(0, self.runtime_mode_combo.findData(previous)))
                self.runtime_mode_combo.blockSignals(False)
                self._append_log(f"运行模式切换失败: {exc}")
                if source == "ui":
                    QtWidgets.QMessageBox.critical(self, "运行模式切换失败", str(exc))
                return False
        self.config.training.operating_mode = normalized
        self.runtime_mode_combo.blockSignals(True)
        self.runtime_mode_combo.setCurrentIndex(max(0, self.runtime_mode_combo.findData(normalized)))
        self.runtime_mode_combo.blockSignals(False)
        self.control_state.clear_turn_overrides()
        self._route_freeze.reset()
        self.last_output = None
        self._sync_mode_dependent_controls()
        self._refresh_training_controls()
        self._save_user_settings()
        label = "编号球训练" if normalized == TRAINING_MODE else "规则比赛"
        self._append_log(f"运行模式已切换为 {label} ({source})")
        self._queue_projection_notice(label)
        self._update_module_status()
        return True

    def _set_training_scenario(self, scenario_id: str, *, source: str) -> bool:
        scenario = get_training_scenario(scenario_id)
        self.config.training.scenario_id = scenario.scenario_id
        index = self.training_scenario_combo.findData(scenario.scenario_id)
        self.training_scenario_combo.blockSignals(True)
        self.training_scenario_combo.setCurrentIndex(max(0, index))
        self.training_scenario_combo.blockSignals(False)
        if self.pipeline is not None:
            self.pipeline.select_training_scenario(scenario.scenario_id)
        self._save_user_settings()
        self._refresh_training_controls()
        self._refresh_projection()
        self._append_log(f"训练项目已切换为 {scenario.title} ({source})")
        return True

    @QtCore.pyqtSlot()
    def start_training_validation(self) -> bool:
        if not self._set_operating_mode(TRAINING_MODE, source="ui"):
            return False
        scenario = get_training_scenario(self.config.training.scenario_id)
        if scenario.projection_only:
            self._projection_calibration_mode = False
            self._ensure_projection_window()
            self._refresh_projection()
            self.training_status_label.setText(
                "出杆检测投影已启动\n纯投影训练 · 无需工业相机"
            )
            self.training_start_btn.setText("刷新投影")
            self._append_log("出杆检测投影已启动，无需工业相机")
            return True
        if self.pipeline is None:
            self.start_pipeline()
        if self.pipeline is None:
            self._append_log("训练启动失败：采集尚未启动")
            return False
        ok, state = self.pipeline.start_training()
        self._refresh_training_controls(state)
        self._append_log(state.message)
        if ok:
            self._queue_projection_notice("训练开始")
        return bool(ok)

    @QtCore.pyqtSlot()
    def reset_training_validation(self) -> bool:
        scenario = get_training_scenario(self.config.training.scenario_id)
        if scenario.projection_only:
            self._refresh_projection()
            self.training_status_label.setText(
                "出杆检测投影已恢复初始位置\n纯投影训练 · 无需工业相机"
            )
            self._append_log("出杆检测投影已恢复初始位置")
            return True
        if self.pipeline is None:
            self._refresh_training_controls()
            return False
        state = self.pipeline.reset_training()
        self._refresh_training_controls(state)
        self._append_log("本次训练已重置，请重新确认摆球")
        self._queue_projection_notice("训练重置")
        return True

    def _sync_mode_dependent_controls(self) -> None:
        training_active = normalize_operating_mode(self.config.training.operating_mode) == TRAINING_MODE
        self.shot_mode_combo.setEnabled(not training_active)
        self.state_machine_combo.setEnabled(not training_active)
        self.training_scenario_combo.setEnabled(True)
        self.training_start_btn.setEnabled(training_active)
        self.training_reset_btn.setEnabled(training_active)

    def _refresh_training_controls(self, state=None) -> None:
        scenario = get_training_scenario(self.config.training.scenario_id)
        self.training_instruction_label.setText(scenario.setup_instructions)
        if scenario.projection_only:
            self.training_status_label.setText(
                "纯投影训练：无需启动工业相机，点击“开始投影”即可显示"
            )
            self.training_start_btn.setText("开始投影")
            self.training_reset_btn.setText("恢复初始位置")
            return
        self.training_reset_btn.setText("重置本次")
        training_session = getattr(self.pipeline, "training_session", None) if self.pipeline is not None else None
        if state is None and training_session is not None:
            state = training_session.state
        if state is None:
            self.training_status_label.setText("切换到编号球训练模式，开始采集并按说明摆球")
            self.training_start_btn.setText("开始验证")
            return
        self.training_status_label.setText(
            f"{state.message}\n{state.mode_hint} · 进度 {state.progress_current}/{state.progress_total}"
            f" · 错误 {state.error_count} · 用时 {state.elapsed_s:.1f}s"
            + (
                "\n剩余球：" + "、".join(str(number) for number in state.remaining_numbers)
                if state.remaining_numbers
                else ""
            )
        )
        self.training_start_btn.setText("重新开始" if state.phase in {"passed", "failed"} else "开始验证")

    def _set_state_machine_engine(self, engine: str, *, source: str) -> None:
        normalized = normalize_state_machine_engine(engine)
        current = normalize_state_machine_engine(getattr(self.config.state, "engine", "legacy"))
        idx = self.state_machine_combo.findData(normalized)
        self.state_machine_combo.blockSignals(True)
        self.state_machine_combo.setCurrentIndex(max(0, idx))
        self.state_machine_combo.blockSignals(False)
        if normalized == current:
            return
        self.config.state.engine = normalized
        self._save_user_settings()
        label = "状态机（新）" if normalized == "modern" else "状态机（旧）"
        self._append_log(f"台球状态机已切换为 {label} ({source})")
        if self.pipeline is not None:
            self._append_log("状态机实现已变更，正在重启采集以应用设置")
            self.stop_pipeline()
            self.start_pipeline()
        elif self._pipeline_start_pending:
            self._append_log("状态机实现已变更，将在本次启动完成后下次采集生效")

    def _set_star_formula_enabled(self, enabled: bool, *, source: str) -> None:
        self.star_formula.enabled = bool(enabled)
        if self.projection_window is not None:
            self.projection_window.set_star_formula(self.star_formula)
        self._save_user_settings()
        self._refresh_projection()
        self._queue_projection_notice("颗星公式开启" if self.star_formula.enabled else "颗星公式关闭")
        self._append_log(f"颗星公式已{'开启' if self.star_formula.enabled else '关闭'} ({source})")

    def _refresh_current_plan(self) -> None:
        if self.pipeline is None or self.last_output is None:
            self._refresh_projection()
            return
        if getattr(self.pipeline, "operating_mode", RULES_MODE) == TRAINING_MODE:
            self._refresh_projection()
            return
        state_for_plan = self.last_output.state
        live_turn_group = getattr(self.pipeline.state_machine, "turn_target_group", state_for_plan.turn_target_group)
        if live_turn_group != state_for_plan.turn_target_group:
            state_for_plan = replace(state_for_plan, turn_target_group=live_turn_group)
        effective_shot_mode = self.control_state.effective_shot_mode(self.config.planner.shot_mode)
        effective_turn_target_group = self.control_state.effective_turn_target_group(
            state_for_plan.turn_target_group,
            effective_shot_mode,
        )
        plan = self.pipeline.planner.plan(
            state_for_plan,
            frame_bgr=self.last_output.frame.image,
            forced_shot_mode=effective_shot_mode,
            forced_turn_target_group=effective_turn_target_group,
        )
        secondary_correction = getattr(self.pipeline, "secondary_correction", None)
        if secondary_correction is not None:
            secondary_correction.arm_from_plan(state_for_plan, plan)
        overlay = self.pipeline.overlay_builder.from_plan(plan)
        self.last_output = self._apply_route_display_filters(
            replace(self.last_output, state=state_for_plan, plan=plan, overlay=overlay),
            force_raw=True,
        )
        self.pipeline._last_state = state_for_plan
        self.pipeline._last_plan = plan
        self.pipeline._last_overlay = overlay
        self._update_plan(self.last_output)
        self._update_preview(self.last_output)
        self._refresh_projection()

    def _apply_route_display_filters(self, out: PipelineOutput, *, force_raw: bool = False) -> PipelineOutput:
        if out.training is not None:
            self._route_freeze.reset()
            return out
        if force_raw:
            decision = self._route_freeze.force(out.state, out.plan, out.overlay)
        else:
            decision = self._route_freeze.update(out.state, out.plan, out.overlay)
        if decision.plan is out.plan and decision.overlay is out.overlay:
            return out
        return replace(out, plan=decision.plan, overlay=decision.overlay)

    def _toggle_turn_target_group(self, *, source: str) -> bool:
        current_group = self._current_turn_target_group()
        next_group = toggled_object_group(current_group)
        if next_group is None:
            self._append_log(f"当前目标花色尚未确定，未执行切换 ({source})")
            return False
        self._pending_turn_target_group = next_group
        if self.pipeline is not None:
            frame_id, ts_cam_ns = self._last_frame_marker()
            self.pipeline.state_machine.set_turn_target_group(
                next_group,
                frame_id=frame_id,
                ts_cam_ns=ts_cam_ns,
                reason=source,
            )
            self._refresh_current_plan()
            self._update_module_status(self.last_output)
            self._append_log(f"当前目标花色已切换为 {next_group} ({source})")
            return True
        self._append_log(f"当前目标花色已预设为 {next_group}，开始采集后生效 ({source})")
        return True

    def _arm_hook_shot_once(self, *, source: str) -> None:
        self.control_state.arm_hook_shot()
        self._refresh_current_plan()
        self._update_module_status(self.last_output)
        self._append_log(f"已启用下一杆勾球 ({source})")

        self._queue_projection_notice("下一杆勾球")

    def _arm_black_shot_once(self, *, source: str) -> None:
        if normalize_shot_mode(self.config.planner.shot_mode) != "rule":
            self._append_log(f"当前为勾球模式，黑球单杆指令未生效 ({source})")
            return
        if self.control_state.hook_shot_active:
            self._append_log(f"当前单杆已被勾球覆盖，黑球单杆指令未生效 ({source})")
            return
        self.control_state.arm_black_shot()
        self._refresh_current_plan()
        self._update_module_status(self.last_output)
        self._append_log(f"已启用单杆黑球击打 ({source})")

        self._queue_projection_notice("下一杆黑球")

    def _process_remote_commands(self) -> None:
        self.web_control.process_actions(self._handle_web_action, limit=32)
        try:
            commands = self._remote_command_queue.drain(limit=24)
        except Exception as exc:
            self._append_log(f"远程命令读取失败: {exc}")
            return
        for command in commands:
            self._execute_remote_command(command)

    def _execute_remote_command(self, command: RemoteCommand) -> None:
        action = command.action
        try:
            if action == "start_capture":
                if self.pipeline is None:
                    self.start_pipeline()
                else:
                    self._append_log("远程命令: 采集已在运行")
            elif action == "stop_capture":
                if self.pipeline is not None:
                    self.stop_pipeline()
                else:
                    self._append_log("远程命令: 采集当前未运行")
            elif action == "toggle_capture":
                self.toggle_capture()
            elif action == "start_projection":
                if self.projection_window is None:
                    self.toggle_projection_window()
                else:
                    self._append_log("远程命令: 投影已在运行")
            elif action == "stop_projection":
                if self.projection_window is not None:
                    self.toggle_projection_window()
                else:
                    self._append_log("远程命令: 投影当前未运行")
            elif action == "toggle_projection":
                self.toggle_projection_window()
            elif action == "toggle_shot_mode":
                next_mode = "hook" if normalize_shot_mode(self.config.planner.shot_mode) == "rule" else "rule"
                self._set_base_shot_mode(next_mode, source="remote")
            elif action == "toggle_target_group":
                self._toggle_turn_target_group(source="remote")
            elif action == "hook_shot_once":
                self._arm_hook_shot_once(source="remote")
            elif action == "black_shot_once":
                self._arm_black_shot_once(source="remote")
            elif action == "toggle_star_formula":
                self._set_star_formula_enabled(not bool(self.star_formula.enabled), source="remote")
            elif action == "save_retro_clip":
                self._trigger_instant_replay_export(source="remote")
            else:
                self._append_log(f"远程命令未识别: {action}")
        except Exception as exc:
            self._append_log(f"远程命令执行失败 {action}: {exc}")

    def _pipeline_restart_signature(self) -> tuple:
        return (
            self.config.camera.backend,
            self.config.camera.exposure_control,
            self.config.camera.device_index,
            self.config.camera.nori_device_id,
            self.config.camera.width,
            self.config.camera.height,
            self.config.camera.fps,
            self.config.camera.frame_rotation_degrees,
            self.config.camera.video_path,
            self.config.camera.nori_sdk_root,
            self.config.camera.exposure_auto,
            self.config.camera.exposure_level,
            self.config.camera.white_balance_auto,
            self.config.camera.white_balance_value,
            self.config.camera.distortion_correction_enabled,
            self.config.camera.distortion_correction_file,
            self.config.detector.backend,
            self.config.detector.model_path,
            self.config.detector.class_file_path,
            tuple(self.config.detector.class_names),
            self.config.detector.conf,
            self.config.detector.iou,
            self.config.detector.device,
            self.config.detector.tile_size,
            self.config.detector.tile_overlap,
            self.config.detector.max_det_per_tile,
            self.config.detector.batch_size,
            self.config.detector.detect_interval_frames,
            self.config.detector.detect_fps_limit_hz,
            self.config.training_detector.backend,
            self.config.training_detector.model_path,
            self.config.training_detector.class_file_path,
            tuple(self.config.training_detector.class_names),
            self.config.training_detector.conf,
            self.config.training_detector.iou,
            self.config.training_detector.device,
            self.config.training_detector.tile_size,
            self.config.training_detector.tile_overlap,
            self.config.training_detector.max_det_per_tile,
            self.config.training_detector.batch_size,
            self.config.training_detector.detect_interval_frames,
            self.config.training_detector.detect_fps_limit_hz,
            self.config.training.pocket_proximity_mm,
            self.config.calibration.camera_file,
            self.config.calibration.normalized_projection_mode(),
            self.config.calibration.active_projection_file(),
            self.config.calibration.engineered_ball_compensation_file,
            self.config.calibration.table_width_mm,
            self.config.calibration.table_height_mm,
            self.config.calibration.ball_diameter_mm,
            self.config.projection.calibration_rotation_degrees,
            self.config.geometry.outline_path,
            self.config.geometry.inline_path,
            self.config.geometry.pocket_path,
            tuple(sorted(vars(self.config.state).items())),
            self.config.planner.enabled,
            self.config.planner.shot_mode,
            self.config.planner.max_cut_angle_deg,
            self.config.planner.top_k,
            self.config.planner.cue_path_margin_mm,
            self.config.planner.object_path_margin_mm,
            self.config.planner.collision_padding_mm,
            self.config.learning.ranker_enabled,
            self.config.learning.ranker_model_path,
            self.config.learning.score_blend,
            self.config.learning.collect_enabled,
            self.config.learning.samples_directory,
            self.config.learning.min_candidates,
            self.config.replay.enabled,
            self.config.replay.directory,
            self.config.replay.write_video,
            self.config.replay.write_debug_frames,
        )

    def _projection_signature(self) -> tuple:
        return (
            self.config.projection.enabled,
            self.config.projection.projector_width,
            self.config.projection.projector_height,
            self.config.projection.screen_index,
            self.config.projection.fullscreen,
            self.config.projection.calibration_rotation_degrees,
            self.config.projection.output_rotation_degrees,
        )

    def _state_machine_or_none(self):
        if self.pipeline is None:
            self._append_log("状态机尚未启动，请先开始采集")
            return None
        return self.pipeline.state_machine

    def _last_frame_marker(self) -> tuple[int, int]:
        if self.last_output is None:
            return 0, 0
        return int(self.last_output.frame.frame_id), int(self.last_output.frame.ts_cam_ns)

    def _pending_review_payload(self) -> dict[str, object]:
        if self.pipeline is None:
            return {}
        snapshot_fn = getattr(self.pipeline.state_machine, "debug_snapshot", None)
        if not callable(snapshot_fn):
            return {}
        try:
            snapshot = snapshot_fn()
        except Exception:
            return {}
        pending = snapshot.get("pending_review")
        return dict(pending) if isinstance(pending, dict) else {}

    @staticmethod
    def _format_pending_review_entry(pocket: dict[str, object], review_reasons: list[str]) -> str:
        parts: list[str] = []
        decision_id = str(pocket.get("decision_id") or "").strip()
        group = str(pocket.get("group") or "").strip()
        pocket_index = pocket.get("pocket_index")
        reason_codes = [str(item) for item in list(pocket.get("reason_codes") or []) if str(item).strip()]
        if decision_id:
            parts.append(decision_id)
        if group:
            parts.append(f"group {group}")
        if pocket_index is not None:
            parts.append(f"pocket {pocket_index}")
        if reason_codes:
            parts.append("reason " + "/".join(reason_codes))
        elif review_reasons:
            parts.append("reason " + "/".join(review_reasons))
        return " | ".join(parts or ["待复核 pocket"])

    def _refresh_review_controls(self) -> None:
        pending = self._pending_review_payload()
        decision_ids = [str(item) for item in list(pending.get("decision_ids") or []) if str(item).strip()]
        review_reasons = [str(item) for item in list(pending.get("review_reasons") or []) if str(item).strip()]
        group_choice_required = bool(pending.get("group_choice_required"))
        items: list[str] = []
        for pocket in list(pending.get("review_pockets") or []):
            if isinstance(pocket, dict):
                items.append(self._format_pending_review_entry(pocket, review_reasons))
        if not items:
            for pocket in list(pending.get("committed_pockets") or []):
                if isinstance(pocket, dict):
                    items.append(self._format_pending_review_entry(pocket, review_reasons))
        if not items and review_reasons:
            items.append("reason " + "/".join(review_reasons))
        if not items and decision_ids:
            items.append("decision " + ", ".join(decision_ids))

        self.review_pending_list.clear()
        for text in items:
            self.review_pending_list.addItem(QtWidgets.QListWidgetItem(text))

        has_pending = bool(items or decision_ids or review_reasons)
        if has_pending:
            label = f"待复核 {max(len(items), len(decision_ids), 1)} 项"
            if group_choice_required:
                label += " / 需要指定开台花色"
            self.review_status_label.setText(label)
            if not self.review_section.isExpanded():
                self.review_section.setExpanded(True)
        else:
            self.review_status_label.setText("暂无待复核")

        self.confirm_review_btn.setEnabled(has_pending)
        self.reject_review_btn.setEnabled(has_pending)
        self.review_group_combo.setEnabled(has_pending and group_choice_required)
        self.resolve_open_table_group_btn.setEnabled(has_pending and group_choice_required)

    @QtCore.pyqtSlot()
    def force_state_phase(self) -> None:
        sm = self._state_machine_or_none()
        if sm is None:
            return
        frame_id, ts_cam_ns = self._last_frame_marker()
        phase = self.manual_phase_combo.currentText()
        sm.force_phase(phase, frame_id=frame_id, ts_cam_ns=ts_cam_ns, reason="operator_ui")
        self._append_log(f"已人工强制状态机进入 {phase}")
        self._update_module_status(self.last_output)

    @QtCore.pyqtSlot()
    def toggle_state_hold(self) -> None:
        sm = self._state_machine_or_none()
        if sm is None:
            return
        frame_id, ts_cam_ns = self._last_frame_marker()
        enabled = not sm.operator_hold
        sm.set_operator_hold(enabled, frame_id=frame_id, ts_cam_ns=ts_cam_ns, reason="operator_ui")
        self.hold_state_btn.setText("恢复自动状态机" if enabled else "冻结状态机")
        self._append_log("状态机已冻结，实时检测仍会记录但不自动推进" if enabled else "状态机已恢复自动推进")
        self._update_module_status(self.last_output)

    @QtCore.pyqtSlot()
    def snapshot_stable_layout(self) -> None:
        sm = self._state_machine_or_none()
        if sm is None:
            return
        if self.last_output is None:
            self._append_log("当前没有可确认的布局帧")
            return
        sm.snapshot_layout(self.last_output.tracks)
        sm.force_phase(MatchPhase.STABLE_IDLE, frame_id=self.last_output.frame.frame_id, ts_cam_ns=self.last_output.frame.ts_cam_ns, reason="operator_stable_layout")
        self._append_log(f"已确认当前布局稳定: {len(self.last_output.tracks.tracks)} 条轨迹")
        self._update_module_status(self.last_output)

    @QtCore.pyqtSlot()
    def reset_state_machine(self) -> None:
        sm = self._state_machine_or_none()
        if sm is None:
            return
        sm.reset()
        self._clear_pocket_notices()
        self.hold_state_btn.setText("冻结状态机")
        self.event_list.clear()
        self._append_log("状态机已重置")
        self._update_module_status(self.last_output)

    @QtCore.pyqtSlot()
    def clear_state_review_flags(self) -> None:
        self.reject_state_review()
        return
        sm = self._state_machine_or_none()
        if sm is None:
            return
        frame_id, ts_cam_ns = self._last_frame_marker()
        sm.clear_review_flags(frame_id=frame_id, ts_cam_ns=ts_cam_ns)
        self._append_log("已清除进洞/异常复核标记")
        self._update_module_status(self.last_output)

    @QtCore.pyqtSlot()
    def confirm_state_review(self) -> None:
        sm = self._state_machine_or_none()
        if sm is None:
            return
        confirm_episode = getattr(sm, "confirm_episode", None)
        if not callable(confirm_episode):
            self._append_log("当前状态机不支持确认复核")
            self._update_module_status(self.last_output)
            return
        frame_id, ts_cam_ns = self._last_frame_marker()
        confirmed = bool(confirm_episode(frame_id=frame_id, ts_cam_ns=ts_cam_ns, reason="operator_ui"))
        if confirmed:
            self._append_log("已确认待复核 pocket，本杆结果已提交")
            self._refresh_current_plan()
        else:
            self._append_log("当前没有可确认的待复核项")
        self._update_module_status(self.last_output)

    @QtCore.pyqtSlot()
    def reject_state_review(self) -> None:
        sm = self._state_machine_or_none()
        if sm is None:
            return
        reject_episode = getattr(sm, "reject_episode", None)
        if not callable(reject_episode):
            self._append_log("当前状态机不支持驳回复核")
            self._update_module_status(self.last_output)
            return
        frame_id, ts_cam_ns = self._last_frame_marker()
        rejected = bool(reject_episode(frame_id=frame_id, ts_cam_ns=ts_cam_ns, reason="operator_ui"))
        if rejected:
            self._append_log("已驳回待复核 pocket，本杆保持冻结前结果")
            self._refresh_current_plan()
        else:
            self._append_log("当前没有可驳回的待复核项")
        self._update_module_status(self.last_output)

    @QtCore.pyqtSlot()
    def resolve_state_open_table_group(self) -> None:
        sm = self._state_machine_or_none()
        if sm is None:
            return
        resolve_group = getattr(sm, "resolve_open_table_group", None)
        if not callable(resolve_group):
            self._append_log("当前状态机不支持开台花色复核")
            self._update_module_status(self.last_output)
            return
        frame_id, ts_cam_ns = self._last_frame_marker()
        group = str(self.review_group_combo.currentData() or self.review_group_combo.currentText() or "").strip().lower()
        resolved = bool(resolve_group(group, frame_id=frame_id, ts_cam_ns=ts_cam_ns, reason="operator_ui"))
        if resolved:
            label = self.review_group_combo.currentText()
            self._append_log(f"已确认开台花色为 {label}，本杆结果已提交")
            self._refresh_current_plan()
        else:
            self._append_log("当前没有需要指定开台花色的待复核项")
        self._update_module_status(self.last_output)

    @QtCore.pyqtSlot()
    def export_diagnostic_snapshot(self) -> None:
        out_dir = Path("local_settings") / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"diagnostic_{time.strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": to_jsonable(self.config),
            "star_formula": self.star_formula.to_dict(),
            "runtime": self._runtime_diagnostic_payload(),
            "module_status": self._module_status_payload(),
            "last_output": None,
        }
        if self.last_output is not None:
            payload["last_output"] = {
                "frame": self._diagnostic_frame_payload(self.last_output.frame),
                "detections": to_jsonable(self.last_output.detections),
                "tracks": to_jsonable(self.last_output.tracks),
                "state": to_jsonable(self.last_output.state),
                "plan": to_jsonable(self.last_output.plan),
                "overlay": to_jsonable(self.last_output.overlay),
            }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._append_log(f"诊断快照已导出: {path}")

    @QtCore.pyqtSlot()
    def capture_raw_photo(self) -> None:
        frame = self._current_raw_frame()
        if frame is None:
            self._append_log("当前没有可用的相机原始画面")
            return
        out_dir = self._capture_output_dir()
        frame_id = self.last_output.frame.frame_id if self.last_output is not None else 0
        path = out_dir / f"no_line_photo_{time.strftime('%Y%m%d_%H%M%S')}_f{frame_id:06d}.jpg"
        ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            self._append_log(f"相机原始照片保存失败: {path}")
            return
        self._append_log(f"相机原始照片已保存: {path}")

    @QtCore.pyqtSlot()
    def trigger_instant_replay_export(self) -> None:
        self._trigger_instant_replay_export(source="ui")

    def _trigger_instant_replay_export(self, *, source: str) -> bool:
        if self.pipeline is None or self.last_output is None:
            self._append_log(f"前60秒纯净视频导出失败：请先开始采集 ({source})")
            return False
        if not self.config.instant_replay.enabled:
            self._append_log(f"前60秒纯净视频缓存已关闭，当前导出请求忽略 ({source})")
            return False
        if self._instant_replay_start_failed:
            self._append_log(f"前60秒纯净视频缓存未正常启动，当前导出请求忽略 ({source})")
            return False
        frame = self._current_recording_frame()
        if frame is None:
            self._append_log(f"{self._recording_frame_error()} ({source})")
            return False
        self._ensure_instant_replay_started(frame)
        if not self._instant_replay.is_running:
            self._append_log(f"前60秒纯净视频缓存仍在预热帧率，请等待几帧后再试 ({source})")
            return False
        if self._instant_replay_start_failed:
            self._append_log(f"前60秒纯净视频缓存启动失败，无法导出 ({source})")
            return False
        result = self._instant_replay.request_export(trigger_ts_ns=self.last_output.frame.ts_cam_ns)
        self._append_log(f"{result.message} ({source})")
        if result.accepted:
            self._queue_projection_notice("精彩时刻已触发")
        self._flush_instant_replay_events()
        self._update_module_status(self.last_output)
        return bool(result.accepted)

    def _ensure_instant_replay_started(self, frame: np.ndarray) -> None:
        if not self.config.instant_replay.enabled or self._instant_replay_start_failed:
            return
        if self._instant_replay.is_running:
            return
        fps = self._recording_fps(require_observed=True)
        if fps is None:
            return
        h, w = frame.shape[:2]
        try:
            self._instant_replay.start(width=w, height=h, fps=fps)
        except FileNotFoundError:
            self._instant_replay_start_failed = True
            self._append_log("前60秒纯净视频缓存启动失败：未找到 ffmpeg。")
        except Exception as exc:
            self._instant_replay_start_failed = True
            self._append_log(f"前60秒纯净视频缓存启动失败: {exc}")

    def _stop_instant_replay(self) -> None:
        try:
            self._instant_replay.stop()
        except Exception as exc:
            self._append_log(f"前60秒纯净视频缓存停止异常: {exc}")
        self._flush_instant_replay_events()

    def _flush_instant_replay_events(self) -> None:
        for message in self._instant_replay.drain_events():
            self._append_log(message)
            if "已导出" in str(message):
                self._queue_projection_notice("精彩时刻已导出")

    def _recording_base_frame(self, out: PipelineOutput) -> Optional[np.ndarray]:
        frame = out.frame.image
        return frame if frame is not None else None

    def _recording_frame_error(self) -> str:
        return "当前没有可用于录制的画面"

    def _observed_recording_fps(self, *, require_ready: bool) -> Optional[float]:
        fps = self._recording_fps_estimator.estimate(require_ready=require_ready)
        if fps is None:
            return None
        return max(1.0, min(240.0, float(fps)))

    @QtCore.pyqtSlot()
    def toggle_raw_video_recording(self) -> None:
        if self._deep_debug_session is not None:
            self._append_log("深入调试进行中，请先停止深入调试，再单独切换无画线视频")
            return
        if self._raw_video_recorder is None:
            self._start_video_recording(route=False)
        else:
            self._stop_video_recording(route=False)

    @QtCore.pyqtSlot()
    def toggle_route_video_recording(self) -> None:
        if self._deep_debug_session is not None:
            self._append_log("深入调试进行中，请先停止深入调试，再单独切换进洞路线视频")
            return
        if self._route_video_recorder is None:
            self._start_video_recording(route=True)
        else:
            self._stop_video_recording(route=True)

    @QtCore.pyqtSlot()
    def toggle_deep_debug_mode(self) -> None:
        if self._deep_debug_session is None:
            self._start_deep_debug_mode()
        else:
            self._stop_deep_debug_mode(reason="ui")

    def _start_video_recording(self, *, route: bool) -> bool:
        if self.pipeline is None:
            self._append_log("请先开始采集再录制视频")
            return False
        frame = self._current_recording_frame()
        if frame is None:
            self._append_log(self._recording_frame_error())
            return False
        record_fps = self._recording_fps(require_observed=True)
        if record_fps is None:
            self._append_log("录制帧率仍在预热，请等待几帧后再试")
            return False
        h, w = frame.shape[:2]
        prefix = "pocket_route_video" if route else "no_line_video"
        path = self._capture_output_dir() / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        try:
            recorder = FfmpegH264Recorder(
                path,
                width=w,
                height=h,
                fps=record_fps,
                bitrate_kbps=6000,
            )
        except FileNotFoundError:
            self._append_log("启动录制失败: 未找到 ffmpeg，请确认 ffmpeg 在 PATH 中")
            return False
        except Exception as exc:
            self._append_log(f"启动录制失败: {exc}")
            return False
        if route:
            self._route_video_recorder = recorder
            self._route_video_path = path
            self.route_video_btn.setText(self.ROUTE_VIDEO_STOP_LABEL)
            if self.last_output is not None and self.last_output.plan.best is None:
                self._append_log("当前无最佳进洞路线，视频会在路线出现后开始画线")
            self._append_log(f"进洞路线画线视频开始录制: {path} ({record_fps:.1f}fps, H.264 6000kbps)")
        else:
            self._raw_video_recorder = recorder
            self._raw_video_path = path
            self.raw_video_btn.setText(self.RAW_VIDEO_STOP_LABEL)
            self._append_log(f"无画线视频开始录制: {path} ({record_fps:.1f}fps, H.264 6000kbps)")
        self._update_module_status(self.last_output)
        self._update_deep_debug_controls(running=self.pipeline is not None)
        return True

    def _stop_video_recording(self, *, route: bool) -> None:
        recorder = self._route_video_recorder if route else self._raw_video_recorder
        path = self._route_video_path if route else self._raw_video_path
        if recorder is None:
            return
        frames = recorder.frames_written
        try:
            code = recorder.close()
        except Exception as exc:
            code = -1
            self._append_log(f"停止录制时 ffmpeg 异常: {exc}")
        if route:
            self._route_video_recorder = None
            self.route_video_btn.setText(self.ROUTE_VIDEO_START_LABEL)
            self._route_video_path = None
        else:
            self._raw_video_recorder = None
            self.raw_video_btn.setText(self.RAW_VIDEO_START_LABEL)
            self._raw_video_path = None
        label = "进洞路线画线视频" if route else "无画线视频"
        if code == 0:
            self._append_log(f"{label}已保存: {path} ({frames} 帧, H.264 6000kbps)")
        else:
            self._append_log(f"{label}保存可能不完整: ffmpeg code={code}, path={path}")
        self._update_module_status(self.last_output)
        self._update_deep_debug_controls(running=self.pipeline is not None)

    def _stop_all_media_recordings(self) -> None:
        if self._raw_video_recorder is not None:
            self._stop_video_recording(route=False)
        if self._route_video_recorder is not None:
            self._stop_video_recording(route=True)

    def _capture_output_dir(self) -> Path:
        path = Path("local_settings") / "captures"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _deep_debug_output_dir(self) -> Path:
        path = Path("local_settings") / "deep_debug"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _start_deep_debug_mode(self) -> None:
        if self.pipeline is None or self.last_output is None:
            self._append_log("请先开始采集，再开启深入调试")
            return
        raw_owned = False
        route_owned = False
        if self._raw_video_recorder is None:
            raw_owned = self._start_video_recording(route=False)
            if not raw_owned:
                return
        if self._route_video_recorder is None:
            route_owned = self._start_video_recording(route=True)
            if not route_owned:
                if raw_owned and self._raw_video_recorder is not None:
                    self._stop_video_recording(route=False)
                return
        self._deep_debug_session = StateDebugSession(
            self._deep_debug_output_dir(),
            capture_fps=self._recording_fps(),
            raw_video_path=self._raw_video_path,
            route_video_path=self._route_video_path,
        )
        self._deep_debug_started_raw_video = raw_owned
        self._deep_debug_started_route_video = route_owned
        self._deep_debug_last_result = None
        self._append_log(f"深入调试已开启: {self._deep_debug_session.session_dir}")
        if not raw_owned or not route_owned:
            reused = []
            if not raw_owned and self._raw_video_recorder is not None:
                reused.append("无画线视频")
            if not route_owned and self._route_video_recorder is not None:
                reused.append("进洞路线视频")
            if reused:
                self._append_log(f"深入调试复用了已在录制的内容: {', '.join(reused)}")
        self._update_module_status(self.last_output)
        self._update_deep_debug_controls(running=True)

    def _stop_deep_debug_mode(self, *, reason: str) -> None:
        session = self._deep_debug_session
        if session is None:
            return
        raw_video_path = self._raw_video_path
        route_video_path = self._route_video_path
        if self._deep_debug_started_raw_video and self._raw_video_recorder is not None:
            self._stop_video_recording(route=False)
        if self._deep_debug_started_route_video and self._route_video_recorder is not None:
            self._stop_video_recording(route=True)
        result = session.close(
            raw_video_path=raw_video_path,
            route_video_path=route_video_path,
            status="completed" if reason == "ui" else reason,
        )
        self._deep_debug_last_result = result
        self._deep_debug_session = None
        self._deep_debug_started_raw_video = False
        self._deep_debug_started_route_video = False
        self._append_log(
            "深入调试已停止: "
            f"{result.session_dir} "
            f"(jsonl={result.jsonl_path.name}, srt={result.srt_path.name}, frames={result.frame_count})"
        )
        self._update_module_status(self.last_output)
        self._update_deep_debug_controls(running=self.pipeline is not None)

    def _record_deep_debug_frame(self, *, raw_out: PipelineOutput, display_out: PipelineOutput) -> None:
        if self._deep_debug_session is None or self.pipeline is None:
            return
        try:
            self._deep_debug_session.record_frame(
                frame=display_out.frame,
                detections=raw_out.detections,
                tracks=raw_out.tracks,
                state=raw_out.state,
                raw_plan=raw_out.plan,
                display_plan=display_out.plan,
                state_debug=self.pipeline.state_machine.debug_snapshot(),
                route_status_text=self._route_freeze.last_status_text,
            )
        except Exception as exc:
            self._append_log(f"深入调试写入失败，已自动停止: {exc}")
            self._stop_deep_debug_mode(reason="write_error")

    def _update_deep_debug_controls(self, *, running: bool) -> None:
        active = self._deep_debug_session is not None
        self.deep_debug_btn.setEnabled(bool(running))
        self.deep_debug_btn.setText("停止深入调试" if active else "开始深入调试")
        self.raw_video_btn.setEnabled(bool(running) and not active)
        self.route_video_btn.setEnabled(bool(running) and not active)

    def _current_raw_frame(self) -> Optional[np.ndarray]:
        if self.last_output is None:
            return None
        frame = self.last_output.frame.image
        return frame if frame is not None else None

    def _current_recording_frame(self) -> Optional[np.ndarray]:
        if self.last_output is None:
            return None
        return self._recording_base_frame(self.last_output)

    def _recording_fps(self, *, require_observed: bool = False) -> Optional[float]:
        observed = self._observed_recording_fps(require_ready=require_observed)
        if observed is not None:
            return observed
        if require_observed:
            return None
        fps = 0.0
        if self.pipeline is not None:
            try:
                fps = float(self.pipeline.capture.info().fps)
            except Exception:
                fps = 0.0
        if fps <= 0.0:
            try:
                fps = float(self.config.camera.fps)
            except Exception:
                fps = 30.0
        observed = self._observed_recording_fps(require_ready=False)
        if observed is not None and abs(observed - fps) >= max(1.0, fps * 0.05):
            fps = observed
        if self.started_at > 0.0 and self.frame_count >= 5:
            actual = self.frame_count / max(1e-6, time.perf_counter() - self.started_at)
            if 1.0 <= actual < fps * 0.9:
                fps = actual
        return max(1.0, min(240.0, fps))

    def _record_media_frames(self, out: PipelineOutput) -> None:
        self._recording_fps_estimator.observe(out.frame.ts_cam_ns)
        recording_frame = self._recording_base_frame(out)
        if recording_frame is None:
            if out.frame.image is not None and self.config.instant_replay.enabled and not self._instant_replay_start_failed:
                self._instant_replay_start_failed = True
                self._append_log(f"前60秒纯净视频缓存未启动: {self._recording_frame_error()}")
            return
        if recording_frame is not None and self.config.instant_replay.enabled:
            self._ensure_instant_replay_started(recording_frame)
            if not self._instant_replay_start_failed:
                try:
                    self._instant_replay.write(recording_frame, ts_ns=out.frame.ts_cam_ns)
                except Exception as exc:
                    self._instant_replay_start_failed = True
                    self._append_log(f"前60秒纯净视频缓存写入失败，已停止自动缓存: {exc}")
        if self._raw_video_recorder is not None and recording_frame is not None:
            self._write_video_frame(route=False, frame=recording_frame)
        if self._route_video_recorder is not None and recording_frame is not None:
            self._write_video_frame(route=True, frame=self._route_capture_frame(out, base_frame=recording_frame))
        self._flush_instant_replay_events()

    def _write_video_frame(self, *, route: bool, frame: np.ndarray) -> None:
        recorder = self._route_video_recorder if route else self._raw_video_recorder
        if recorder is None:
            return
        try:
            recorder.write(frame)
        except Exception as exc:
            label = "进洞路线画线视频" if route else "无画线视频"
            self._append_log(f"{label}写入失败，已停止录制: {exc}")
            try:
                recorder.close()
            except Exception:
                pass
            if route:
                self._route_video_recorder = None
                self.route_video_btn.setText(self.ROUTE_VIDEO_START_LABEL)
            else:
                self._raw_video_recorder = None
                self.raw_video_btn.setText(self.RAW_VIDEO_START_LABEL)
            self._update_module_status(self.last_output)

    def _route_capture_frame(self, out: PipelineOutput, *, base_frame: Optional[np.ndarray] = None) -> np.ndarray:
        frame = base_frame if base_frame is not None else self._recording_base_frame(out)
        if frame is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        img = frame.copy()
        self._draw_plan_preview(img, out)
        return img

    def _table_mm_to_camera_px(self, points) -> np.ndarray:
        if self.pipeline is None:
            return np.zeros((0, 2), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape((-1, 2))
        if pts.size == 0:
            return pts
        return self.pipeline.calibration.table_mm_to_camera_px(pts).astype(np.float32)

    def _draw_arrow_polyline(self, image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        pts = np.asarray(points, dtype=np.float32).reshape((-1, 2))
        if pts.shape[0] < 2:
            return
        ints = np.round(pts).astype(np.int32)
        cv2.polylines(image, [ints.reshape((-1, 1, 2))], False, color, thickness, cv2.LINE_AA)
        cv2.arrowedLine(
            image,
            (int(ints[-2][0]), int(ints[-2][1])),
            (int(ints[-1][0]), int(ints[-1][1])),
            color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.08,
        )

    def _draw_plan_preview(self, image: np.ndarray, out: PipelineOutput) -> None:
        if self.pipeline is None:
            return
        if out.plan.shot_mode == "free":
            if out.plan.free_route is not None:
                self._draw_free_plan_preview(image, out.plan.free_route)
            return
        if out.plan.best is not None:
            self._draw_rule_plan_preview(image, out.plan.best)

    def _draw_geometry_reference_preview(self, image: np.ndarray) -> None:
        if self.pipeline is None:
            return
        draw_geometry_reference_lines(
            image,
            self.pipeline.geometry,
            enabled=bool(self.config.projection.geometry_reference_enabled),
        )

    def _draw_cue_sector_preview(self, image: np.ndarray) -> None:
        if self.pipeline is None or getattr(self.pipeline, "operating_mode", RULES_MODE) == TRAINING_MODE or not self._cue_sector_preview_enabled:
            return
        draw_cue_sector_candidate_box(
            image,
            getattr(self.pipeline.planner.cue_sector, "last_debug_view", None),
            enabled=True,
            ui_scale=float(self._ui_scale),
        )

    def _route_preview_stroke_style(self):
        return projection_route_stroke_style(
            (int(self.config.projection.projector_width), int(self.config.projection.projector_height)),
            self.star_formula,
        )

    def _draw_rule_plan_preview(self, image: np.ndarray, candidate) -> None:
        route_color = (255, 255, 255)
        radius_mm = 0.5 * float(self.pipeline.calibration.table.ball_diameter_mm) if self.pipeline is not None else 28.0
        cue = np.asarray(candidate.cue_ball, dtype=np.float32)
        ghost = np.asarray(candidate.ghost_ball, dtype=np.float32)
        target = np.asarray(candidate.object_ball, dtype=np.float32)
        pocket = np.asarray(candidate.pocket_point, dtype=np.float32)
        style = self._route_preview_stroke_style()
        radius_px = max(6, int(round(self._camera_radius_px(ghost, radius_mm))))
        table = self.pipeline.calibration.table if self.pipeline is not None else None
        inner = self._inner_polygon_table()

        guide_start = cue_alignment_start(
            cue,
            ghost,
            inner,
            table_width_mm=float(table.width_mm) if table is not None else 0.0,
            table_height_mm=float(table.height_mm) if table is not None else 0.0,
        )
        self._draw_segment_trimmed(image, guide_start, cue, route_color, style.solid_line_width, 0.0, radius_mm)
        self._draw_segment_trimmed(image, cue, ghost, route_color, style.solid_line_width, radius_mm, radius_mm)
        object_nodes = [np.asarray(point, dtype=np.float32) for point in (getattr(candidate, "object_line", None) or [])]
        if len(object_nodes) < 2:
            object_nodes = [target, pocket]
        for idx in range(len(object_nodes) - 1):
            self._draw_dashed_segment_trimmed(
                image,
                object_nodes[idx],
                object_nodes[idx + 1],
                route_color,
                style.dashed_line_width,
                radius_mm if idx == 0 else 0.0,
                0.0,
            )
        cue_sep_end = rule_cue_separation_end(
            cue,
            ghost,
            target,
            inner,
            fallback_length_mm=float(np.hypot(table.width_mm, table.height_mm)) if table is not None else 600.0,
        )
        if cue_sep_end is not None:
            self._draw_dashed_segment_trimmed(
                image,
                ghost,
                cue_sep_end,
                route_color,
                style.dashed_line_width,
                radius_mm,
                0.0,
            )

        for point in (ghost, target, *object_nodes[1:-1]):
            cam = self._table_mm_to_camera_px([point])
            if cam.shape[0] >= 1:
                cv2.circle(image, _point_int(cam[0]), radius_px, route_color, style.circle_width, cv2.LINE_AA)
        pocket_px = self._table_mm_to_camera_px([pocket])
        if pocket_px.shape[0] >= 1:
            cv2.circle(image, _point_int(pocket_px[0]), max(6, radius_px // 2), route_color, style.circle_width, cv2.LINE_AA)

    def _draw_free_plan_preview(self, image: np.ndarray, route) -> None:
        route_color = (255, 255, 255)
        cue_stick_color = route_color
        radius_mm = float(max(1.0, route.cue_radius))
        cue = np.asarray(route.cue_ball, dtype=np.float32)
        tip = np.asarray(route.cue_stick_tip, dtype=np.float32)
        tail = np.asarray(route.cue_stick_tail, dtype=np.float32)
        aim_dir = unit(np.asarray(route.aim_direction, dtype=np.float32))
        style = self._route_preview_stroke_style()
        radius_px = max(6, int(round(self._camera_radius_px(cue, radius_mm))))

        stick_px = self._table_mm_to_camera_px([tail, tip])
        if stick_px.shape[0] >= 2:
            cv2.line(image, _point_int(stick_px[0]), _point_int(stick_px[1]), cue_stick_color, style.solid_line_width, cv2.LINE_AA)

        guide_back = cue - aim_dir * max(26.0, 2.2 * radius_mm)
        self._draw_segment_trimmed(image, guide_back, cue, route_color, style.solid_line_width, 0.0, radius_mm)

        nodes = [np.asarray(p, dtype=np.float32) for p in route.path_points or []]
        collision_count = len(route.collision_points or [])
        for i in range(max(0, len(nodes) - 1)):
            start_radius = radius_mm if i <= collision_count else 0.0
            end_radius = radius_mm if (i + 1) <= collision_count else 0.0
            self._draw_segment_trimmed(image, nodes[i], nodes[i + 1], route_color, style.solid_line_width, start_radius, end_radius)

        collisions = [np.asarray(p, dtype=np.float32) for p in route.collision_points or []]
        normals = [np.asarray(n, dtype=np.float32) for n in route.collision_normals or []]
        collision_types = list(route.collision_types or [])
        for idx, collision in enumerate(collisions):
            cam = self._table_mm_to_camera_px([collision])
            if cam.shape[0] >= 1:
                cv2.circle(image, _point_int(cam[0]), radius_px, route_color, style.circle_width, cv2.LINE_AA)
            if idx >= len(collision_types) or str(collision_types[idx]) != "ball":
                continue
            normal = unit(normals[idx] if idx < len(normals) else aim_dir)
            hit_center = collision + normal * (2.0 * radius_mm)
            table = self.pipeline.calibration.table if self.pipeline is not None else None
            hit_end = estimate_route_end(
                hit_center,
                normal,
                self._inner_polygon_table(),
                fallback_length_mm=float(np.hypot(table.width_mm, table.height_mm)) if table is not None else 600.0,
            )
            self._draw_dashed_segment_trimmed(image, hit_center, hit_end, route_color, style.dashed_line_width, radius_mm, 0.0)

    def _camera_radius_px(self, point_mm, radius_mm: float) -> float:
        point = np.asarray(point_mm, dtype=np.float32).reshape((2,))
        pts = self._table_mm_to_camera_px([point, point + np.asarray([float(radius_mm), 0.0], dtype=np.float32)])
        if pts.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(pts[1] - pts[0]))

    def _draw_segment_trimmed(
        self,
        image: np.ndarray,
        start_mm: np.ndarray,
        end_mm: np.ndarray,
        color: tuple[int, int, int],
        thickness: int,
        start_radius_mm: float = 0.0,
        end_radius_mm: float = 0.0,
    ) -> None:
        points = self._trimmed_camera_segment(start_mm, end_mm, start_radius_mm, end_radius_mm)
        if points is None:
            return
        cv2.line(image, _point_int(points[0]), _point_int(points[1]), color, max(1, int(thickness)), cv2.LINE_AA)

    def _draw_dashed_segment_trimmed(
        self,
        image: np.ndarray,
        start_mm: np.ndarray,
        end_mm: np.ndarray,
        color: tuple[int, int, int],
        thickness: int,
        start_radius_mm: float = 0.0,
        end_radius_mm: float = 0.0,
        dash_px: float = 12.0,
        gap_px: float = 16.0,
    ) -> None:
        points = self._trimmed_camera_segment(start_mm, end_mm, start_radius_mm, end_radius_mm)
        if points is None:
            return
        start = points[0]
        end = points[1]
        v = end - start
        dist = float(np.linalg.norm(v))
        if dist < 1e-3:
            return
        d = v / dist
        step = float(max(2.0, dash_px + gap_px))
        drawn = 0.0
        while drawn < dist:
            seg_e = min(dist, drawn + float(max(2.0, dash_px)))
            p1 = start + d * drawn
            p2 = start + d * seg_e
            if int(round(float(p1[0]))) != int(round(float(p2[0]))) or int(round(float(p1[1]))) != int(round(float(p2[1]))):
                cv2.line(image, _point_int(p1), _point_int(p2), color, max(1, int(thickness)), cv2.LINE_AA)
            drawn += step

    def _trimmed_camera_segment(
        self,
        start_mm: np.ndarray,
        end_mm: np.ndarray,
        start_radius_mm: float,
        end_radius_mm: float,
    ) -> Optional[np.ndarray]:
        pts = self._table_mm_to_camera_px([start_mm, end_mm])
        if pts.shape[0] < 2:
            return None
        a = pts[0].astype(np.float32)
        b = pts[1].astype(np.float32)
        v = b - a
        dist = float(np.linalg.norm(v))
        if dist < 1e-3:
            return None
        start_trim = self._camera_radius_px(start_mm, start_radius_mm) if start_radius_mm > 0.0 else 0.0
        end_trim = self._camera_radius_px(end_mm, end_radius_mm) if end_radius_mm > 0.0 else 0.0
        if start_trim + end_trim >= dist - 1.0:
            return None
        d = v / dist
        return np.vstack([a + d * start_trim, b - d * end_trim]).astype(np.float32)

    def _inner_polygon_table(self) -> np.ndarray:
        if self.pipeline is None:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(self.pipeline.calibration.table.inner_polygon_mm, dtype=np.float32).reshape((-1, 2))

    def _diagnostic_frame_payload(self, frame) -> dict[str, object]:
        payload = to_jsonable(frame)
        image = getattr(frame, "image", None)
        if image is not None:
            payload["image"] = {
                "shape": list(image.shape),
                "dtype": str(image.dtype),
            }
        return payload

    def _runtime_diagnostic_payload(self) -> dict[str, object]:
        elapsed = max(1e-6, time.perf_counter() - self.started_at) if self.started_at > 0.0 else 0.0
        return {
            "frame_count": int(self.frame_count),
            "elapsed_s": float(elapsed),
            "avg_fps": float(self.frame_count / elapsed) if elapsed > 0.0 else 0.0,
            "target_timer_interval_ms": int(self._capture_timer_interval_ms()),
            "timer_single_shot": bool(self.timer.isSingleShot()),
            "timer_active": bool(self.timer.isActive()),
            "frame_busy": bool(self._frame_busy),
            "preview_fps_limit": float(self._preview_fps_limit),
            "metrics_fps_limit": float(self._metrics_fps_limit),
            "heavy_ui_fps_limit": float(self._heavy_ui_fps_limit),
            "pipeline_timings_ms": dict(getattr(self.pipeline, "last_timings_ms", {}) if self.pipeline is not None else {}),
            "recordings": {
                "no_line_video": str(self._raw_video_path) if self._raw_video_recorder is not None and self._raw_video_path is not None else None,
                "route_line_video": str(self._route_video_path) if self._route_video_recorder is not None and self._route_video_path is not None else None,
            },
            "deep_debug": {
                "active": bool(self._deep_debug_session is not None),
                "session_dir": str(self._deep_debug_session.session_dir) if self._deep_debug_session is not None else None,
                "last_session_dir": str(self._deep_debug_last_result.session_dir) if self._deep_debug_last_result is not None else None,
            },
        }

    def _module_status_payload(self) -> list[dict[str, str]]:
        rows = []
        for row in range(self.module_status.rowCount()):
            values = []
            for col in range(self.module_status.columnCount()):
                item = self.module_status.item(row, col)
                values.append(item.text() if item is not None else "")
            rows.append({"module": values[0], "status": values[1], "detail": values[2]})
        return rows

    def ensure_projection_window_for_operator(self) -> None:
        self._projection_calibration_mode = True
        self._ensure_projection_window()
        if self.projection_window is not None:
            self.projection_window.set_calibration_mode(True)
        self.projection_btn.setText("停止投影")
        self._append_log("投影窗口已打开，当前处于校正模式")
        self._update_module_status(self.last_output)

    def resume_runtime_projection(self) -> None:
        self._projection_calibration_mode = False
        if self.projection_window is not None:
            self.projection_window.set_calibration_mode(False)
        self._refresh_projection()
        self._append_log("投影已恢复实时 overlay")
        self._update_module_status(self.last_output)

    def show_projector_calibration_result(self, calibration=None) -> None:
        self.ensure_projection_window_for_operator()
        calibration = calibration or _create_calibration_workflow_service(self.config, self.pipeline)
        overlay = self._projector_calibration_overlay(calibration)
        if self.projection_window is not None:
            self.projection_window.set_overlay(overlay)
        stats = calibration.projection.calibration_error_stats()
        if stats:
            self._append_log(
                "投影校正结果已显示: "
                f"mean={stats.get('mean_px', 0.0):.2f}px "
                f"p95={stats.get('p95_px', stats.get('max_px', 0.0)):.2f}px"
            )
        else:
            self._append_log("投影校正结果已显示，但当前文件没有误差统计")

    def show_projector_residual_overlay(self, calibration=None) -> None:
        self.ensure_projection_window_for_operator()
        calibration = calibration or _create_calibration_workflow_service(self.config, self.pipeline)
        overlay = self._projector_calibration_overlay(calibration)
        controls = np.asarray(calibration.projection.residual_field.control_points_cam, dtype=np.float32).reshape((-1, 2))
        offsets = np.asarray(calibration.projection.residual_field.offsets_proj, dtype=np.float32).reshape((-1, 2))
        if controls.shape[0] > 0 and controls.shape == offsets.shape:
            base = calibration.projection.camera_to_projector_points(controls, refined=False).astype(np.float32)
            refined = base + offsets
            for idx, (a, b) in enumerate(zip(base[:80], refined[:80])):
                start = (float(a[0]), float(a[1]))
                end = (float(b[0]), float(b[1]))
                overlay.lines.append(OverlayLine(points=[start, end], color=(255, 255, 255), width=2, label=f"r{idx}"))
                overlay.circles.append(OverlayCircle(center=end, radius=5.0, color=(255, 255, 255)))
            self._append_log(f"已显示 {min(80, controls.shape[0])} 个局部残差箭头")
        else:
            self._append_log("当前校正文件没有局部残差控制点，已显示基础校正结果")
        if self.projection_window is not None:
            self.projection_window.set_overlay(overlay)

    def _update_module_status(self, out: Optional[PipelineOutput] = None) -> None:
        projection_state = "校正模式" if self._projection_calibration_mode else ("运行中" if self.projection_window is not None else "关闭")
        self._set_module_status("投影", projection_state, f"{self.config.projection.projector_width}x{self.config.projection.projector_height}")
        self._set_module_status("回放", "启用" if self.config.replay.enabled else "关闭", self.config.replay.directory)
        capture_recording = self._raw_video_recorder is not None or self._route_video_recorder is not None
        instant_replay_active = self.config.instant_replay.enabled and not self._instant_replay_start_failed
        capture_state = "录制中" if capture_recording else ("缓存中" if instant_replay_active else "待机")
        self._set_module_status("抓取", capture_state, self._media_capture_status_detail())
        if self.pipeline is None:
            self._set_module_status("采集", "离线", self.config.camera.backend)
            idle_detector = self.config.training_detector.backend if self.config.training.operating_mode == TRAINING_MODE else self.config.detector.backend
            self._set_module_status("检测", "待机", idle_detector)
            self._set_module_status("跟踪", "待机", "TemporalTracker")
            idle_state_detail = "自动"
            if self._deep_debug_last_result is not None:
                idle_state_detail += f" / debug {self._deep_debug_last_result.session_dir.name}"
            self._set_module_status("状态机", "待机", idle_state_detail)
            self._set_module_status("规划", "待机", "GeometryPhysics")
            self._set_module_status("标定", "未加载", self.config.calibration.active_projection_file() or "未设置")
            self.hold_state_btn.setText("冻结状态机")
            self._refresh_review_controls()
            return
        info = self.pipeline.capture.info()
        timings = getattr(self.pipeline, "last_timings_ms", {})
        total_ms = float(timings.get("total_ms", 0.0) or 0.0)
        capture_detail = f"{info.backend} {info.width}x{info.height}@{info.fps:.0f}"
        if total_ms > 0.0:
            capture_detail += f" / step {total_ms:.1f}ms"
        self._set_module_status("采集", "运行中", capture_detail)
        calib = self.pipeline.calibration
        self._set_module_status(
            "标定",
            "有效" if calib.projection.is_valid else "缺失",
            f"cam={'Y' if calib.camera.is_valid else 'N'} proj={'Y' if calib.projection.is_valid else 'N'}",
        )
        detector_version = getattr(self.pipeline.detector.detector, "version", self.config.detector.backend)
        if out is None:
            det_detail = detector_version
        else:
            cache_ratio = float(timings.get("detect_cached_ratio", 0.0) or 0.0)
            det_detail = f"{out.detections.latency_ms:.1f}ms / {len(out.detections.detections)} / cache {cache_ratio:.0%}"
        self._set_module_status("检测", "运行中", det_detail)
        track_detail = self.pipeline.tracker.version if out is None else f"{out.tracks.latency_ms:.1f}ms / {len(out.tracks.tracks)}"
        self._set_module_status("跟踪", "运行中", track_detail)
        if out is not None and out.training is not None:
            training = out.training
            self._set_module_status("状态机", training.phase, training.message)
            self._set_module_status(
                "规划",
                "训练验证",
                f"{training.scenario_title} / {training.progress_current}/{training.progress_total}",
            )
            self.hold_state_btn.setText("训练模式无需规则状态机")
            self._refresh_review_controls()
            return
        state_hold = "冻结" if self.pipeline.state_machine.operator_hold else "自动"
        state_detail = state_hold if out is None else f"{out.state.phase} / conf {out.state.confidence:.2f}"
        if self._deep_debug_session is not None:
            state_detail += f" / debug {self._deep_debug_session.session_dir.name}"
        elif self._deep_debug_last_result is not None:
            state_detail += f" / last {self._deep_debug_last_result.session_dir.name}"
        self._set_module_status("状态机", state_hold, state_detail)
        ranker_version = getattr(self.pipeline.planner.learning_ranker, "version", "learning_unknown")
        mode_name = self.control_state.effective_shot_mode(self.config.planner.shot_mode)
        if out is None:
            plan_detail = f"{mode_name} / {self.pipeline.planner.version} / {ranker_version}"
        elif out.plan.shot_mode == "hook":
            plan_detail = f"hook / {len(out.plan.candidates)} candidates / {out.plan.hook_status}"
            if out.plan.locked_target_id is not None:
                plan_detail += f" / lock #{out.plan.locked_target_id}"
        elif out.plan.shot_mode == "target":
            plan_detail = f"target / {len(out.plan.candidates)} candidates / {out.plan.target_shot_status}"
            if out.plan.locked_target_id is not None:
                plan_detail += f" / lock #{out.plan.locked_target_id}"
        else:
            plan_detail = f"rule / {len(out.plan.candidates)} candidates / {ranker_version}"
            if out.plan.locked_target_id is not None:
                plan_detail += f" / lock #{out.plan.locked_target_id} {out.plan.target_lock_status}"
        if self.config.planner.route_freeze_enabled:
            plan_detail += f" / freeze {self._route_freeze.last_status_text}"
        self._set_module_status("规划", "运行中" if self.config.planner.enabled else "关闭", plan_detail)
        if self.pipeline.recorder is not None:
            self._set_module_status("回放", "记录中", self.pipeline.recorder.session_id)
        self.hold_state_btn.setText("恢复自动状态机" if self.pipeline.state_machine.operator_hold else "冻结状态机")
        self._refresh_review_controls()

    def _media_capture_status_detail(self) -> str:
        parts = []
        if self._raw_video_recorder is not None:
            parts.append(f"无画线 {self._raw_video_recorder.frames_written}帧")
        if self._route_video_recorder is not None:
            parts.append(f"路线 {self._route_video_recorder.frames_written}帧")
        if self._deep_debug_session is not None:
            parts.append(f"深调 {self._deep_debug_session.session_dir.name}")
        if self.config.instant_replay.enabled:
            detail = self._instant_replay.status_detail()
            if self._instant_replay_start_failed:
                detail = "回看启动失败"
            parts.append(detail)
        return "; ".join(parts) if parts else "local_settings/captures"

    @QtCore.pyqtSlot()
    def initialize_graphics_image_module(self) -> None:
        self._sync_config_from_controls()
        self._save_user_settings()
        self._append_log("正在初始化图形图像模块")
        try:
            calibration = _create_runtime_calibration_service(self.config, self.pipeline)
            geometry = TableGeometryLoader.load_optional(
                self.config.geometry.outline_path,
                self.config.geometry.inline_path,
                self.config.geometry.pocket_path,
            )
            detector_config = (
                self.config.training_detector
                if normalize_operating_mode(self.config.training.operating_mode) == TRAINING_MODE
                else self.config.detector
            )
            detector = create_detector(detector_config)
            geometry_state = "几何已加载" if not geometry.is_empty else "几何为空"
            camera_state = "相机标定有效" if calibration.camera.is_valid else "相机标定无效"
            projection_state = "投影校正有效" if calibration.projection.is_valid else "投影校正无效"
            self._append_log(
                "图形图像模块已初始化: "
                f"{getattr(detector, 'version', 'unknown')} | {geometry_state} | "
                f"{camera_state} | {projection_state}"
            )
            self._set_module_status("检测", "已加载", getattr(detector, "version", "unknown"))
            self._set_module_status("标定", "有效" if calibration.projection.is_valid else "缺失", f"{camera_state} / {projection_state}")
            self._queue_projection_notice("系统初始化成功")
            if self.config.projection.geometry_reference_enabled and not geometry.is_empty:
                self._append_log("几何参考线将在前端实时画面中显示，不会输出到投影仪")
        except Exception as exc:
            self._append_log(f"图形图像模块初始化失败: {exc}")
            QtWidgets.QMessageBox.critical(self, "初始化失败", str(exc))

    @QtCore.pyqtSlot()
    def open_calibration_workbench(self) -> None:
        self._sync_config_from_controls()
        self.config.calibration.set_projection_mode("engineered")
        self._save_user_settings()
        self._append_log("正在打开独立几何校准工作台")
        try:
            dialog = CalibrationWorkbenchDialog(self, self)
            dialog.exec_()
        except Exception as exc:
            self._append_log(f"校准工作台打开失败: {exc}")
            QtWidgets.QMessageBox.critical(self, "校准工作台打开失败", str(exc))

    @QtCore.pyqtSlot()
    def run_linked_projector_calibration(self) -> None:
        dialog = JointCalibrationWizardDialog(self, self)
        dialog.exec_()

    @QtCore.pyqtSlot()
    def run_engineered_ball_compensation_wizard(self) -> None:
        dialog = EngineeredBallCompensationWizardDialog(self, self)
        dialog.exec_()

    def _projector_calibration_overlay(self, calibration) -> ProjectionOverlay:
        size = (int(self.config.projection.projector_width), int(self.config.projection.projector_height))
        overlay = ProjectionOverlay(overlay_id="projector_calibration", frame_id=0, projector_size=size)
        poly = np.asarray(calibration.projection.table_polygon_proj, dtype=np.float32).reshape((-1, 2))
        if poly.shape[0] >= 3:
            closed = [(float(x), float(y)) for x, y in np.vstack([poly, poly[0]])]
            overlay.lines.append(OverlayLine(points=closed, color=(255, 255, 255), width=3, label="table"))
        points = np.asarray(calibration.projection.proj_points, dtype=np.float32).reshape((-1, 2))
        if points.shape[0] == 0 and poly.shape[0] >= 3:
            points = poly
        for idx, point in enumerate(points[:60]):
            x, y = float(point[0]), float(point[1])
            overlay.circles.append(OverlayCircle(center=(x, y), radius=7.0, color=(255, 255, 255)))
            overlay.labels.append(((x + 10.0, y - 10.0), f"P{idx}", (255, 255, 255)))
        if poly.shape[0] < 3 and points.shape[0] == 0:
            w, h = size
            overlay.lines.append(
                OverlayLine(
                    points=[(40.0, 40.0), (w - 40.0, 40.0), (w - 40.0, h - 40.0), (40.0, h - 40.0), (40.0, 40.0)],
                    color=(255, 255, 255),
                    width=3,
                    label="fallback",
                )
            )
        return overlay

    @QtCore.pyqtSlot()
    def toggle_capture(self) -> None:
        if self.pipeline is None:
            self.start_pipeline()
        else:
            self.stop_pipeline()

    def _update_video_timeline_visibility(self, *_args) -> None:
        widget = getattr(self, "video_timeline_widget", None)
        if widget is None:
            return
        is_video_mode = str(self.backend_combo.currentText()).strip().lower() == "video"
        widget.setVisible(is_video_mode)
        if not is_video_mode:
            self._video_timeline_state = None
            self.video_timeline_slider.setEnabled(False)
            self.video_pause_btn.setEnabled(False)

    def _configure_video_timeline(self) -> None:
        self._update_video_timeline_visibility()
        if self.pipeline is None or self.video_timeline_widget.isHidden():
            self.video_timeline_slider.setEnabled(False)
            self.video_pause_btn.setEnabled(False)
            return
        state_getter = getattr(self.pipeline, "video_timeline_state", None)
        state = state_getter() if callable(state_getter) else None
        self._set_video_timeline_state(state)

    def _set_video_timeline_state(
        self,
        state: Optional[VideoTimelineState],
        *,
        current_frame: Optional[int] = None,
    ) -> None:
        self._video_timeline_state = state
        if state is None or state.total_frames <= 0:
            self._video_timeline_internal_update = True
            try:
                self.video_timeline_slider.setRange(0, 0)
                self.video_timeline_slider.setValue(0)
            finally:
                self._video_timeline_internal_update = False
            self.video_timeline_slider.setEnabled(False)
            self.video_pause_btn.setEnabled(False)
            self.video_timeline_label.setText("00:00 / --:--")
            return
        frame = state.current_frame if current_frame is None else int(current_frame)
        frame = min(max(0, frame), state.total_frames - 1)
        self._video_timeline_internal_update = True
        try:
            self.video_timeline_slider.setRange(0, state.total_frames - 1)
            self.video_timeline_slider.setValue(frame)
        finally:
            self._video_timeline_internal_update = False
        self.video_timeline_slider.setEnabled(self.pipeline is not None and state.total_frames > 1)
        self.video_pause_btn.setEnabled(self.pipeline is not None)
        self._update_video_timeline_label(frame)

    def _update_video_timeline_label(self, frame_index: int) -> None:
        state = self._video_timeline_state
        if state is None or state.fps <= 0.0:
            self.video_timeline_label.setText("00:00 / --:--")
            return
        current_seconds = float(max(0, int(frame_index))) / state.fps
        self.video_timeline_label.setText(
            f"{format_video_time(current_seconds)} / {format_video_time(state.duration_seconds)}"
        )

    def _video_timeline_slider_pressed(self) -> None:
        self._video_timeline_dragging = True

    def _video_timeline_slider_moved(self, value: int) -> None:
        self._update_video_timeline_label(value)

    def _video_timeline_slider_released(self) -> None:
        self._video_timeline_dragging = False
        # With tracking disabled Qt commits value() only after sliderReleased.
        # sliderPosition() is the actual handle location at this point.
        self._seek_video_timeline(self.video_timeline_slider.sliderPosition())

    def _video_timeline_value_changed(self, value: int) -> None:
        if self._video_timeline_internal_update or self._video_timeline_dragging:
            return
        self._seek_video_timeline(value)

    def _seek_video_timeline(self, frame_index: int) -> None:
        if self.pipeline is None or not self.video_timeline_slider.isEnabled():
            return
        target = int(frame_index)
        if self._video_timeline_last_seek_value == target:
            return
        seek = getattr(self.pipeline, "seek_video", None)
        if not callable(seek):
            return
        self._video_timeline_last_seek_value = target
        try:
            state = seek(target)
        except Exception as exc:
            self._append_log(f"视频定位失败: {exc}")
            self._configure_video_timeline()
            return
        self.last_output = None
        self._last_preview_update_ts = 0.0
        self._route_freeze.reset()
        self._clear_pocket_notices()
        self._set_video_timeline_state(state)
        self._append_log(f"视频已定位到 {format_video_time(state.current_seconds)}")
        if self._video_playback_paused:
            self._tick(allow_paused_frame=True)

    @QtCore.pyqtSlot()
    def toggle_video_pause(self) -> None:
        if self.pipeline is None or self._video_timeline_state is None:
            return
        self._set_video_playback_paused(not self._video_playback_paused, write_log=True)

    def _set_video_playback_paused(self, paused: bool, *, write_log: bool = False) -> None:
        self._video_playback_paused = bool(paused)
        self.video_pause_btn.setText("继续" if self._video_playback_paused else "暂停")
        if self._video_playback_paused:
            self.timer.stop()
        elif self.pipeline is not None:
            self.timer.start(1)
        if write_log:
            self._append_log("视频检测已暂停" if self._video_playback_paused else "视频检测已继续")

    def start_pipeline(self) -> None:
        if self._pipeline_start_pending:
            self._append_log("采集正在启动中")
            return
        self._sync_config_from_controls()
        self._save_user_settings()
        self._append_log("启动采集中")
        self._set_starting(True)
        self._pipeline_start_pending = True
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        try:
            pipeline = RuntimePipeline(self.config, star_formula=self.star_formula, control_state=self.control_state)
        except Exception as exc:
            LOGGER.exception("Failed to start runtime pipeline.")
            self._pipeline_failed(str(exc))
        else:
            self._pipeline_started(pipeline)

    @QtCore.pyqtSlot(object)
    def _pipeline_started(self, pipeline: RuntimePipeline) -> None:
        self._pipeline_start_pending = False
        self.pipeline = pipeline
        if self._manual_web_target_id is not None:
            self.pipeline.planner.set_manual_target(self._manual_web_target_id)
        self._recording_fps_estimator.reset()
        self._instant_replay = InstantReplayBuffer(self.config.instant_replay)
        self._instant_replay_start_failed = False
        self._route_freeze.reset()
        self._clear_pocket_notices()
        self._apply_pending_turn_target_group()
        self.last_output = None
        self.frame_count = 0
        self.started_at = time.perf_counter()
        self._frame_busy = False
        self._last_preview_update_ts = 0.0
        self._last_metrics_update_ts = 0.0
        self._last_heavy_ui_update_ts = 0.0
        self._last_perf_log_ts = 0.0
        self._video_playback_paused = False
        self.video_pause_btn.setText("暂停")
        self._set_running(True)
        self._configure_video_timeline()
        self._sync_mode_dependent_controls()
        self._refresh_training_controls()
        self._update_module_status()
        self._append_log("采集已启动")
        self.timer.start(self._capture_timer_interval_ms())

    @QtCore.pyqtSlot(str)
    def _pipeline_failed(self, message: str) -> None:
        self._pipeline_start_pending = False
        self.pipeline = None
        self._set_running(False)
        self._append_log(f"启动失败: {message}")
        QtWidgets.QMessageBox.critical(self, "启动失败", message)

    def stop_pipeline(self) -> None:
        self.timer.stop()
        self._frame_busy = False
        self._recording_fps_estimator.reset()
        self._stop_deep_debug_mode(reason="pipeline_stop")
        self._stop_instant_replay()
        self._instant_replay_start_failed = False
        self._stop_all_media_recordings()
        self._route_freeze.reset()
        if self.pipeline is not None:
            self._pending_turn_target_group = self.pipeline.state_machine.turn_target_group
            try:
                self.pipeline.close()
            finally:
                self.pipeline = None
        self._set_running(False)
        self._set_video_playback_paused(False)
        self._configure_video_timeline()
        self._update_module_status()
        self._append_log("采集已停止")

    @QtCore.pyqtSlot()
    def open_settings(self) -> None:
        pipeline_signature_before = self._pipeline_restart_signature()
        projection_signature_before = self._projection_signature()
        web_signature_before = (self.config.web_control.host, self.config.web_control.port)
        dialog = SettingsDialog(self.config, self.star_formula, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        dialog.apply_to_config(self.config)
        self._route_freeze.reset()
        self.star_formula = dialog.star_formula_config()
        self._sync_controls_from_config()
        self._save_user_settings()
        pipeline_changed = pipeline_signature_before != self._pipeline_restart_signature()
        projection_changed = projection_signature_before != self._projection_signature()
        web_changed = web_signature_before != (self.config.web_control.host, self.config.web_control.port)
        if web_changed and self.web_control.is_running:
            self.stop_web_control()
            self.start_web_control()
        if self.projection_window is not None:
            if projection_changed:
                self.projection_window.close()
                self.projection_window = None
                self._ensure_projection_window()
                self._append_log("投影设置已更新，投影窗口已重建")
            else:
                self.projection_window.set_star_formula(self.star_formula)
            self._refresh_projection()
        self._append_log("设置已保存")
        if self.pipeline is not None and pipeline_changed:
            self._append_log("运行参数已变化，正在重启采集以应用设置")
            self.stop_pipeline()
            self.start_pipeline()
        elif self._pipeline_start_pending and pipeline_changed:
            self._append_log("运行参数已变化，将在本次启动完成后下次采集生效")

    @QtCore.pyqtSlot()
    def probe_camera_devices(self) -> None:
        self._append_log("正在探测相机")
        self.nori_device_combo.clear()
        self.nori_device_combo.addItem("默认", None)
        try:
            rows = probe_cameras(
                max_index=12,
                nori_sdk_root=self.config.camera.nori_sdk_root,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )
        except Exception as exc:
            self._append_log(f"探测失败: {exc}")
            return
        if not rows:
            self._append_log("未发现相机")
            return
        for backend, idx, width, height, fps in rows:
            self._append_log(f"{backend} id={idx} {width}x{height} {fps:.2f}fps")
            if backend.startswith("nori"):
                self.nori_device_combo.addItem(f"Nori {idx}  {width}x{height}@{fps:.0f}", idx)

    @QtCore.pyqtSlot()
    def toggle_projection_window(self) -> None:
        if self.projection_window is None:
            self._projection_calibration_mode = False
            self._ensure_projection_window()
            self.projection_btn.setText("停止投影")
            self._append_log("投影已启动")
        else:
            self.projection_window.close()
            self.projection_window = None
            self._projection_calibration_mode = False
            self._projection_effect_timer.stop()
            self.projection_btn.setText("开始投影")
            self._append_log("投影已停止")
        self._update_module_status(self.last_output)

    def _ensure_projection_window(self) -> None:
        if self.projection_window is not None:
            return
        self.projection_window = ProjectionWindow(self.config.projection)
        self.projection_window.set_calibration_mode(self._projection_calibration_mode)
        self.projection_window.set_star_formula(self.star_formula)
        self.projection_window.show_on_configured_screen()
        self._projection_interaction.notify_boot_ready()
        self._arm_projection_effect_timer()
        self._refresh_projection()

    def _refresh_projection(self) -> None:
        if self.projection_window is None:
            return
        if self._projection_calibration_mode:
            return
        config = self.__dict__.get("config")
        scenario = (
            get_training_scenario(config.training.scenario_id)
            if config is not None
            else None
        )
        projection_only = (
            config is not None
            and scenario is not None
            and normalize_operating_mode(config.training.operating_mode)
            == TRAINING_MODE
            and scenario.projection_only
        )
        if projection_only:
            calibration = (
                self.pipeline.calibration
                if self.pipeline is not None
                else _create_runtime_calibration_service(config)
            )
            frame_id = (
                int(self.last_output.frame.frame_id)
                if self.last_output is not None
                else 0
            )
            overlay = build_projection_drill_overlay(
                scenario,
                config.projection,
                calibration,
                frame_id=frame_id,
            )
        elif self.last_output is not None:
            overlay = self._projection_overlay_for_output(self.last_output)
        else:
            overlay = ProjectionOverlay(
                overlay_id="blank",
                frame_id=0,
                projector_size=(self.config.projection.projector_width, self.config.projection.projector_height),
            )
        calibration = self.pipeline.calibration if self.pipeline is not None else None
        image = self._projection_interaction.compose_frame(
            overlay,
            star_formula=self.star_formula,
            calibration=calibration,
        )
        self.projection_window.set_image(image)

    def _projection_overlay_for_output(self, out: PipelineOutput) -> ProjectionOverlay:
        base = out.overlay
        if not self._projection_debug_enabled:
            return base
        overlay = ProjectionOverlay(
            overlay_id=f"{base.overlay_id}_debug",
            frame_id=base.frame_id,
            projector_size=base.projector_size,
            lines=list(base.lines),
            circles=list(base.circles),
            labels=list(base.labels),
            texts=list(base.texts),
            suppress_star_formula=base.suppress_star_formula,
        )
        self._append_projection_debug_overlay(overlay, out)
        return overlay

    def _append_projection_debug_overlay(self, overlay: ProjectionOverlay, out: PipelineOutput) -> None:
        if self.pipeline is None or out.frame.image is None:
            return
        append_projected_boundary_overlays(overlay, self.pipeline.calibration)
        h, w = out.frame.image.shape[:2]
        _, inline_px, pockets_px = self.pipeline.geometry.reference_scaled(w, h)
        for points in inline_px:
            self._append_camera_polyline_projection(overlay, points, color=(0, 255, 120), width=2, label="debug_inline")
        for points in pockets_px:
            self._append_camera_polyline_projection(overlay, points, color=(0, 180, 255), width=2, label="debug_pocket")
        append_projected_ball_overlays(
            overlay,
            self.pipeline.calibration,
            tracks=out.tracks.tracks,
            detections=out.detections.detections,
        )

    def _append_camera_polyline_projection(
        self,
        overlay: ProjectionOverlay,
        points_px: np.ndarray,
        *,
        color: tuple[int, int, int],
        width: int,
        label: str,
    ) -> None:
        if self.pipeline is None:
            return
        pts = np.asarray(points_px, dtype=np.float32).reshape((-1, 2))
        if pts.shape[0] < 2:
            return
        try:
            proj = self.pipeline.calibration.camera_px_to_projector_px(pts).astype(np.float32)
        except Exception as exc:
            LOGGER.debug("Projection debug geometry mapping failed: %s", exc)
            return
        overlay.lines.append(
            OverlayLine(
                points=[(float(x), float(y)) for x, y in proj],
                color=color,
                width=width,
                label=label,
            )
        )

    def _tick(self, *, allow_paused_frame: bool = False) -> None:
        tick_start = time.perf_counter()
        if bool(self.__dict__.get("_video_playback_paused", False)) and not allow_paused_frame:
            return
        if self._frame_busy:
            if not bool(self.__dict__.get("_video_playback_paused", False)):
                self.timer.start(self._capture_timer_interval_ms())
            return
        if self.pipeline is None:
            return
        self._frame_busy = True
        try:
            raw_out = self.pipeline.step()
            if raw_out is None:
                self.stop_pipeline()
                return
            self._release_web_target_after_shot(raw_out.state.events)
            new_pocket_notices = self._observe_web_events(raw_out.state.events)
            if new_pocket_notices:
                self._show_desktop_pocket_notice(new_pocket_notices)
            out = self._apply_route_display_filters(raw_out)
            self.last_output = out
            self._video_timeline_last_seek_value = None
            if self._video_timeline_state is not None and not self._video_timeline_dragging:
                self._set_video_timeline_state(self._video_timeline_state, current_frame=out.frame.frame_id)
            self._pending_turn_target_group = self.pipeline.state_machine.turn_target_group
            self.frame_count += 1
            now = time.perf_counter()
            if self._projection_interaction.observe_output(
                state=out.state,
                plan=out.plan,
                fps_hint=self._projection_effect_fps_hint(),
            ):
                self._arm_projection_effect_timer()
            self._record_media_frames(out)
            self._record_deep_debug_frame(raw_out=raw_out, display_out=out)
            if now - self._last_preview_update_ts >= 1.0 / max(1.0, self._preview_fps_limit):
                self._update_preview(out)
                self._last_preview_update_ts = now
            self._update_events(out)
            if now - self._last_metrics_update_ts >= 1.0 / max(1.0, self._metrics_fps_limit):
                self._update_stats(out)
                self._last_metrics_update_ts = now
            if now - self._last_heavy_ui_update_ts >= 1.0 / max(1.0, self._heavy_ui_fps_limit):
                self._update_plan(out)
                self._update_module_status(out)
                self._refresh_training_controls(out.training)
                self._last_heavy_ui_update_ts = now
            self._log_performance_periodically(now)
            self._refresh_projection()
        except Exception as exc:
            self._handle_runtime_failure(exc)
        finally:
            self._frame_busy = False
            if self.pipeline is not None and not bool(self.__dict__.get("_video_playback_paused", False)):
                elapsed_ms = (time.perf_counter() - tick_start) * 1000.0
                self.timer.start(self._capture_timer_interval_ms(elapsed_ms=elapsed_ms))

    def _handle_runtime_failure(self, exc: Exception) -> None:
        LOGGER.exception("Runtime pipeline tick failed.")
        message = str(exc).strip() or exc.__class__.__name__
        self._append_log(f"运行异常: {message}")
        try:
            self.stop_pipeline()
        except Exception:
            LOGGER.exception("Failed to stop pipeline cleanly after a runtime error.")
            self.timer.stop()
            self.pipeline = None
            self._frame_busy = False
            self._set_running(False)
            self._update_module_status()
        QtWidgets.QMessageBox.critical(self, "运行异常", message)

    def _update_preview(self, out: PipelineOutput) -> None:
        frame = out.frame.image
        if frame is None:
            return
        img = frame.copy()
        for det in out.detections.detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (235, 235, 235), 2, cv2.LINE_AA)
            cv2.putText(img, f"{det.cls_name} {det.conf:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * self._ui_scale, (245, 245, 245), 1, cv2.LINE_AA)
        for tr in out.tracks.tracks:
            cx, cy = [int(round(v)) for v in tr.center_px]
            cv2.circle(img, (cx, cy), max(4, int(round(tr.radius_px))), (250, 250, 250), 2, cv2.LINE_AA)
            cv2.putText(img, f"#{tr.track_id} {tr.group}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * self._ui_scale, (230, 230, 230), 1, cv2.LINE_AA)
        if out.training is not None:
            expected = set(out.training.expected_numbers)
            for tr in out.tracks.tracks:
                try:
                    number = int(tr.cls_name)
                except (TypeError, ValueError):
                    continue
                if tr.visibility != "visible" or number not in expected:
                    continue
                center = tuple(int(round(value)) for value in tr.center_px)
                cv2.circle(img, center, max(8, int(round(tr.radius_px * 1.45))), (0, 220, 255), 3, cv2.LINE_AA)
            cv2.putText(
                img,
                f"TRAINING {out.training.phase.upper()} {out.training.progress_current}/{out.training.progress_total} "
                f"TARGET {'/'.join(str(number) for number in out.training.expected_numbers) or '-'}",
                (24, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72 * self._ui_scale,
                (0, 220, 255) if out.training.phase not in {"failed", "passed"} else ((0, 220, 80) if out.training.phase == "passed" else (0, 80, 255)),
                2,
                cv2.LINE_AA,
            )
        self._draw_geometry_reference_preview(img)
        self._draw_cue_sector_preview(img)
        self._draw_plan_preview(img, out)
        self._set_preview_image(img)

    def _set_preview_image(self, image_bgr: np.ndarray) -> None:
        self._last_preview_bgr = image_bgr.copy()
        self.web_control.update_frame(image_bgr)
        self._refresh_preview_pixmap()

    def _refresh_preview_pixmap(self) -> None:
        if self._last_preview_bgr is None:
            return
        image = self._last_preview_bgr
        target = self.preview_label.size()
        target_w = max(1, int(target.width()))
        target_h = max(1, int(target.height()))
        h0, w0 = image.shape[:2]
        scale = min(target_w / max(1, w0), target_h / max(1, h0))
        scaled_size = (max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale))))
        if scaled_size != (w0, h0):
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            image = cv2.resize(image, scaled_size, interpolation=interpolation)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self.preview_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _update_stats(self, out: PipelineOutput) -> None:
        elapsed = max(1e-6, time.perf_counter() - self.started_at)
        self.fps_metric.value_label.setText(f"{self.frame_count / elapsed:.1f}")  # type: ignore[attr-defined]
        self.det_metric.value_label.setText(str(len(out.detections.detections)))  # type: ignore[attr-defined]
        self.track_metric.value_label.setText(str(len(out.tracks.tracks)))  # type: ignore[attr-defined]
        phase = out.training.phase if out.training is not None else out.state.phase
        self.phase_metric.value_label.setText(phase)  # type: ignore[attr-defined]

    def _update_plan(self, out: PipelineOutput) -> None:
        if out.training is not None:
            training = out.training
            self.best_label.setText(
                f"{training.scenario_title}\n{training.message}\n"
                f"进度 {training.progress_current}/{training.progress_total}  用时 {training.elapsed_s:.1f}s"
            )
            self.candidates.setRowCount(0)
            return
        if out.plan.shot_mode == "hook" and out.plan.best is None:
            lock_id = f" #{out.plan.locked_target_id}" if out.plan.locked_target_id is not None else ""
            self.best_label.setText(f"勾球模式{lock_id}\n状态 {out.plan.hook_status}\n无理论进球路线")
            self.candidates.setRowCount(0)
            return
        if out.plan.shot_mode == "free":
            collision_count = len(out.plan.free_route.collision_points) if out.plan.free_route is not None else 0
            self.best_label.setText(f"自由模式\n状态 {out.plan.free_status}\n碰撞 {collision_count}")
            self.candidates.setRowCount(0)
            return
        if out.plan.shot_mode == "target" and out.plan.best is None:
            lock_id = f" #{out.plan.locked_target_id}" if out.plan.locked_target_id is not None else ""
            self.best_label.setText(f"目标击球模式{lock_id}\n状态 {out.plan.target_shot_status}\n无理论进球路线")
            self.candidates.setRowCount(0)
            return
        best = out.plan.best
        lock_text = ""
        if out.plan.locked_target_id is not None:
            lock_text = f"\nlock #{out.plan.locked_target_id} {out.plan.target_lock_status}"
        if best is None:
            self.best_label.setText("无")
        else:
            self.best_label.setText(f"目标 #{best.target_track_id}\n袋口 {best.pocket_index}\n评分 {best.score:.2f}  风险 {best.risk:.2f}\n切角 {best.cut_angle_deg:.1f}°")
        if lock_text:
            self.best_label.setText(self.best_label.text() + lock_text)
        self.candidates.setRowCount(len(out.plan.candidates))
        for row, cand in enumerate(out.plan.candidates):
            for col, value in enumerate([cand.candidate_id, str(cand.pocket_index), f"{cand.score:.2f}", f"{cand.risk:.2f}"]):
                self.candidates.setItem(row, col, QtWidgets.QTableWidgetItem(value))
        self.candidates.resizeColumnsToContents()

    def _update_events(self, out: PipelineOutput) -> None:
        for event in out.state.events:
            text = f"{out.frame.frame_id}: {event.name} {self._event_summary(event.payload)}".strip()
            if self.event_list.count() == 0 or self.event_list.item(self.event_list.count() - 1).text() != text:
                item = QtWidgets.QListWidgetItem(text)
                if event.payload:
                    item.setToolTip(json.dumps(event.payload, ensure_ascii=False, indent=2))
                self.event_list.addItem(item)
        while self.event_list.count() > 80:
            self.event_list.takeItem(0)
        self.event_list.scrollToBottom()

    def _event_summary(self, payload: dict) -> str:
        if not payload:
            return ""
        keys = ["track_id", "track_a", "track_b", "pocket_index", "side", "phase", "distance", "distance_to_pocket_mm", "distance_to_rail"]
        parts = []
        for key in keys:
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, float):
                value = f"{value:.1f}"
            parts.append(f"{key}={value}")
        return "[" + ", ".join(parts) + "]" if parts else ""

    def _set_running(self, running: bool) -> None:
        self.capture_btn.setText("结束采集" if running else "开始采集")
        self.capture_btn.setEnabled(True)
        self.backend_combo.setEnabled(not running)
        self.nori_device_combo.setEnabled(not running)
        self.device_spin.setEnabled(not running)
        self.resolution_combo.setEnabled(not running)
        self.fps_combo.setEnabled(not running)
        self.init_module_btn.setEnabled(not running)
        self.replay_check.setEnabled(not running)
        self.raw_photo_btn.setEnabled(running)
        self.instant_replay_export_btn.setEnabled(running and bool(self.config.instant_replay.enabled))
        if not running:
            self.raw_video_btn.setText(self.RAW_VIDEO_START_LABEL)
            self.route_video_btn.setText(self.ROUTE_VIDEO_START_LABEL)
            self.video_timeline_slider.setEnabled(False)
            self.video_pause_btn.setEnabled(False)
        self.status_label.setText("运行中" if running else "离线")
        self._update_deep_debug_controls(running=running)

    def _set_starting(self, starting: bool) -> None:
        if not starting:
            self._set_running(False)
            return
        self.capture_btn.setText("启动中...")
        self.capture_btn.setEnabled(False)
        self.backend_combo.setEnabled(False)
        self.nori_device_combo.setEnabled(False)
        self.device_spin.setEnabled(False)
        self.resolution_combo.setEnabled(False)
        self.fps_combo.setEnabled(False)
        self.init_module_btn.setEnabled(False)
        self.replay_check.setEnabled(False)
        self.raw_photo_btn.setEnabled(False)
        self.instant_replay_export_btn.setEnabled(False)
        self.raw_video_btn.setEnabled(False)
        self.route_video_btn.setEnabled(False)
        self.video_timeline_slider.setEnabled(False)
        self.video_pause_btn.setEnabled(False)
        self.deep_debug_btn.setEnabled(False)
        self.status_label.setText("启动中")

    def _capture_timer_interval_ms(self, *, elapsed_ms: float = 0.0) -> int:
        try:
            fps = float(self.config.camera.fps)
        except Exception:
            fps = 30.0
        base_ms = 1000.0 / max(1.0, fps)
        return max(1, int(round(base_ms - max(0.0, float(elapsed_ms)))))

    def _log_performance_periodically(self, now: float) -> None:
        if self.pipeline is None or now - self._last_perf_log_ts < 10.0:
            return
        elapsed = max(1e-6, now - self.started_at)
        timings = getattr(self.pipeline, "last_timings_ms", {})
        self._append_log(
            "性能: "
            f"fps={self.frame_count / elapsed:.1f} "
            f"step={float(timings.get('total_ms', 0.0) or 0.0):.1f}ms "
            f"cap={float(timings.get('capture_ms', 0.0) or 0.0):.1f}ms "
            f"det={float(timings.get('detect_ms', 0.0) or 0.0):.1f}ms "
            f"cache={float(timings.get('detect_cached_ratio', 0.0) or 0.0):.0%}"
        )
        self._last_perf_log_ts = now

    def _append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{stamp}] {text}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self._apply_scale()
        super().resizeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._projection_effect_timer.stop()
        self.web_control.stop()
        self.stop_pipeline()
        if self.projection_window is not None:
            self.projection_window.close()
        super().closeEvent(event)


def _point_int(point) -> tuple[int, int]:
    arr = np.asarray(point, dtype=np.float32).reshape((2,))
    return (int(round(float(arr[0]))), int(round(float(arr[1]))))


def run_operator_ui(config: AppConfig) -> int:
    configure_logging(config.logging.directory, config.logging.level)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("BAS Control Console")
    app.setStyle("Fusion")
    window = OperatorWindow(config)
    window.showMaximized()
    return int(app.exec_())
