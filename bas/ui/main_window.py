from __future__ import annotations

import logging
import time
import json
from pathlib import Path
from typing import Optional

from ..config import AppConfig
from ..runtime_env import prepare_runtime_environment, preload_torch_for_backend
from ..user_settings import UserSettings

try:
    _active_config = UserSettings.load().apply_to_config(AppConfig.load()).resolve_paths()
    prepare_runtime_environment()
    preload_torch_for_backend(_active_config.detector.backend)
except Exception:
    prepare_runtime_environment()

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..app import PipelineOutput, RuntimePipeline
from ..calibration import create_calibration_service
from ..calibration.charuco import CharucoBoardSpec, render_charuco_board
from ..calibration.verification import format_holdout_report, verify_holdout_file
from ..capture import probe_cameras
from ..geometry import TableGeometryLoader
from .geometry_reference import draw_geometry_reference_lines
from ..logging_config import configure_logging
from ..media_capture import FfmpegH264Recorder
from ..perception import create_detector
from ..projection.star_formula import StarFormulaConfig
from ..projection.window import ProjectionWindow
from ..schemas import MatchPhase, OverlayLine, ProjectionOverlay, to_jsonable
from ..utils import unit

LOGGER = logging.getLogger(__name__)


COMMON_RESOLUTIONS = ["1920x1080", "3840x2160", "2560x1440", "1280x720", "1280x800", "640x480"]
COMMON_FPS = ["30", "60", "120", "164"]


class _PipelineStartWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, config: AppConfig, star_formula: StarFormulaConfig):
        super().__init__()
        self.config = config
        self.star_formula = star_formula

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            pipeline = RuntimePipeline(self.config, star_formula=self.star_formula)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(pipeline)


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, config: AppConfig, star_formula: StarFormulaConfig, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("设置")
        self.resize(760, 620)
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, 1)

        general = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(general)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        self.model_path = self._path_row(config.detector.model_path, "模型文件 (*.pt *.onnx *.engine);;所有文件 (*.*)")
        self.class_file_path = self._path_row(config.detector.class_file_path, "类别文件 (*.txt *.json);;所有文件 (*.*)")
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
        self.distortion_enabled = QtWidgets.QCheckBox("启用")
        self.distortion_enabled.setChecked(bool(config.camera.distortion_correction_enabled))
        self.distortion_file = self._path_row(config.camera.distortion_correction_file, "OpenCV 标定文件 (*.yaml *.yml *.xml);;所有文件 (*.*)")
        distortion_box = QtWidgets.QWidget()
        distortion_layout = QtWidgets.QHBoxLayout(distortion_box)
        distortion_layout.setContentsMargins(0, 0, 0, 0)
        distortion_layout.addWidget(self.distortion_enabled)
        distortion_layout.addWidget(self.distortion_file, 1)
        self.exposure_auto = QtWidgets.QCheckBox("自动曝光")
        self.exposure_auto.setChecked(bool(config.camera.exposure_auto))
        self.exposure_level = self._spin(int(config.camera.exposure_level if config.camera.exposure_level is not None else -5), -10, 0)
        self.exposure_level.setEnabled(not self.exposure_auto.isChecked())
        self.exposure_auto.toggled.connect(lambda checked: self.exposure_level.setEnabled(not checked))
        exposure_box = QtWidgets.QWidget()
        exposure_layout = QtWidgets.QHBoxLayout(exposure_box)
        exposure_layout.setContentsMargins(0, 0, 0, 0)
        exposure_layout.addWidget(self.exposure_auto)
        exposure_layout.addWidget(QtWidgets.QLabel("手动档位"))
        exposure_layout.addWidget(self.exposure_level)
        exposure_layout.addStretch(1)
        self.outline_path = self._path_row(config.geometry.outline_path, "JSON (*.json);;所有文件 (*.*)")
        self.inline_path = self._path_row(config.geometry.inline_path, "JSON (*.json);;所有文件 (*.*)")
        self.pocket_path = self._path_row(config.geometry.pocket_path, "JSON (*.json);;所有文件 (*.*)")
        self.camera_calibration_file = self._path_row(config.calibration.camera_file, "OpenCV 标定文件 (*.yaml *.yml *.xml);;所有文件 (*.*)")
        self.projection_calibration_file = self._path_row(config.calibration.projection_file, "投影校正文件 (*.json);;所有文件 (*.*)")
        self.detector_backend = QtWidgets.QComboBox()
        self.detector_backend.addItems(["disabled", "ultralytics", "debug_color"])
        self.detector_backend.setCurrentText(config.detector.backend)
        self.detect_interval = self._spin(int(config.detector.detect_interval_frames), 1, 12)
        self.detect_fps_limit = self._dspin(float(config.detector.detect_fps_limit_hz), 0.0, 30.0, 0.5)
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
        proj_size = QtWidgets.QHBoxLayout()
        proj_size.addWidget(self.proj_w)
        proj_size.addWidget(QtWidgets.QLabel("x"))
        proj_size.addWidget(self.proj_h)
        form.addRow("台球模型路径", self.model_path)
        form.addRow("类别文件路径", self.class_file_path)
        form.addRow("学习排序模型", learning_ranker_box)
        form.addRow("学习样本目录", learning_collect_box)
        form.addRow("视频文件路径", self.video_path)
        form.addRow("Nori SDK 目录", self.nori_sdk_root)
        form.addRow("工业相机畸变矫正", distortion_box)
        form.addRow("工业相机曝光", exposure_box)
        form.addRow("检测后端", self.detector_backend)
        form.addRow("检测间隔(帧)", self.detect_interval)
        form.addRow("检测频率上限(Hz)", self.detect_fps_limit)
        form.addRow("outline.json", self.outline_path)
        form.addRow("inline.json", self.inline_path)
        form.addRow("pocket.json", self.pocket_path)
        form.addRow("相机标定文件", self.camera_calibration_file)
        form.addRow("投影校正文件", self.projection_calibration_file)
        form.addRow("默认投影设备", self.proj_screen)
        form.addRow("默认投影分辨率", proj_size)
        tabs.addTab(general, "基础")

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
        layout.addWidget(buttons)

    def apply_to_config(self, config: AppConfig) -> None:
        config.detector.model_path = self.model_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.detector.class_file_path = self.class_file_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.learning.ranker_enabled = self.learning_ranker_enabled.isChecked()
        config.learning.ranker_model_path = self.learning_ranker_model_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.learning.score_blend = float(self.learning_score_blend.value())
        config.learning.collect_enabled = self.learning_collect_enabled.isChecked()
        config.learning.samples_directory = self.learning_samples_directory.line_edit.text().strip() or "rl/data/samples"  # type: ignore[attr-defined]
        config.camera.video_path = self.video_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.camera.nori_sdk_root = self.nori_sdk_root.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.camera.distortion_correction_enabled = self.distortion_enabled.isChecked()
        config.camera.distortion_correction_file = self.distortion_file.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.camera.exposure_auto = self.exposure_auto.isChecked()
        config.camera.exposure_level = int(self.exposure_level.value())
        config.detector.backend = self.detector_backend.currentText()
        config.detector.detect_interval_frames = int(self.detect_interval.value())
        config.detector.detect_fps_limit_hz = float(self.detect_fps_limit.value())
        config.geometry.outline_path = self.outline_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.geometry.inline_path = self.inline_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.geometry.pocket_path = self.pocket_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.calibration.camera_file = self.camera_calibration_file.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.calibration.projection_file = self.projection_calibration_file.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.projection.screen_index = int(self.proj_screen.currentData() or 0)
        config.projection.projector_width = int(self.proj_w.value())
        config.projection.projector_height = int(self.proj_h.value())

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

    def _dspin(self, value: float, lo: float, hi: float, step: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(1)
        spin.setValue(float(value))
        return spin

    def _spin(self, value: int, lo: int, hi: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(int(value))
        return spin

    def _grid_pair(self, grid: QtWidgets.QGridLayout, row: int, label_a: str, widget_a: QtWidgets.QWidget, label_b: str, widget_b: QtWidgets.QWidget) -> None:
        grid.addWidget(QtWidgets.QLabel(label_a), row, 0)
        grid.addWidget(widget_a, row, 1)
        grid.addWidget(QtWidgets.QLabel(label_b), row, 2)
        grid.addWidget(widget_b, row, 3)

    def _move_star(self, dx: int, dy: int) -> None:
        step = int(self.star_move_step.value())
        self.star_offset_x.setValue(self.star_offset_x.value() + dx * step)
        self.star_offset_y.setValue(self.star_offset_y.value() + dy * step)


class ProjectorCalibrationDialog(QtWidgets.QDialog):
    def __init__(self, operator: "OperatorWindow", calibration, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent or operator)
        self.operator = operator
        self.calibration = calibration
        self.setWindowTitle("投影仪校正向导")
        self.resize(840, 680)

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("投影仪校正")
        title.setObjectName("title")
        layout.addWidget(title)

        steps = QtWidgets.QPlainTextEdit()
        steps.setReadOnly(True)
        steps.setPlainText(
            "\n".join(
                [
                    "1. 先确认相机内参 ChArUco 标定文件有效，投影仪分辨率和屏幕编号正确。",
                    "2. 在台面放置主标定板，点击“显示 ChArUco 校正码”或“显示编码网格”，让相机采集 >=20 个不同平移/俯仰姿态。",
                    "3. 在六个袋口、两条长边中段、近投影端、远投影端放置 10-14 张 A6/A7 小 ChArUco 纸板，采集局部精修数据。",
                    "4. 用外部标定脚本生成或替换 projection_calibration.json，再点击“显示当前校正结果”检查桌面多边形和对应点。",
                    "5. 点击“显示残差箭头”检查局部误差方向；近端、远端、袋口区域都应有控制点覆盖。",
                    "6. 最终验收需查看 holdout：图像重投影 mean/P95、台面毫米 median/P95、分区 P95 和误差随距离梯度。",
                ]
            )
        )
        layout.addWidget(steps, 1)

        button_grid = QtWidgets.QGridLayout()
        actions = [
            ("打开投影窗口", self.operator.ensure_projection_window_for_operator),
            ("显示 ChArUco 校正码", self.operator.show_projector_charuco_code),
            ("显示编码网格", self.operator.show_projector_encoded_grid),
            ("显示当前校正结果", lambda: self.operator.show_projector_calibration_result(self.calibration)),
            ("显示残差箭头", lambda: self.operator.show_projector_residual_overlay(self.calibration)),
            ("验证 Holdout", self._verify_holdout),
            ("恢复实时投影", self.operator.resume_runtime_projection),
        ]
        for idx, (text, slot) in enumerate(actions):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            button_grid.addWidget(button, idx // 3, idx % 3)
        layout.addLayout(button_grid)

        self.result = QtWidgets.QLabel(self._summary_text())
        self.result.setObjectName("bestShot")
        self.result.setWordWrap(True)
        layout.addWidget(self.result)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _summary_text(self) -> str:
        projection = self.calibration.projection
        stats = projection.calibration_error_stats()
        lines = [
            f"文件: {projection.source_path or '未设置'}",
            f"模式: {projection.mode} | 投影尺寸: {projection.projector_size[0]}x{projection.projector_size[1]}",
            f"对应点: {projection.cam_points.shape[0]} | 局部残差控制点: {projection.residual_field.control_points_cam.shape[0]}",
            f"桌面多边形点: {projection.table_polygon_proj.shape[0]} | 有效: {'是' if projection.is_valid else '否'}",
        ]
        if stats:
            p95 = stats.get("p95_px", stats.get("max_px", 0.0))
            verdict = "通过预检查" if p95 <= 3.0 else "需要复核/重标定"
            lines.append(
                "像素误差: "
                f"mean={stats.get('mean_px', 0.0):.2f}px, "
                f"median={stats.get('median_px', 0.0):.2f}px, "
                f"p95={p95:.2f}px, max={stats.get('max_px', 0.0):.2f}px | {verdict}"
            )
        else:
            lines.append("像素误差: 当前 JSON 没有可统计的 paired correspondences。")
        lines.append("注意: 像素误差只是预检查，正式验收还需要独立 holdout 的毫米误差、分区误差和距离梯度。")
        return "\n".join(lines)

    def _verify_holdout(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 Holdout JSON", str(Path.cwd()), "JSON (*.json);;所有文件 (*.*)")
        if not path:
            return
        try:
            report = verify_holdout_file(path, self.calibration)
            text = format_holdout_report(report)
            self.result.setText(self._summary_text() + "\n\nHoldout 验证:\n" + text)
            self.operator._append_log("Holdout 验证完成: " + text.replace("\n", " | "))
        except Exception as exc:
            self.operator._append_log(f"Holdout 验证失败: {exc}")
            QtWidgets.QMessageBox.critical(self, "Holdout 验证失败", str(exc))


class OperatorWindow(QtWidgets.QMainWindow):
    BASE_WIDTH = 1420
    BASE_HEIGHT = 860

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.user_settings = UserSettings.load()
        self.star_formula = StarFormulaConfig.from_mapping(self.user_settings.star_formula)
        self.pipeline: Optional[RuntimePipeline] = None
        self._pipeline_start_thread: Optional[QtCore.QThread] = None
        self._pipeline_start_worker: Optional[_PipelineStartWorker] = None
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
        self._raw_video_recorder: Optional[FfmpegH264Recorder] = None
        self._route_video_recorder: Optional[FfmpegH264Recorder] = None
        self._raw_video_path: Optional[Path] = None
        self._route_video_path: Optional[Path] = None

        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self._frame_busy = False

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

        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        self.title = QtWidgets.QLabel("BAS 台球智能辅助")
        self.title.setObjectName("title")
        self.subtitle = QtWidgets.QLabel("实时视觉 / 状态估计 / 路线投影")
        self.subtitle.setObjectName("muted")
        title_box.addWidget(self.title)
        title_box.addWidget(self.subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        self.status_label = QtWidgets.QLabel("离线")
        self.status_label.setObjectName("status")
        header.addWidget(self.status_label)
        self.outer_layout.addLayout(header)

        self.main_layout = QtWidgets.QHBoxLayout()
        self.outer_layout.addLayout(self.main_layout, 1)

        self.sidebar = self._panel()
        self.side_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.capture_btn = self._button("开始采集")
        self.projection_btn = self._button("开始投影")
        self.init_module_btn = self._button("初始化图形图像模块")
        self.projector_calib_btn = self._button("校正投影仪")
        self.settings_btn = self._button("设置")
        self.probe_btn = self._button("探测相机")
        self.export_diag_btn = self._button("导出诊断快照")
        self.capture_btn.clicked.connect(self.toggle_capture)
        self.projection_btn.clicked.connect(self.toggle_projection_window)
        self.init_module_btn.clicked.connect(self.initialize_graphics_image_module)
        self.projector_calib_btn.clicked.connect(self.calibrate_projector)
        self.settings_btn.clicked.connect(self.open_settings)
        self.probe_btn.clicked.connect(self.probe_camera_devices)
        self.export_diag_btn.clicked.connect(self.export_diagnostic_snapshot)
        for btn in [self.capture_btn, self.projection_btn, self.init_module_btn, self.projector_calib_btn, self.settings_btn, self.probe_btn, self.export_diag_btn]:
            self.side_layout.addWidget(btn)
        self.side_layout.addSpacing(6)

        self.side_layout.addWidget(self._section_label("采集"))
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(["auto", "nori", "opencv", "video", "synthetic"])
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
        self.side_layout.addWidget(self._field("输入类型", self.backend_combo))
        self.side_layout.addWidget(self._field("工业相机", self.nori_device_combo))
        self.side_layout.addWidget(self._field("OpenCV 设备", self.device_spin))
        self.side_layout.addWidget(self._field("分辨率", self.resolution_combo))
        self.side_layout.addWidget(self._field("帧率", self.fps_combo))

        self.replay_check = QtWidgets.QCheckBox("记录回放")
        self.replay_check.setChecked(self.config.replay.enabled)
        self.side_layout.addWidget(self.replay_check)
        self.geometry_reference_check = QtWidgets.QCheckBox("显示几何参考线")
        self.geometry_reference_check.setChecked(bool(self.config.projection.geometry_reference_enabled))
        self.geometry_reference_check.toggled.connect(self._geometry_reference_toggled)
        self.side_layout.addWidget(self.geometry_reference_check)
        self.shot_mode_combo = QtWidgets.QComboBox()
        self.shot_mode_combo.addItem("规则模式", "rule")
        self.shot_mode_combo.addItem("自由模式", "free")
        self.shot_mode_combo.currentIndexChanged.connect(self._shot_mode_changed)
        self.side_layout.addWidget(self._field("画线模式", self.shot_mode_combo))
        self.side_layout.addWidget(self._section_label("现场抓取"))
        self.raw_photo_btn = self._button("抓取无画线照片")
        self.raw_video_btn = self._button("开始无画线视频")
        self.route_video_btn = self._button("开始进洞路线视频")
        self.raw_photo_btn.clicked.connect(self.capture_raw_photo)
        self.raw_video_btn.clicked.connect(self.toggle_raw_video_recording)
        self.route_video_btn.clicked.connect(self.toggle_route_video_recording)
        for btn in [self.raw_photo_btn, self.raw_video_btn, self.route_video_btn]:
            self.side_layout.addWidget(btn)
        self.side_layout.addStretch(1)
        self.config_label = QtWidgets.QLabel("设置保存在 local_settings/user_settings.json")
        self.config_label.setObjectName("muted")
        self.config_label.setWordWrap(True)
        self.side_layout.addWidget(self.config_label)
        self.main_layout.addWidget(self.sidebar)

        self.center_layout = QtWidgets.QVBoxLayout()
        self.preview_panel = self._panel()
        self.preview_layout = QtWidgets.QVBoxLayout(self.preview_panel)
        self.preview_label = QtWidgets.QLabel("等待画面")
        self.preview_label.setObjectName("preview")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_layout.addWidget(self.preview_label, 1)
        self.center_layout.addWidget(self.preview_panel, 1)
        self.metric_layout = QtWidgets.QHBoxLayout()
        self.fps_metric = self._metric("FPS", "--")
        self.det_metric = self._metric("检测", "--")
        self.track_metric = self._metric("跟踪", "--")
        self.phase_metric = self._metric("状态", "--")
        for item in [self.fps_metric, self.det_metric, self.track_metric, self.phase_metric]:
            self.metric_layout.addWidget(item)
        self.center_layout.addLayout(self.metric_layout)
        self.main_layout.addLayout(self.center_layout, 1)

        self.right_panel = self._panel()
        self.right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        self.right_layout.addWidget(self._section_label("模块状态"))
        self.module_status = QtWidgets.QTableWidget(0, 3)
        self.module_status.setHorizontalHeaderLabels(["模块", "状态", "细节"])
        self.module_status.horizontalHeader().setStretchLastSection(True)
        self.module_status.verticalHeader().setVisible(False)
        self.module_status.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.module_status.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.right_layout.addWidget(self.module_status)
        self._module_rows = {}
        for name in ["采集", "检测", "跟踪", "状态机", "规划", "投影", "回放", "抓取", "标定"]:
            self._add_module_row(name)

        self.right_layout.addWidget(self._section_label("状态机人工介入"))
        self.manual_phase_combo = QtWidgets.QComboBox()
        self.manual_phase_combo.addItems([phase.value for phase in MatchPhase])
        self.force_phase_btn = self._button("强制状态")
        self.hold_state_btn = self._button("冻结状态机")
        self.snapshot_state_btn = self._button("确认当前布局稳定")
        self.reset_state_btn = self._button("重置状态机")
        self.clear_review_btn = self._button("清除复核标记")
        self.force_phase_btn.clicked.connect(self.force_state_phase)
        self.hold_state_btn.clicked.connect(self.toggle_state_hold)
        self.snapshot_state_btn.clicked.connect(self.snapshot_stable_layout)
        self.reset_state_btn.clicked.connect(self.reset_state_machine)
        self.clear_review_btn.clicked.connect(self.clear_state_review_flags)
        self.right_layout.addWidget(self._field("目标状态", self.manual_phase_combo))
        manual_grid = QtWidgets.QGridLayout()
        for idx, button in enumerate([self.force_phase_btn, self.hold_state_btn, self.snapshot_state_btn, self.reset_state_btn, self.clear_review_btn]):
            manual_grid.addWidget(button, idx // 2, idx % 2)
        self.right_layout.addLayout(manual_grid)

        self.right_layout.addWidget(self._section_label("最佳路线"))
        self.best_label = QtWidgets.QLabel("无")
        self.best_label.setObjectName("bestShot")
        self.best_label.setWordWrap(True)
        self.right_layout.addWidget(self.best_label)
        self.right_layout.addWidget(self._section_label("候选"))
        self.candidates = QtWidgets.QTableWidget(0, 4)
        self.candidates.setHorizontalHeaderLabels(["ID", "袋口", "评分", "风险"])
        self.candidates.horizontalHeader().setStretchLastSection(True)
        self.candidates.verticalHeader().setVisible(False)
        self.candidates.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.candidates.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.right_layout.addWidget(self.candidates, 1)
        self.right_layout.addWidget(self._section_label("事件"))
        self.event_list = QtWidgets.QListWidget()
        self.right_layout.addWidget(self.event_list, 1)
        self.main_layout.addWidget(self.right_panel)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setObjectName("logBox")
        self.log_box.setReadOnly(True)
        self.outer_layout.addWidget(self.log_box)

    def _panel(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("panel")
        return frame

    def _section_label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("section")
        return label

    def _button(self, text: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setCursor(QtCore.Qt.PointingHandCursor)
        return button

    def _field(self, label_text: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel(label_text)
        label.setObjectName("muted")
        layout.addWidget(label)
        layout.addWidget(widget)
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
        self.main_layout.setSpacing(px(12))
        self.side_layout.setContentsMargins(px(14), px(14), px(14), px(14))
        self.side_layout.setSpacing(px(10))
        self.center_layout.setSpacing(px(12))
        self.preview_layout.setContentsMargins(px(8), px(8), px(8), px(8))
        self.metric_layout.setSpacing(px(10))
        self.right_layout.setContentsMargins(px(14), px(14), px(14), px(14))
        self.right_layout.setSpacing(px(10))
        self.sidebar.setFixedWidth(px(275))
        self.right_panel.setFixedWidth(px(390))
        self.preview_label.setMinimumSize(px(620), px(390))
        self.log_box.setMaximumHeight(px(130))
        self.module_status.setMaximumHeight(px(215))
        self.module_status.verticalHeader().setDefaultSectionSize(px(22))
        self.candidates.verticalHeader().setDefaultSectionSize(px(30))
        for metric in [self.fps_metric, self.det_metric, self.track_metric, self.phase_metric]:
            metric.metric_layout.setContentsMargins(px(12), px(9), px(12), px(9))  # type: ignore[attr-defined]
            metric.metric_layout.setSpacing(px(2))  # type: ignore[attr-defined]
        self.setStyleSheet(self._stylesheet(scale))
        self._refresh_preview_pixmap()

    def _stylesheet(self, scale: float) -> str:
        def px(value: float) -> int:
            return max(1, int(round(value * scale)))

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
        QFrame#panel, QFrame#metric {{
            background: #181818;
            border: 1px solid #343434;
            border-radius: {px(2)}px;
        }}
        QLabel#preview {{
            background: #050505;
            border: 1px solid #3a3a3a;
            color: #cfcfcf;
            font-size: {px(16)}px;
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
        QComboBox, QSpinBox, QDoubleSpinBox {{
            background: #111111;
            border: 1px solid #444444;
            border-radius: {px(2)}px;
            padding: {px(6)}px;
            min-height: {px(25)}px;
            color: #ffffff;
        }}
        QCheckBox {{ color: #ffffff; spacing: {px(8)}px; }}
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
        mode = str(getattr(self.config.planner, "shot_mode", "rule") or "rule").strip().lower()
        idx = self.shot_mode_combo.findData("free" if mode in {"free", "free_shot"} else "rule")
        self.shot_mode_combo.blockSignals(True)
        self.shot_mode_combo.setCurrentIndex(max(0, idx))
        self.shot_mode_combo.blockSignals(False)
        self.geometry_reference_check.blockSignals(True)
        self.geometry_reference_check.setChecked(bool(self.config.projection.geometry_reference_enabled))
        self.geometry_reference_check.blockSignals(False)

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
        self.config.planner.shot_mode = str(self.shot_mode_combo.currentData() or "rule")

    @QtCore.pyqtSlot(int)
    def _shot_mode_changed(self, _index: int = 0) -> None:
        mode = str(self.shot_mode_combo.currentData() or "rule")
        self.config.planner.shot_mode = mode
        if self.pipeline is not None:
            self.pipeline.config.planner.shot_mode = mode
            self.pipeline.planner.config.shot_mode = mode
            self.pipeline._last_plan = None
            self.pipeline._last_overlay = None
        self._save_user_settings()
        self._append_log(f"画线模式已切换为 {'自由模式' if mode == 'free' else '规则模式'}")
        self._update_module_status(self.last_output)

    @QtCore.pyqtSlot(bool)
    def _geometry_reference_toggled(self, checked: bool) -> None:
        self.config.projection.geometry_reference_enabled = bool(checked)
        self._save_user_settings()
        if self.last_output is not None:
            self._update_preview(self.last_output)
        self._append_log("几何参考线已开启" if checked else "几何参考线已关闭")

    def _save_user_settings(self) -> None:
        UserSettings.from_config(self.config, self.star_formula.to_dict()).save()

    def _pipeline_restart_signature(self) -> tuple:
        return (
            self.config.camera.backend,
            self.config.camera.device_index,
            self.config.camera.nori_device_id,
            self.config.camera.width,
            self.config.camera.height,
            self.config.camera.fps,
            self.config.camera.video_path,
            self.config.camera.nori_sdk_root,
            self.config.camera.exposure_auto,
            self.config.camera.exposure_level,
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
            self.config.calibration.camera_file,
            self.config.calibration.projection_file,
            self.config.calibration.table_width_mm,
            self.config.calibration.table_height_mm,
            self.config.calibration.ball_diameter_mm,
            self.config.geometry.outline_path,
            self.config.geometry.inline_path,
            self.config.geometry.pocket_path,
            self.config.planner.enabled,
            self.config.planner.shot_mode,
            self.config.planner.max_cut_angle_deg,
            self.config.planner.top_k,
            self.config.planner.cue_path_margin_mm,
            self.config.planner.object_path_margin_mm,
            self.config.planner.collision_padding_mm,
            self.config.planner.free_max_collisions,
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
        self.hold_state_btn.setText("冻结状态机")
        self.event_list.clear()
        self._append_log("状态机已重置")
        self._update_module_status(self.last_output)

    @QtCore.pyqtSlot()
    def clear_state_review_flags(self) -> None:
        sm = self._state_machine_or_none()
        if sm is None:
            return
        frame_id, ts_cam_ns = self._last_frame_marker()
        sm.clear_review_flags(frame_id=frame_id, ts_cam_ns=ts_cam_ns)
        self._append_log("已清除进洞/异常复核标记")
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
            self._append_log("当前没有可抓取的原始画面")
            return
        out_dir = self._capture_output_dir()
        frame_id = self.last_output.frame.frame_id if self.last_output is not None else 0
        path = out_dir / f"no_line_photo_{time.strftime('%Y%m%d_%H%M%S')}_f{frame_id:06d}.jpg"
        ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            self._append_log(f"无画线照片保存失败: {path}")
            return
        self._append_log(f"无画线照片已保存: {path}")

    @QtCore.pyqtSlot()
    def toggle_raw_video_recording(self) -> None:
        if self._raw_video_recorder is None:
            self._start_video_recording(route=False)
        else:
            self._stop_video_recording(route=False)

    @QtCore.pyqtSlot()
    def toggle_route_video_recording(self) -> None:
        if self._route_video_recorder is None:
            self._start_video_recording(route=True)
        else:
            self._stop_video_recording(route=True)

    def _start_video_recording(self, *, route: bool) -> None:
        if self.pipeline is None:
            self._append_log("请先开始采集再录制视频")
            return
        frame = self._current_raw_frame()
        if frame is None:
            self._append_log("请等待第一帧画面后再录制视频")
            return
        h, w = frame.shape[:2]
        prefix = "pocket_route_video" if route else "no_line_video"
        path = self._capture_output_dir() / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        record_fps = self._recording_fps()
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
            return
        except Exception as exc:
            self._append_log(f"启动录制失败: {exc}")
            return
        if route:
            self._route_video_recorder = recorder
            self._route_video_path = path
            self.route_video_btn.setText("停止进洞路线视频")
            if self.last_output is not None and self.last_output.plan.best is None:
                self._append_log("当前无最佳进洞路线，视频会在路线出现后开始画线")
            self._append_log(f"进洞路线画线视频开始录制: {path} ({record_fps:.1f}fps, H.264 6000kbps)")
        else:
            self._raw_video_recorder = recorder
            self._raw_video_path = path
            self.raw_video_btn.setText("停止无画线视频")
            self._append_log(f"无画线视频开始录制: {path} ({record_fps:.1f}fps, H.264 6000kbps)")
        self._update_module_status(self.last_output)

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
            self.route_video_btn.setText("开始进洞路线视频")
        else:
            self._raw_video_recorder = None
            self.raw_video_btn.setText("开始无画线视频")
        label = "进洞路线画线视频" if route else "无画线视频"
        if code == 0:
            self._append_log(f"{label}已保存: {path} ({frames} 帧, H.264 6000kbps)")
        else:
            self._append_log(f"{label}保存可能不完整: ffmpeg code={code}, path={path}")
        self._update_module_status(self.last_output)

    def _stop_all_media_recordings(self) -> None:
        if self._raw_video_recorder is not None:
            self._stop_video_recording(route=False)
        if self._route_video_recorder is not None:
            self._stop_video_recording(route=True)

    def _capture_output_dir(self) -> Path:
        path = Path("local_settings") / "captures"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _current_raw_frame(self) -> Optional[np.ndarray]:
        if self.last_output is None:
            return None
        frame = self.last_output.frame.image
        return frame if frame is not None else None

    def _recording_fps(self) -> float:
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
        if self.started_at > 0.0 and self.frame_count >= 5:
            actual = self.frame_count / max(1e-6, time.perf_counter() - self.started_at)
            if 1.0 <= actual < fps * 0.9:
                fps = actual
        return max(1.0, min(240.0, fps))

    def _record_media_frames(self, out: PipelineOutput) -> None:
        if self._raw_video_recorder is not None and out.frame.image is not None:
            self._write_video_frame(route=False, frame=out.frame.image)
        if self._route_video_recorder is not None and out.frame.image is not None:
            self._write_video_frame(route=True, frame=self._route_capture_frame(out))

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
                self.route_video_btn.setText("开始进洞路线视频")
            else:
                self._raw_video_recorder = None
                self.raw_video_btn.setText("开始无画线视频")
            self._update_module_status(self.last_output)

    def _route_capture_frame(self, out: PipelineOutput) -> np.ndarray:
        frame = out.frame.image
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

    def _draw_rule_plan_preview(self, image: np.ndarray, candidate) -> None:
        route_color = (255, 255, 255)
        radius_mm = 0.5 * float(self.pipeline.calibration.table.ball_diameter_mm) if self.pipeline is not None else 28.0
        cue = np.asarray(candidate.cue_ball, dtype=np.float32)
        ghost = np.asarray(candidate.ghost_ball, dtype=np.float32)
        target = np.asarray(candidate.object_ball, dtype=np.float32)
        pocket = np.asarray(candidate.pocket_point, dtype=np.float32)
        thickness = max(2, int(round(3 * self._ui_scale)))
        radius_px = max(6, int(round(self._camera_radius_px(ghost, radius_mm))))

        guide_start = self._cue_alignment_start_table(cue, ghost)
        self._draw_segment_trimmed(image, guide_start, cue, route_color, thickness, 0.0, radius_mm)
        self._draw_segment_trimmed(image, cue, ghost, route_color, thickness + 1, radius_mm, radius_mm)
        self._draw_dashed_segment_trimmed(image, target, pocket, route_color, thickness, radius_mm, 0.0)
        cue_sep_end = self._rule_cue_separation_end_table(cue, ghost, target)
        if cue_sep_end is not None:
            self._draw_dashed_segment_trimmed(image, ghost, cue_sep_end, route_color, thickness, radius_mm, 0.0)

        for point in (ghost, target):
            cam = self._table_mm_to_camera_px([point])
            if cam.shape[0] >= 1:
                cv2.circle(image, _point_int(cam[0]), radius_px, route_color, 2, cv2.LINE_AA)
        pocket_px = self._table_mm_to_camera_px([pocket])
        if pocket_px.shape[0] >= 1:
            cv2.circle(image, _point_int(pocket_px[0]), max(6, radius_px // 2), route_color, 2, cv2.LINE_AA)

    def _draw_free_plan_preview(self, image: np.ndarray, route) -> None:
        route_color = (255, 255, 255)
        cue_stick_color = route_color
        radius_mm = float(max(1.0, route.cue_radius))
        cue = np.asarray(route.cue_ball, dtype=np.float32)
        tip = np.asarray(route.cue_stick_tip, dtype=np.float32)
        tail = np.asarray(route.cue_stick_tail, dtype=np.float32)
        aim_dir = unit(np.asarray(route.aim_direction, dtype=np.float32))
        thickness = max(2, int(round(3 * self._ui_scale)))
        radius_px = max(6, int(round(self._camera_radius_px(cue, radius_mm))))

        stick_px = self._table_mm_to_camera_px([tail, tip])
        if stick_px.shape[0] >= 2:
            cv2.line(image, _point_int(stick_px[0]), _point_int(stick_px[1]), cue_stick_color, 2, cv2.LINE_AA)

        guide_back = cue - aim_dir * max(26.0, 2.2 * radius_mm)
        self._draw_segment_trimmed(image, guide_back, cue, route_color, thickness, 0.0, radius_mm)

        nodes = [np.asarray(p, dtype=np.float32) for p in route.path_points or []]
        collision_count = len(route.collision_points or [])
        for i in range(max(0, len(nodes) - 1)):
            start_radius = radius_mm if i <= collision_count else 0.0
            end_radius = radius_mm if (i + 1) <= collision_count else 0.0
            self._draw_segment_trimmed(image, nodes[i], nodes[i + 1], route_color, thickness + 1, start_radius, end_radius)

        collisions = [np.asarray(p, dtype=np.float32) for p in route.collision_points or []]
        normals = [np.asarray(n, dtype=np.float32) for n in route.collision_normals or []]
        collision_types = list(route.collision_types or [])
        for idx, collision in enumerate(collisions):
            cam = self._table_mm_to_camera_px([collision])
            if cam.shape[0] >= 1:
                cv2.circle(image, _point_int(cam[0]), radius_px, route_color, 2, cv2.LINE_AA)
            if idx >= len(collision_types) or str(collision_types[idx]) != "ball":
                continue
            normal = unit(normals[idx] if idx < len(normals) else aim_dir)
            hit_center = collision + normal * (2.0 * radius_mm)
            hit_end = self._estimate_route_end_table(hit_center, normal)
            self._draw_dashed_segment_trimmed(image, hit_center, hit_end, route_color, thickness, radius_mm, 0.0)

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
                cv2.line(image, _point_int(p1), _point_int(p2), color, max(1, int(thickness) - 1), cv2.LINE_AA)
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

    def _cue_alignment_start_table(self, cue: np.ndarray, ghost: np.ndarray) -> np.ndarray:
        aim = ghost - cue
        if float(np.linalg.norm(aim)) < 1e-6:
            return cue.copy()
        if self.pipeline is None:
            return cue.copy()
        table = self.pipeline.calibration.table
        max_length = 0.55 * float(np.hypot(table.width_mm, table.height_mm))
        return self._trace_ray_inside_table(cue, -unit(aim), max_length=max_length, step_mm=12.0)

    def _trace_ray_inside_table(self, origin: np.ndarray, direction: np.ndarray, max_length: float, step_mm: float) -> np.ndarray:
        inner = self._inner_polygon_table()
        if inner.shape[0] < 3:
            return (origin + unit(direction) * float(max_length)).astype(np.float32)
        best = origin.copy().astype(np.float32)
        d = unit(direction)
        for dist in np.arange(float(step_mm), float(max_length) + float(step_mm), float(step_mm)):
            p = origin + d * float(dist)
            if cv2.pointPolygonTest(inner.reshape((-1, 1, 2)).astype(np.float32), (float(p[0]), float(p[1])), False) < 0.0:
                break
            best = p.astype(np.float32)
        return best

    def _rule_cue_separation_end_table(self, cue: np.ndarray, ghost: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
        incoming = ghost - cue
        object_dir = target - ghost
        if float(np.linalg.norm(incoming)) < 1e-6 or float(np.linalg.norm(object_dir)) < 1e-6:
            return None
        incoming_u = unit(incoming)
        object_u = unit(object_dir)
        cue_after = incoming_u - object_u * float(np.dot(incoming_u, object_u))
        if float(np.linalg.norm(cue_after)) < 1e-3:
            return None
        return self._estimate_route_end_table(ghost, unit(cue_after))

    def _estimate_route_end_table(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        inner = self._inner_polygon_table()
        hit = self._ray_polygon_first_hit_table(origin, direction, inner, min_t=1.0)
        if hit is not None:
            return hit.astype(np.float32)
        if self.pipeline is None:
            return (origin + unit(direction) * 600.0).astype(np.float32)
        table = self.pipeline.calibration.table
        fallback_len = float(np.hypot(table.width_mm, table.height_mm))
        return (origin + unit(direction) * max(120.0, fallback_len)).astype(np.float32)

    def _inner_polygon_table(self) -> np.ndarray:
        if self.pipeline is None:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(self.pipeline.calibration.table.inner_polygon_mm, dtype=np.float32).reshape((-1, 2))

    @staticmethod
    def _ray_polygon_first_hit_table(origin: np.ndarray, direction: np.ndarray, polygon: np.ndarray, min_t: float) -> Optional[np.ndarray]:
        if polygon.size < 6 or polygon.shape[0] < 3:
            return None
        d = unit(direction)
        best_t: Optional[float] = None
        best_hit: Optional[np.ndarray] = None
        for idx in range(polygon.shape[0]):
            a = polygon[idx].astype(np.float32)
            b = polygon[(idx + 1) % polygon.shape[0]].astype(np.float32)
            s = b - a
            denom = float(d[0] * s[1] - d[1] * s[0])
            if abs(denom) < 1e-7:
                continue
            ao = a - origin
            t = float((ao[0] * s[1] - ao[1] * s[0]) / denom)
            u = float((ao[0] * d[1] - ao[1] * d[0]) / denom)
            if t < min_t or u < -1e-4 or u > 1.0001:
                continue
            if best_t is None or t < best_t:
                best_t = t
                best_hit = origin + d * t
        return best_hit

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
        self.projection_btn.setText("停止投影")
        self._append_log("投影窗口已打开，当前处于校正模式")
        self._update_module_status(self.last_output)

    def resume_runtime_projection(self) -> None:
        self._projection_calibration_mode = False
        self._refresh_projection()
        self._append_log("投影已恢复实时 overlay")
        self._update_module_status(self.last_output)

    def show_projector_charuco_code(self) -> None:
        self.ensure_projection_window_for_operator()
        used_fallback = False
        try:
            image = render_charuco_board(
                CharucoBoardSpec(squares_x=10, squares_y=7, square_length_m=0.035, marker_length_m=0.026),
                int(self.config.projection.projector_width),
                int(self.config.projection.projector_height),
            )
        except Exception as exc:
            image = self._encoded_grid_image()
            used_fallback = True
            self._append_log(f"ChArUco 生成失败，已改用编码网格: {exc}")
        if self.projection_window is not None:
            self.projection_window.set_image(image)
        self._append_log("已显示编码网格" if used_fallback else "已显示全屏 ChArUco 校正码")

    def show_projector_encoded_grid(self) -> None:
        self.ensure_projection_window_for_operator()
        image = self._encoded_grid_image()
        if self.projection_window is not None:
            self.projection_window.set_image(image)
        self._append_log("已显示编码网格/定位十字")

    def show_projector_calibration_result(self, calibration=None) -> None:
        self.ensure_projection_window_for_operator()
        calibration = calibration or create_calibration_service(
            self.config.calibration,
            frame_undistorted=bool(self.config.camera.distortion_correction_enabled),
        )
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
        calibration = calibration or create_calibration_service(
            self.config.calibration,
            frame_undistorted=bool(self.config.camera.distortion_correction_enabled),
        )
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
                overlay.circles.append((end, 5.0, (255, 255, 255)))
            self._append_log(f"已显示 {min(80, controls.shape[0])} 个局部残差箭头")
        else:
            self._append_log("当前校正文件没有局部残差控制点，已显示基础校正结果")
        if self.projection_window is not None:
            self.projection_window.set_overlay(overlay)

    def _encoded_grid_image(self) -> np.ndarray:
        w = int(self.config.projection.projector_width)
        h = int(self.config.projection.projector_height)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (8, 8, 8)
        step_x = max(40, w // 16)
        step_y = max(40, h // 10)
        for x in range(0, w + 1, step_x):
            color = (255, 255, 255)
            cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
            cv2.putText(img, str(x), (min(w - 70, x + 4), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        for y in range(0, h + 1, step_y):
            color = (255, 255, 255)
            cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
            cv2.putText(img, str(y), (8, min(h - 10, y + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        for x, y, label in [(w // 2, h // 2, "C"), (40, 40, "LT"), (w - 40, 40, "RT"), (w - 40, h - 40, "RB"), (40, h - 40, "LB")]:
            cv2.drawMarker(img, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 34, 2, cv2.LINE_AA)
            cv2.putText(img, label, (x + 12, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def _update_module_status(self, out: Optional[PipelineOutput] = None) -> None:
        projection_state = "校正模式" if self._projection_calibration_mode else ("运行中" if self.projection_window is not None else "关闭")
        self._set_module_status("投影", projection_state, f"{self.config.projection.projector_width}x{self.config.projection.projector_height}")
        self._set_module_status("回放", "启用" if self.config.replay.enabled else "关闭", self.config.replay.directory)
        capture_recording = self._raw_video_recorder is not None or self._route_video_recorder is not None
        self._set_module_status("抓取", "录制中" if capture_recording else "待机", self._media_capture_status_detail())
        if self.pipeline is None:
            self._set_module_status("采集", "离线", self.config.camera.backend)
            self._set_module_status("检测", "待机", self.config.detector.backend)
            self._set_module_status("跟踪", "待机", "TemporalTracker")
            self._set_module_status("状态机", "待机", "自动")
            self._set_module_status("规划", "待机", "GeometryPhysics")
            self._set_module_status("标定", "未加载", self.config.calibration.projection_file or "未设置")
            self.hold_state_btn.setText("冻结状态机")
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
        state_hold = "冻结" if self.pipeline.state_machine.operator_hold else "自动"
        state_detail = state_hold if out is None else f"{out.state.phase} / conf {out.state.confidence:.2f}"
        self._set_module_status("状态机", state_hold, state_detail)
        ranker_version = getattr(self.pipeline.planner.learning_ranker, "version", "learning_unknown")
        mode_name = "free" if str(self.config.planner.shot_mode).lower() == "free" else "rule"
        if out is None:
            plan_detail = f"{mode_name} / {self.pipeline.planner.version} / {ranker_version}"
        elif out.plan.shot_mode == "free":
            collision_count = len(out.plan.free_route.collision_points) if out.plan.free_route is not None else 0
            plan_detail = f"free / {collision_count} collisions / {out.plan.free_status}"
        else:
            plan_detail = f"rule / {len(out.plan.candidates)} candidates / {ranker_version}"
        self._set_module_status("规划", "运行中" if self.config.planner.enabled else "关闭", plan_detail)
        if self.pipeline.recorder is not None:
            self._set_module_status("回放", "记录中", self.pipeline.recorder.session_id)
        self.hold_state_btn.setText("恢复自动状态机" if self.pipeline.state_machine.operator_hold else "冻结状态机")

    def _media_capture_status_detail(self) -> str:
        parts = []
        if self._raw_video_recorder is not None:
            parts.append(f"无画线 {self._raw_video_recorder.frames_written}帧")
        if self._route_video_recorder is not None:
            parts.append(f"路线 {self._route_video_recorder.frames_written}帧")
        return "; ".join(parts) if parts else "local_settings/captures"

    @QtCore.pyqtSlot()
    def initialize_graphics_image_module(self) -> None:
        self._sync_config_from_controls()
        self._save_user_settings()
        self._append_log("正在初始化图形图像模块")
        try:
            calibration = create_calibration_service(
                self.config.calibration,
                frame_undistorted=bool(self.config.camera.distortion_correction_enabled),
            )
            geometry = TableGeometryLoader.load_optional(
                self.config.geometry.outline_path,
                self.config.geometry.inline_path,
                self.config.geometry.pocket_path,
            )
            detector = create_detector(self.config.detector)
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
            if self.config.projection.geometry_reference_enabled and not geometry.is_empty:
                self._append_log("几何参考线将在前端实时画面中显示，不会输出到投影仪")
        except Exception as exc:
            self._append_log(f"图形图像模块初始化失败: {exc}")
            QtWidgets.QMessageBox.critical(self, "初始化失败", str(exc))

    @QtCore.pyqtSlot()
    def calibrate_projector(self) -> None:
        self._sync_config_from_controls()
        self._save_user_settings()
        self._append_log("正在打开投影仪校正向导")
        try:
            calibration = create_calibration_service(
                self.config.calibration,
                frame_undistorted=bool(self.config.camera.distortion_correction_enabled),
            )
            if calibration.projection.is_valid:
                self.show_projector_calibration_result(calibration)
            else:
                self.show_projector_encoded_grid()
                self._append_log(f"投影校正文件无效或缺失: {self.config.calibration.projection_file}")
            dialog = ProjectorCalibrationDialog(self, calibration, self)
            dialog.exec_()
        except Exception as exc:
            self._append_log(f"投影仪校正失败: {exc}")
            QtWidgets.QMessageBox.critical(self, "投影仪校正失败", str(exc))

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
            overlay.circles.append(((x, y), 7.0, (255, 255, 255)))
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

    def start_pipeline(self) -> None:
        if self._pipeline_start_thread is not None:
            self._append_log("采集正在启动中")
            return
        self._sync_config_from_controls()
        self._save_user_settings()
        self._append_log("启动采集中")
        self._set_starting(True)
        thread = QtCore.QThread(self)
        worker = _PipelineStartWorker(self.config, self.star_formula)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._pipeline_started)
        worker.failed.connect(self._pipeline_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_pipeline_start_worker)
        self._pipeline_start_thread = thread
        self._pipeline_start_worker = worker
        thread.start()

    @QtCore.pyqtSlot(object)
    def _pipeline_started(self, pipeline: RuntimePipeline) -> None:
        self.pipeline = pipeline
        self.last_output = None
        self.frame_count = 0
        self.started_at = time.perf_counter()
        self._frame_busy = False
        self._last_preview_update_ts = 0.0
        self._last_metrics_update_ts = 0.0
        self._last_heavy_ui_update_ts = 0.0
        self._last_perf_log_ts = 0.0
        self._set_running(True)
        self._update_module_status()
        self._append_log("采集已启动")
        self.timer.start(self._capture_timer_interval_ms())

    @QtCore.pyqtSlot(str)
    def _pipeline_failed(self, message: str) -> None:
        self.pipeline = None
        self._set_running(False)
        self._append_log(f"启动失败: {message}")
        QtWidgets.QMessageBox.critical(self, "启动失败", message)

    @QtCore.pyqtSlot()
    def _clear_pipeline_start_worker(self) -> None:
        self._pipeline_start_thread = None
        self._pipeline_start_worker = None

    def stop_pipeline(self) -> None:
        self.timer.stop()
        self._frame_busy = False
        self._stop_all_media_recordings()
        if self.pipeline is not None:
            try:
                self.pipeline.close()
            finally:
                self.pipeline = None
        self._set_running(False)
        self._update_module_status()
        self._append_log("采集已停止")

    @QtCore.pyqtSlot()
    def open_settings(self) -> None:
        pipeline_signature_before = self._pipeline_restart_signature()
        projection_signature_before = self._projection_signature()
        dialog = SettingsDialog(self.config, self.star_formula, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        dialog.apply_to_config(self.config)
        self.star_formula = dialog.star_formula_config()
        self._sync_controls_from_config()
        self._save_user_settings()
        pipeline_changed = pipeline_signature_before != self._pipeline_restart_signature()
        projection_changed = projection_signature_before != self._projection_signature()
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
        elif self._pipeline_start_thread is not None and pipeline_changed:
            self._append_log("运行参数已变化，将在本次启动完成后下次采集生效")

    @QtCore.pyqtSlot()
    def probe_camera_devices(self) -> None:
        self._append_log("正在探测相机")
        self.nori_device_combo.clear()
        self.nori_device_combo.addItem("默认", None)
        try:
            rows = probe_cameras(max_index=12, nori_sdk_root=self.config.camera.nori_sdk_root)
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
            self.projection_btn.setText("开始投影")
            self._append_log("投影已停止")
        self._update_module_status(self.last_output)

    def _ensure_projection_window(self) -> None:
        if self.projection_window is not None:
            return
        self.projection_window = ProjectionWindow(self.config.projection)
        self.projection_window.set_star_formula(self.star_formula)
        self.projection_window.show_on_configured_screen()
        self._refresh_projection()

    def _refresh_projection(self) -> None:
        if self.projection_window is None:
            return
        if self._projection_calibration_mode:
            return
        if self.last_output is not None:
            self.projection_window.set_overlay(self.last_output.overlay)
        else:
            overlay = ProjectionOverlay(
                overlay_id="blank",
                frame_id=0,
                projector_size=(self.config.projection.projector_width, self.config.projection.projector_height),
            )
            self.projection_window.set_overlay(overlay)

    def _tick(self) -> None:
        tick_start = time.perf_counter()
        if self._frame_busy:
            self.timer.start(self._capture_timer_interval_ms())
            return
        if self.pipeline is None:
            return
        self._frame_busy = True
        try:
            out = self.pipeline.step()
            if out is None:
                self.stop_pipeline()
                return
            self.last_output = out
            self.frame_count += 1
            now = time.perf_counter()
            self._record_media_frames(out)
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
                self._last_heavy_ui_update_ts = now
            self._log_performance_periodically(now)
            self._refresh_projection()
        finally:
            self._frame_busy = False
            if self.pipeline is not None:
                elapsed_ms = (time.perf_counter() - tick_start) * 1000.0
                self.timer.start(self._capture_timer_interval_ms(elapsed_ms=elapsed_ms))

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
        self._draw_geometry_reference_preview(img)
        self._draw_plan_preview(img, out)
        self._set_preview_image(img)

    def _set_preview_image(self, image_bgr: np.ndarray) -> None:
        self._last_preview_bgr = image_bgr.copy()
        self._refresh_preview_pixmap()

    def _refresh_preview_pixmap(self) -> None:
        if self._last_preview_bgr is None:
            return
        image = self._last_preview_bgr
        target = self.preview_label.size()
        target_w = max(1, int(target.width()))
        target_h = max(1, int(target.height()))
        h0, w0 = image.shape[:2]
        scale = min(target_w / max(1, w0), target_h / max(1, h0), 1.0)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self.preview_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _update_stats(self, out: PipelineOutput) -> None:
        elapsed = max(1e-6, time.perf_counter() - self.started_at)
        self.fps_metric.value_label.setText(f"{self.frame_count / elapsed:.1f}")  # type: ignore[attr-defined]
        self.det_metric.value_label.setText(str(len(out.detections.detections)))  # type: ignore[attr-defined]
        self.track_metric.value_label.setText(str(len(out.tracks.tracks)))  # type: ignore[attr-defined]
        self.phase_metric.value_label.setText(out.state.phase)  # type: ignore[attr-defined]

    def _update_plan(self, out: PipelineOutput) -> None:
        if out.plan.shot_mode == "free":
            collision_count = len(out.plan.free_route.collision_points) if out.plan.free_route is not None else 0
            self.best_label.setText(f"自由模式\n状态 {out.plan.free_status}\n碰撞 {collision_count}")
            self.candidates.setRowCount(0)
            return
        best = out.plan.best
        if best is None:
            self.best_label.setText("无")
        else:
            self.best_label.setText(f"目标 #{best.target_track_id}\n袋口 {best.pocket_index}\n评分 {best.score:.2f}  风险 {best.risk:.2f}\n切角 {best.cut_angle_deg:.1f}°")
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
        self.raw_video_btn.setEnabled(running)
        self.route_video_btn.setEnabled(running)
        if not running:
            self.raw_video_btn.setText("开始无画线视频")
            self.route_video_btn.setText("开始进洞路线视频")
        self.status_label.setText("运行中" if running else "离线")

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
        self.raw_video_btn.setEnabled(False)
        self.route_video_btn.setEnabled(False)
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
