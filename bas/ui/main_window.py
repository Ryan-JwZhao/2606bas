from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..app import PipelineOutput, RuntimePipeline
from ..capture import probe_cameras
from ..config import AppConfig
from ..logging_config import configure_logging
from ..projection.star_formula import StarFormulaConfig
from ..projection.window import ProjectionWindow
from ..schemas import ProjectionOverlay
from ..user_settings import UserSettings

LOGGER = logging.getLogger(__name__)


COMMON_RESOLUTIONS = ["1920x1080", "3840x2160", "2560x1440", "1280x720", "1280x800", "640x480"]
COMMON_FPS = ["30", "60", "120", "164"]


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
        self.video_path = self._path_row(config.camera.video_path, "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)")
        self.nori_sdk_root = self._dir_row(config.camera.nori_sdk_root)
        self.outline_path = self._path_row(config.geometry.outline_path, "JSON (*.json);;所有文件 (*.*)")
        self.inline_path = self._path_row(config.geometry.inline_path, "JSON (*.json);;所有文件 (*.*)")
        self.pocket_path = self._path_row(config.geometry.pocket_path, "JSON (*.json);;所有文件 (*.*)")
        self.detector_backend = QtWidgets.QComboBox()
        self.detector_backend.addItems(["disabled", "ultralytics", "debug_color"])
        self.detector_backend.setCurrentText(config.detector.backend)
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
        form.addRow("视频文件路径", self.video_path)
        form.addRow("Nori SDK 目录", self.nori_sdk_root)
        form.addRow("检测后端", self.detector_backend)
        form.addRow("outline.json", self.outline_path)
        form.addRow("inline.json", self.inline_path)
        form.addRow("pocket.json", self.pocket_path)
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
        config.camera.video_path = self.video_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.camera.nori_sdk_root = self.nori_sdk_root.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.detector.backend = self.detector_backend.currentText()
        config.geometry.outline_path = self.outline_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.geometry.inline_path = self.inline_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
        config.geometry.pocket_path = self.pocket_path.line_edit.text().strip() or None  # type: ignore[attr-defined]
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


class OperatorWindow(QtWidgets.QMainWindow):
    BASE_WIDTH = 1420
    BASE_HEIGHT = 860

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.user_settings = UserSettings.load()
        self.star_formula = StarFormulaConfig.from_mapping(self.user_settings.star_formula)
        self.pipeline: Optional[RuntimePipeline] = None
        self.projection_window: Optional[ProjectionWindow] = None
        self.last_output: Optional[PipelineOutput] = None
        self._last_preview_bgr: Optional[np.ndarray] = None
        self._ui_scale = 1.0
        self.frame_count = 0
        self.started_at = 0.0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.setWindowTitle("BAS Control Console")
        self.resize(self.BASE_WIDTH, self.BASE_HEIGHT)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self._sync_controls_from_config()
        self._apply_scale(force=True)
        self._set_running(False)
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
        self.settings_btn = self._button("设置")
        self.probe_btn = self._button("探测相机")
        self.capture_btn.clicked.connect(self.toggle_capture)
        self.projection_btn.clicked.connect(self.toggle_projection_window)
        self.settings_btn.clicked.connect(self.open_settings)
        self.probe_btn.clicked.connect(self.probe_camera_devices)
        for btn in [self.capture_btn, self.projection_btn, self.settings_btn, self.probe_btn]:
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
        self.right_panel.setFixedWidth(px(330))
        self.preview_label.setMinimumSize(px(620), px(390))
        self.log_box.setMaximumHeight(px(130))
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
        QComboBox, QSpinBox {{
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

    def _save_user_settings(self) -> None:
        UserSettings.from_config(self.config, self.star_formula.to_dict()).save()

    @QtCore.pyqtSlot()
    def toggle_capture(self) -> None:
        if self.pipeline is None:
            self.start_pipeline()
        else:
            self.stop_pipeline()

    def start_pipeline(self) -> None:
        self._sync_config_from_controls()
        self._save_user_settings()
        self._append_log("启动采集中")
        try:
            self.pipeline = RuntimePipeline(self.config, star_formula=self.star_formula)
        except Exception as exc:
            self.pipeline = None
            self._append_log(f"启动失败: {exc}")
            QtWidgets.QMessageBox.critical(self, "启动失败", str(exc))
            return
        self.frame_count = 0
        self.started_at = time.perf_counter()
        self.timer.start(max(1, int(1000 / max(1, self.config.camera.fps))))
        self._set_running(True)
        self._append_log("采集已启动")

    def stop_pipeline(self) -> None:
        self.timer.stop()
        if self.pipeline is not None:
            try:
                self.pipeline.close()
            finally:
                self.pipeline = None
        self._set_running(False)
        self._append_log("采集已停止")

    @QtCore.pyqtSlot()
    def open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self.star_formula, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        dialog.apply_to_config(self.config)
        self.star_formula = dialog.star_formula_config()
        self._sync_controls_from_config()
        self._save_user_settings()
        if self.projection_window is not None:
            self.projection_window.set_star_formula(self.star_formula)
            self._refresh_projection()
        self._append_log("设置已保存")

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
            self._ensure_projection_window()
            self.projection_btn.setText("停止投影")
            self._append_log("投影已启动")
        else:
            self.projection_window.close()
            self.projection_window = None
            self.projection_btn.setText("开始投影")
            self._append_log("投影已停止")

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
        if self.pipeline is None:
            return
        out = self.pipeline.step()
        if out is None:
            self.stop_pipeline()
            return
        self.last_output = out
        self.frame_count += 1
        self._update_preview(out)
        self._update_stats(out)
        self._update_plan(out)
        self._update_events(out)
        self._refresh_projection()

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
        self._set_preview_image(img)

    def _set_preview_image(self, image_bgr: np.ndarray) -> None:
        self._last_preview_bgr = image_bgr.copy()
        self._refresh_preview_pixmap()

    def _refresh_preview_pixmap(self) -> None:
        if self._last_preview_bgr is None:
            return
        rgb = cv2.cvtColor(self._last_preview_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def _update_stats(self, out: PipelineOutput) -> None:
        elapsed = max(1e-6, time.perf_counter() - self.started_at)
        self.fps_metric.value_label.setText(f"{self.frame_count / elapsed:.1f}")  # type: ignore[attr-defined]
        self.det_metric.value_label.setText(str(len(out.detections.detections)))  # type: ignore[attr-defined]
        self.track_metric.value_label.setText(str(len(out.tracks.tracks)))  # type: ignore[attr-defined]
        self.phase_metric.value_label.setText(out.state.phase)  # type: ignore[attr-defined]

    def _update_plan(self, out: PipelineOutput) -> None:
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
            text = f"{out.frame.frame_id}: {event.name}"
            if self.event_list.count() == 0 or self.event_list.item(self.event_list.count() - 1).text() != text:
                self.event_list.addItem(text)
        while self.event_list.count() > 80:
            self.event_list.takeItem(0)
        self.event_list.scrollToBottom()

    def _set_running(self, running: bool) -> None:
        self.capture_btn.setText("结束采集" if running else "开始采集")
        self.backend_combo.setEnabled(not running)
        self.nori_device_combo.setEnabled(not running)
        self.device_spin.setEnabled(not running)
        self.resolution_combo.setEnabled(not running)
        self.fps_combo.setEnabled(not running)
        self.replay_check.setEnabled(not running)
        self.status_label.setText("运行中" if running else "离线")

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


def run_operator_ui(config: AppConfig) -> int:
    configure_logging(config.logging.directory, config.logging.level)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("BAS Control Console")
    app.setStyle("Fusion")
    window = OperatorWindow(config)
    window.showMaximized()
    return int(app.exec_())
