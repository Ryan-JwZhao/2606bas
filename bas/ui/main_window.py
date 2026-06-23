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
from ..projection.window import ProjectionWindow

LOGGER = logging.getLogger(__name__)


class OperatorWindow(QtWidgets.QMainWindow):
    BASE_WIDTH = 1420
    BASE_HEIGHT = 860

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
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
        self.start_btn = self._button("启动")
        self.stop_btn = self._button("停止")
        self.probe_btn = self._button("探测相机")
        self.projection_btn = self._button("打开投影")
        self.start_btn.clicked.connect(self.start_pipeline)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        self.probe_btn.clicked.connect(self.probe_camera_devices)
        self.projection_btn.clicked.connect(self.toggle_projection_window)
        self.side_layout.addWidget(self.start_btn)
        self.side_layout.addWidget(self.stop_btn)
        self.side_layout.addWidget(self.probe_btn)
        self.side_layout.addWidget(self.projection_btn)
        self.side_layout.addSpacing(6)

        self.side_layout.addWidget(self._section_label("输入"))
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(["synthetic", "auto", "opencv", "nori", "video"])
        self.device_spin = QtWidgets.QSpinBox()
        self.device_spin.setRange(0, 32)
        self.detector_combo = QtWidgets.QComboBox()
        self.detector_combo.addItems(["debug_color", "disabled", "ultralytics"])
        self.side_layout.addWidget(self._field("相机", self.backend_combo))
        self.side_layout.addWidget(self._field("设备", self.device_spin))
        self.side_layout.addWidget(self._field("检测", self.detector_combo))

        self.replay_check = QtWidgets.QCheckBox("记录回放")
        self.replay_check.setChecked(self.config.replay.enabled)
        self.projection_check = QtWidgets.QCheckBox("同步投影")
        self.projection_check.setChecked(False)
        self.side_layout.addWidget(self.replay_check)
        self.side_layout.addWidget(self.projection_check)
        self.side_layout.addStretch(1)

        self.config_label = QtWidgets.QLabel(str(Path("configs/default.yaml")))
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

    def _metric(self, name: str, value: str) -> QtWidgets.QWidget:
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
        scale_w = max(0.1, self.width() / float(self.BASE_WIDTH))
        scale_h = max(0.1, self.height() / float(self.BASE_HEIGHT))
        return float(max(0.92, min(1.55, min(scale_w, scale_h))))

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

        self.sidebar.setFixedWidth(px(252))
        self.right_panel.setFixedWidth(px(330))
        self.preview_label.setMinimumSize(px(620), px(390))
        self.log_box.setMaximumHeight(px(130))
        self.candidates.verticalHeader().setDefaultSectionSize(px(30))
        self.candidates.horizontalHeader().setDefaultSectionSize(px(34))

        for metric in [self.fps_metric, self.det_metric, self.track_metric, self.phase_metric]:
            metric.metric_layout.setContentsMargins(px(12), px(9), px(12), px(9))  # type: ignore[attr-defined]
            metric.metric_layout.setSpacing(px(2))  # type: ignore[attr-defined]

        self.setStyleSheet(self._stylesheet(scale))
        self._refresh_preview_pixmap()

    def _stylesheet(self, scale: float) -> str:
        def px(value: float) -> int:
            return max(1, int(round(value * scale)))

        base_font = px(13)
        small_font = px(12)
        title_font = px(22)
        metric_font = px(20)
        preview_font = px(16)
        return f"""
        QMainWindow, QWidget {{
            background: #101010;
            color: #f2f2f2;
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: {base_font}px;
        }}
        QLabel#title {{
            font-size: {title_font}px;
            font-weight: 700;
            color: #ffffff;
        }}
        QLabel#muted {{
            color: #b8b8b8;
            font-size: {small_font}px;
        }}
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
            font-size: {preview_font}px;
        }}
        QLabel#section {{
            color: #ffffff;
            font-weight: 700;
            padding-top: {px(5)}px;
        }}
        QLabel#metricName {{
            color: #b8b8b8;
            font-size: {small_font}px;
        }}
        QLabel#metricValue {{
            color: #ffffff;
            font-size: {metric_font}px;
            font-weight: 700;
        }}
        QLabel#bestShot {{
            background: #111111;
            border: 1px solid #3a3a3a;
            padding: {px(10)}px;
            color: #ffffff;
        }}
        QPushButton {{
            background: #202020;
            border: 1px solid #4a4a4a;
            border-radius: {px(2)}px;
            padding: {px(9)}px {px(10)}px;
            color: #ffffff;
            font-weight: 600;
            text-align: center;
        }}
        QPushButton:hover {{
            background: #2b2b2b;
            border-color: #666666;
        }}
        QPushButton:pressed {{
            background: #171717;
        }}
        QPushButton:disabled {{
            color: #777777;
            background: #181818;
            border-color: #303030;
        }}
        QComboBox, QSpinBox {{
            background: #111111;
            border: 1px solid #444444;
            border-radius: {px(2)}px;
            padding: {px(6)}px;
            min-height: {px(25)}px;
            color: #ffffff;
        }}
        QCheckBox {{
            color: #ffffff;
            spacing: {px(8)}px;
        }}
        QTableWidget, QListWidget, QPlainTextEdit {{
            background: #111111;
            border: 1px solid #343434;
            color: #ffffff;
            gridline-color: #303030;
            selection-background-color: #3a3a3a;
            selection-color: #ffffff;
        }}
        QHeaderView::section {{
            background: #202020;
            color: #ffffff;
            border: 0;
            padding: {px(6)}px;
            font-weight: 600;
        }}
        QScrollBar:vertical, QScrollBar:horizontal {{
            background: #111111;
            width: {px(12)}px;
            height: {px(12)}px;
        }}
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
            background: #555555;
            min-height: {px(24)}px;
            min-width: {px(24)}px;
        }}
        """

    def _sync_controls_from_config(self) -> None:
        self.backend_combo.setCurrentText(str(self.config.camera.backend))
        self.device_spin.setValue(int(self.config.camera.device_index))
        self.detector_combo.setCurrentText(str(self.config.detector.backend))

    def _sync_config_from_controls(self) -> None:
        self.config.camera.backend = self.backend_combo.currentText()
        self.config.camera.device_index = int(self.device_spin.value())
        self.config.detector.backend = self.detector_combo.currentText()
        self.config.replay.enabled = self.replay_check.isChecked()

    @QtCore.pyqtSlot()
    def start_pipeline(self) -> None:
        if self.pipeline is not None:
            return
        self._sync_config_from_controls()
        self._append_log("启动中")
        try:
            self.pipeline = RuntimePipeline(self.config)
        except Exception as exc:
            self.pipeline = None
            self._append_log(f"启动失败: {exc}")
            QtWidgets.QMessageBox.critical(self, "启动失败", str(exc))
            return
        self.frame_count = 0
        self.started_at = time.perf_counter()
        self.timer.start(max(1, int(1000 / max(1, self.config.camera.fps))))
        self._set_running(True)
        if self.projection_check.isChecked():
            self._ensure_projection_window()
        self._append_log("已启动")

    @QtCore.pyqtSlot()
    def stop_pipeline(self) -> None:
        self.timer.stop()
        if self.pipeline is not None:
            try:
                self.pipeline.close()
            finally:
                self.pipeline = None
        if self.projection_window is not None:
            self.projection_window.close()
            self.projection_window = None
        self._set_running(False)
        self._append_log("已停止")

    @QtCore.pyqtSlot()
    def probe_camera_devices(self) -> None:
        self._append_log("正在探测相机")
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

    @QtCore.pyqtSlot()
    def toggle_projection_window(self) -> None:
        if self.projection_window is None:
            self._ensure_projection_window()
            self._append_log("投影窗口已打开")
        else:
            self.projection_window.close()
            self.projection_window = None
            self._append_log("投影窗口已关闭")

    def _ensure_projection_window(self) -> None:
        if self.projection_window is not None:
            return
        self.projection_window = ProjectionWindow(self.config.projection)
        self.projection_window.show_on_configured_screen()
        if self.last_output is not None:
            self.projection_window.set_overlay(self.last_output.overlay)

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
        if self.projection_window is not None:
            self.projection_window.set_overlay(out.overlay)

    def _update_preview(self, out: PipelineOutput) -> None:
        frame = out.frame.image
        if frame is None:
            return
        img = frame.copy()
        for det in out.detections.detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (235, 235, 235), 2, cv2.LINE_AA)
            cv2.putText(
                img,
                f"{det.cls_name} {det.conf:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55 * self._ui_scale,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        for tr in out.tracks.tracks:
            cx, cy = [int(round(v)) for v in tr.center_px]
            cv2.circle(img, (cx, cy), max(4, int(round(tr.radius_px))), (250, 250, 250), 2, cv2.LINE_AA)
            cv2.putText(
                img,
                f"#{tr.track_id} {tr.group}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55 * self._ui_scale,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
        if out.plan.best is not None:
            self._draw_plan_on_preview(img, out)
        self._set_preview_image(img)

    def _draw_plan_on_preview(self, img: np.ndarray, out: PipelineOutput) -> None:
        best = out.plan.best
        if best is None or self.pipeline is None:
            return
        calib = self.pipeline.calibration
        pts_mm = np.asarray([best.cue_ball, best.ghost_ball, best.object_ball, best.pocket_point], dtype=np.float32)
        pts_proj = calib.table_mm_to_projector_px(pts_mm)
        if calib.projection.homography is None:
            return
        try:
            inv_h = np.linalg.inv(calib.projection.homography)
            pts_cam = cv2.perspectiveTransform(pts_proj.reshape((-1, 1, 2)).astype(np.float32), inv_h).reshape((-1, 2))
            pts = np.round(pts_cam).astype(np.int32)
            cv2.line(img, tuple(pts[0]), tuple(pts[1]), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(img, tuple(pts[2]), tuple(pts[3]), (190, 190, 190), 3, cv2.LINE_AA)
        except Exception:
            return

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
        self.preview_label.setPixmap(
            pix.scaled(self.preview_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

    def _update_stats(self, out: PipelineOutput) -> None:
        elapsed = max(1e-6, time.perf_counter() - self.started_at)
        fps = self.frame_count / elapsed
        self.fps_metric.value_label.setText(f"{fps:.1f}")  # type: ignore[attr-defined]
        self.det_metric.value_label.setText(str(len(out.detections.detections)))  # type: ignore[attr-defined]
        self.track_metric.value_label.setText(str(len(out.tracks.tracks)))  # type: ignore[attr-defined]
        self.phase_metric.value_label.setText(out.state.phase)  # type: ignore[attr-defined]

    def _update_plan(self, out: PipelineOutput) -> None:
        best = out.plan.best
        if best is None:
            self.best_label.setText("无")
        else:
            self.best_label.setText(
                f"目标 #{best.target_track_id}\n"
                f"袋口 {best.pocket_index}\n"
                f"评分 {best.score:.2f}  风险 {best.risk:.2f}\n"
                f"切角 {best.cut_angle_deg:.1f}°"
            )
        self.candidates.setRowCount(len(out.plan.candidates))
        for row, cand in enumerate(out.plan.candidates):
            values = [cand.candidate_id, str(cand.pocket_index), f"{cand.score:.2f}", f"{cand.risk:.2f}"]
            for col, value in enumerate(values):
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
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.backend_combo.setEnabled(not running)
        self.device_spin.setEnabled(not running)
        self.detector_combo.setEnabled(not running)
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
        super().closeEvent(event)


def run_operator_ui(config: AppConfig) -> int:
    configure_logging(config.logging.directory, config.logging.level)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("BAS Control Console")
    app.setStyle("Fusion")
    window = OperatorWindow(config)
    window.showMaximized()
    return int(app.exec_())

