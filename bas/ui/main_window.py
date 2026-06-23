from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

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
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.pipeline: Optional[RuntimePipeline] = None
        self.projection_window: Optional[ProjectionWindow] = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.frame_count = 0
        self.started_at = 0.0
        self.last_output: Optional[PipelineOutput] = None

        self.setWindowTitle("BAS Control Console")
        self.resize(1420, 860)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self._apply_style()
        self._sync_controls_from_config()
        self._set_running(False)
        self._append_log("就绪")

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        self.title = QtWidgets.QLabel("BAS 台球智能辅助")
        self.subtitle = QtWidgets.QLabel("实时视觉、状态估计、路线投影")
        title_box.addWidget(self.title)
        title_box.addWidget(self.subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        self.status_pill = QtWidgets.QLabel("离线")
        self.status_pill.setObjectName("statusPill")
        header.addWidget(self.status_pill)
        outer.addLayout(header)

        main = QtWidgets.QHBoxLayout()
        main.setSpacing(12)
        outer.addLayout(main, 1)

        self.sidebar = self._panel()
        self.sidebar.setFixedWidth(252)
        side_layout = QtWidgets.QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(12)

        self.start_btn = self._button("启动", QtWidgets.QStyle.SP_MediaPlay)
        self.stop_btn = self._button("停止", QtWidgets.QStyle.SP_MediaStop)
        self.probe_btn = self._button("探测相机", QtWidgets.QStyle.SP_BrowserReload)
        self.projection_btn = self._button("打开投影", QtWidgets.QStyle.SP_ComputerIcon)
        self.start_btn.clicked.connect(self.start_pipeline)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        self.probe_btn.clicked.connect(self.probe_camera_devices)
        self.projection_btn.clicked.connect(self.toggle_projection_window)

        side_layout.addWidget(self.start_btn)
        side_layout.addWidget(self.stop_btn)
        side_layout.addWidget(self.probe_btn)
        side_layout.addWidget(self.projection_btn)
        side_layout.addSpacing(8)

        side_layout.addWidget(self._section_label("输入"))
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(["synthetic", "auto", "opencv", "nori", "video"])
        self.device_spin = QtWidgets.QSpinBox()
        self.device_spin.setRange(0, 32)
        self.detector_combo = QtWidgets.QComboBox()
        self.detector_combo.addItems(["debug_color", "disabled", "ultralytics"])
        side_layout.addWidget(self._field("相机", self.backend_combo))
        side_layout.addWidget(self._field("设备", self.device_spin))
        side_layout.addWidget(self._field("检测", self.detector_combo))

        self.replay_check = QtWidgets.QCheckBox("记录回放")
        self.replay_check.setChecked(self.config.replay.enabled)
        self.projection_check = QtWidgets.QCheckBox("同步投影")
        self.projection_check.setChecked(False)
        side_layout.addWidget(self.replay_check)
        side_layout.addWidget(self.projection_check)
        side_layout.addStretch(1)

        self.config_label = QtWidgets.QLabel(str(Path("configs/default.yaml")))
        self.config_label.setObjectName("muted")
        self.config_label.setWordWrap(True)
        side_layout.addWidget(self.config_label)
        main.addWidget(self.sidebar)

        center = QtWidgets.QVBoxLayout()
        center.setSpacing(12)
        self.preview_panel = self._panel()
        preview_layout = QtWidgets.QVBoxLayout(self.preview_panel)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setObjectName("preview")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 420)
        self.preview_label.setText("等待画面")
        preview_layout.addWidget(self.preview_label, 1)
        center.addWidget(self.preview_panel, 1)

        metric_row = QtWidgets.QHBoxLayout()
        metric_row.setSpacing(10)
        self.fps_value = self._metric("FPS", "--")
        self.det_value = self._metric("检测", "--")
        self.track_value = self._metric("跟踪", "--")
        self.phase_value = self._metric("状态", "--")
        for item in [self.fps_value, self.det_value, self.track_value, self.phase_value]:
            metric_row.addWidget(item)
        center.addLayout(metric_row)
        main.addLayout(center, 1)

        right = self._panel()
        right.setFixedWidth(330)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(12)
        right_layout.addWidget(self._section_label("最佳路线"))
        self.best_label = QtWidgets.QLabel("无")
        self.best_label.setObjectName("bestShot")
        self.best_label.setWordWrap(True)
        right_layout.addWidget(self.best_label)
        right_layout.addWidget(self._section_label("候选"))
        self.candidates = QtWidgets.QTableWidget(0, 4)
        self.candidates.setHorizontalHeaderLabels(["ID", "袋口", "评分", "风险"])
        self.candidates.horizontalHeader().setStretchLastSection(True)
        self.candidates.verticalHeader().setVisible(False)
        self.candidates.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.candidates.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        right_layout.addWidget(self.candidates, 1)
        right_layout.addWidget(self._section_label("事件"))
        self.event_list = QtWidgets.QListWidget()
        right_layout.addWidget(self.event_list, 1)
        main.addWidget(right)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setObjectName("logBox")
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(130)
        outer.addWidget(self.log_box)

    def _apply_style(self) -> None:
        self.title.setObjectName("title")
        self.subtitle.setObjectName("subtitle")
        self.start_btn.setObjectName("primary")
        self.stop_btn.setObjectName("danger")
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #111418;
                color: #eef3f7;
                font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QLabel#muted, QLabel#subtitle {
                color: #93a2ad;
            }
            QLabel#statusPill {
                padding: 6px 14px;
                border-radius: 13px;
                background: #2b343b;
                color: #b7c6d0;
                font-weight: 600;
            }
            QLabel#preview {
                background: #07100c;
                border: 1px solid #283238;
                border-radius: 8px;
                color: #81909a;
                font-size: 16px;
            }
            QFrame#panel, QWidget#metric {
                background: #171d22;
                border: 1px solid #27323a;
                border-radius: 8px;
            }
            QLabel#title {
                font-size: 22px;
                font-weight: 700;
                color: #f6f9fb;
            }
            QLabel#section {
                color: #b8c6cf;
                font-weight: 700;
                padding-top: 6px;
            }
            QLabel#metricName {
                color: #8ea0aa;
                font-size: 12px;
            }
            QLabel#metricValue {
                color: #ffffff;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#bestShot {
                background: #101a17;
                border: 1px solid #2e4f43;
                border-radius: 8px;
                padding: 12px;
                color: #dff9ec;
                line-height: 1.35;
            }
            QPushButton {
                background: #24313a;
                border: 1px solid #38505e;
                border-radius: 8px;
                padding: 10px 12px;
                text-align: left;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #2c3c46;
                border-color: #4c6b7d;
            }
            QPushButton:pressed {
                background: #1d2a31;
            }
            QPushButton#primary {
                background: #0f7b61;
                border-color: #22a47f;
                color: #ffffff;
            }
            QPushButton#danger {
                background: #6b2830;
                border-color: #a34752;
                color: #ffffff;
            }
            QComboBox, QSpinBox {
                background: #10161a;
                border: 1px solid #34434c;
                border-radius: 7px;
                padding: 7px;
                min-height: 24px;
            }
            QCheckBox {
                spacing: 8px;
                color: #c7d2da;
            }
            QTableWidget, QListWidget, QPlainTextEdit {
                background: #10161a;
                border: 1px solid #28343c;
                border-radius: 8px;
                gridline-color: #263039;
                color: #dce7ed;
                selection-background-color: #285e70;
            }
            QHeaderView::section {
                background: #1d262d;
                color: #b7c6d0;
                border: 0;
                padding: 7px;
                font-weight: 700;
            }
            """
        )

    def _panel(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("panel")
        return frame

    def _section_label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("section")
        return label

    def _button(self, text: str, icon_id: QtWidgets.QStyle.StandardPixmap) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setIcon(self.style().standardIcon(icon_id))
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        return btn

    def _field(self, label_text: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QtWidgets.QLabel(label_text)
        label.setObjectName("muted")
        layout.addWidget(label)
        layout.addWidget(widget)
        return box

    def _metric(self, name: str, value: str) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        box.setObjectName("metric")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        name_label = QtWidgets.QLabel(name)
        name_label.setObjectName("metricName")
        value_label = QtWidgets.QLabel(value)
        value_label.setObjectName("metricValue")
        layout.addWidget(name_label)
        layout.addWidget(value_label)
        box.value_label = value_label  # type: ignore[attr-defined]
        return box

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
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 210, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"{det.cls_name} {det.conf:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 250, 255), 1, cv2.LINE_AA)
        for tr in out.tracks.tracks:
            cx, cy = [int(round(v)) for v in tr.center_px]
            cv2.circle(img, (cx, cy), max(4, int(round(tr.radius_px))), (0, 240, 150), 2, cv2.LINE_AA)
            cv2.putText(img, f"#{tr.track_id} {tr.group}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 255, 230), 1, cv2.LINE_AA)
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
        if calib.projection.homography is not None:
            try:
                inv_h = np.linalg.inv(calib.projection.homography)
                pts_cam = cv2.perspectiveTransform(pts_proj.reshape((-1, 1, 2)).astype(np.float32), inv_h).reshape((-1, 2))
                pts = np.round(pts_cam).astype(np.int32)
                cv2.line(img, tuple(pts[0]), tuple(pts[1]), (0, 255, 150), 3, cv2.LINE_AA)
                cv2.line(img, tuple(pts[2]), tuple(pts[3]), (255, 190, 80), 3, cv2.LINE_AA)
            except Exception:
                return

    def _set_preview_image(self, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def _update_stats(self, out: PipelineOutput) -> None:
        elapsed = max(1e-6, time.perf_counter() - self.started_at)
        fps = self.frame_count / elapsed
        self.fps_value.value_label.setText(f"{fps:.1f}")  # type: ignore[attr-defined]
        self.det_value.value_label.setText(str(len(out.detections.detections)))  # type: ignore[attr-defined]
        self.track_value.value_label.setText(str(len(out.tracks.tracks)))  # type: ignore[attr-defined]
        self.phase_value.value_label.setText(out.state.phase)  # type: ignore[attr-defined]

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
        self.status_pill.setText("运行中" if running else "离线")
        self.status_pill.setStyleSheet("background: #0f7b61; color: #ffffff;" if running else "")

    def _append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{stamp}] {text}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_pipeline()
        super().closeEvent(event)


def run_operator_ui(config: AppConfig) -> int:
    configure_logging(config.logging.directory, config.logging.level)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("BAS Control Console")
    app.setStyle("Fusion")
    window = OperatorWindow(config)
    window.show()
    return int(app.exec_())
