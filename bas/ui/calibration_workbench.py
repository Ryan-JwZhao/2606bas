from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PyQt5 import QtCore, QtWidgets

from ..calibration import (
    create_setting_aware_calibration_service,
    format_holdout_report,
    verify_holdout_file,
)
from ..capture import capture_frames_are_distortion_corrected


@dataclass(frozen=True)
class CalibrationWorkbenchStatus:
    stage: str
    projection_valid: bool
    ball_compensation_valid: bool
    ball_calibration_enabled: bool
    verification_enabled: bool
    camera_running: bool
    projection_errors: tuple[str, ...]
    ball_errors: tuple[str, ...]
    next_action: str


def assess_calibration_workbench(
    calibration,
    *,
    detector_enabled: bool,
    camera_running: bool,
) -> CalibrationWorkbenchStatus:
    projection = calibration.projection
    ball_model = calibration.ball_compensation_model
    projection_valid = bool(projection.is_valid)
    ball_valid = bool(projection_valid and ball_model.is_valid)
    if not projection_valid:
        stage = "joint_calibration"
        next_action = "下一步：启动相机预览，确认方向与曝光后进行相机—投影仪联合校准。"
    elif not detector_enabled:
        stage = "detector_required"
        next_action = "下一步：在设置中启用球检测模型，然后进行球心补偿。"
    elif not ball_valid:
        stage = "ball_compensation"
        next_action = "下一步：清空台面，仅保留一颗标准球，进行30点球心补偿。"
    else:
        stage = "verification"
        next_action = "下一步：选择独立 Holdout JSON，在本窗口完成正式验收。"
    return CalibrationWorkbenchStatus(
        stage=stage,
        projection_valid=projection_valid,
        ball_compensation_valid=ball_valid,
        ball_calibration_enabled=bool(projection_valid and detector_enabled),
        verification_enabled=projection_valid,
        camera_running=bool(camera_running),
        projection_errors=tuple(getattr(projection, "compatibility_errors", ()) or ()),
        ball_errors=tuple(getattr(ball_model, "compatibility_errors", ()) or ()),
        next_action=next_action,
    )


class CalibrationWorkbenchDialog(QtWidgets.QDialog):
    """State-driven GUI for the complete independent-geometry calibration flow."""

    def __init__(
        self,
        operator,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        service_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        super().__init__(parent or operator)
        self.operator = operator
        self._service_factory = service_factory or self._create_calibration_service
        self.calibration = None
        self.status: Optional[CalibrationWorkbenchStatus] = None
        self.setWindowTitle("独立几何校准工作台")
        self.resize(980, 760)

        root = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("工业相机—投影仪联合校准与球心补偿")
        title.setObjectName("title")
        root.addWidget(title)
        intro = QtWidgets.QLabel(
            "校准工作台按当前状态开放操作。未完成投影校准时，相机预览仍可正常启动；"
            "联合校准有效后才会开放球心补偿和几何验收。整个流程不使用相机外参。"
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        self.camera_preview = QtWidgets.QLabel("启动相机预览后，当前工业相机画面会显示在这里。")
        self.camera_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_preview.setMinimumHeight(220)
        self.camera_preview.setStyleSheet("background:#111; border:1px solid #333;")
        root.addWidget(self.camera_preview)

        status_box = QtWidgets.QGroupBox("当前状态")
        status_grid = QtWidgets.QGridLayout(status_box)
        self.camera_state = QtWidgets.QLabel()
        self.orientation_state = QtWidgets.QLabel()
        self.projection_state = QtWidgets.QLabel()
        self.ball_state = QtWidgets.QLabel()
        status_grid.addWidget(QtWidgets.QLabel("相机画面"), 0, 0)
        status_grid.addWidget(self.camera_state, 0, 1)
        status_grid.addWidget(QtWidgets.QLabel("方向/坐标域"), 0, 2)
        status_grid.addWidget(self.orientation_state, 0, 3)
        status_grid.addWidget(QtWidgets.QLabel("联合校准"), 1, 0)
        status_grid.addWidget(self.projection_state, 1, 1)
        status_grid.addWidget(QtWidgets.QLabel("球心补偿"), 1, 2)
        status_grid.addWidget(self.ball_state, 1, 3)
        root.addWidget(status_box)

        self.next_action = QtWidgets.QLabel()
        self.next_action.setObjectName("bestShot")
        self.next_action.setWordWrap(True)
        root.addWidget(self.next_action)

        workflow_box = QtWidgets.QGroupBox("校准流程")
        workflow = QtWidgets.QGridLayout(workflow_box)
        self.start_camera_btn = QtWidgets.QPushButton("启动相机预览")
        self.stop_camera_btn = QtWidgets.QPushButton("停止相机预览")
        self.settings_btn = QtWidgets.QPushButton("打开采集与几何设置")
        self.joint_btn = QtWidgets.QPushButton("开始联合校准")
        self.ball_btn = QtWidgets.QPushButton("开始球心补偿")
        self.show_result_btn = QtWidgets.QPushButton("投影显示校准结果")
        self.refresh_btn = QtWidgets.QPushButton("刷新校准状态")
        workflow.addWidget(self.start_camera_btn, 0, 0)
        workflow.addWidget(self.stop_camera_btn, 0, 1)
        workflow.addWidget(self.settings_btn, 0, 2)
        workflow.addWidget(self.joint_btn, 1, 0)
        workflow.addWidget(self.ball_btn, 1, 1)
        workflow.addWidget(self.show_result_btn, 1, 2)
        workflow.addWidget(self.refresh_btn, 2, 0, 1, 3)
        root.addWidget(workflow_box)

        verify_box = QtWidgets.QGroupBox("独立 Holdout 验收")
        verify_layout = QtWidgets.QVBoxLayout(verify_box)
        holdout_row = QtWidgets.QHBoxLayout()
        self.holdout_path = QtWidgets.QLineEdit()
        self.holdout_path.setPlaceholderText("选择独立 Holdout JSON")
        self.browse_holdout_btn = QtWidgets.QPushButton("选择文件")
        self.verify_btn = QtWidgets.QPushButton("开始图形化验收")
        holdout_row.addWidget(self.holdout_path, 1)
        holdout_row.addWidget(self.browse_holdout_btn)
        holdout_row.addWidget(self.verify_btn)
        verify_layout.addLayout(holdout_row)
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("验收结果会显示在这里，无需执行命令。")
        verify_layout.addWidget(self.report, 1)
        root.addWidget(verify_box, 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.start_camera_btn.clicked.connect(self._start_camera)
        self.stop_camera_btn.clicked.connect(self._stop_camera)
        self.settings_btn.clicked.connect(self._open_settings)
        self.joint_btn.clicked.connect(self._run_joint_calibration)
        self.ball_btn.clicked.connect(self._run_ball_compensation)
        self.show_result_btn.clicked.connect(self._show_result)
        self.refresh_btn.clicked.connect(self.refresh)
        self.browse_holdout_btn.clicked.connect(self._browse_holdout)
        self.verify_btn.clicked.connect(self._verify_holdout)
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self._refresh_camera_preview)
        self._preview_timer.start()
        self.refresh()

    def _create_calibration_service(self):
        pipeline = getattr(self.operator, "pipeline", None)
        capture = getattr(pipeline, "capture", None)
        frame_undistorted = (
            bool(getattr(capture, "frame_distortion_corrected", False))
            if capture is not None
            else bool(capture_frames_are_distortion_corrected(self.operator.config.camera))
        )
        return create_setting_aware_calibration_service(
            self.operator.config.calibration,
            self.operator.config.camera,
            frame_undistorted=frame_undistorted,
            detector_config=self.operator.config.detector,
            projection_config=self.operator.config.projection,
        )

    @QtCore.pyqtSlot()
    def refresh(self) -> None:
        try:
            self.calibration = self._service_factory()
            detector_enabled = str(self.operator.config.detector.backend or "disabled").lower() != "disabled"
            camera_running = getattr(self.operator, "pipeline", None) is not None
            self.status = assess_calibration_workbench(
                self.calibration,
                detector_enabled=detector_enabled,
                camera_running=camera_running,
            )
            self._render_status()
        except Exception as exc:
            self.report.setPlainText(f"刷新校准状态失败：{exc}")

    def _render_status(self) -> None:
        assert self.status is not None and self.calibration is not None
        coordinate_domain = "undistorted" if self.calibration.distortion_correction_enabled else "raw"
        camera_rotation = int(self.operator.config.camera.frame_rotation_degrees) % 360
        calibration_rotation = int(
            self.operator.config.projection.calibration_rotation_degrees
        ) % 360
        output_rotation = int(self.operator.config.projection.output_rotation_degrees) % 360
        self.camera_state.setText("运行中" if self.status.camera_running else "未启动（仍可开始校准）")
        self.orientation_state.setText(
            f"相机 {camera_rotation}° / {coordinate_domain}；"
            f"投影校准 {calibration_rotation}° / 输出 {output_rotation}°"
        )
        self.projection_state.setText(self._artifact_text(self.status.projection_valid, self.status.projection_errors))
        self.ball_state.setText(self._artifact_text(self.status.ball_compensation_valid, self.status.ball_errors))
        self.next_action.setText(self.status.next_action)
        self.start_camera_btn.setEnabled(not self.status.camera_running)
        self.stop_camera_btn.setEnabled(self.status.camera_running)
        self.joint_btn.setEnabled(True)
        self.ball_btn.setEnabled(self.status.ball_calibration_enabled)
        self.show_result_btn.setEnabled(self.status.projection_valid)
        self.verify_btn.setEnabled(self.status.verification_enabled)

    @staticmethod
    def _artifact_text(valid: bool, errors: tuple[str, ...]) -> str:
        if valid:
            return "有效"
        if errors:
            return "需重标定：" + ", ".join(errors[:3])
        return "未完成或无效"

    def _start_camera(self) -> None:
        if getattr(self.operator, "pipeline", None) is None:
            self.operator.start_pipeline()
        self.refresh()

    def _refresh_camera_preview(self) -> None:
        source = getattr(self.operator, "preview_label", None)
        pixmap = source.pixmap() if source is not None and hasattr(source, "pixmap") else None
        if pixmap is None or pixmap.isNull():
            return
        self.camera_preview.setPixmap(
            pixmap.scaled(
                self.camera_preview.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def _stop_camera(self) -> None:
        if getattr(self.operator, "pipeline", None) is not None:
            self.operator.stop_pipeline()
        self.refresh()

    def _open_settings(self) -> None:
        self.operator.open_settings()
        self.refresh()

    def _run_joint_calibration(self) -> None:
        self.operator.run_linked_projector_calibration()
        self.refresh()

    def _run_ball_compensation(self) -> None:
        if self.status is None or not self.status.ball_calibration_enabled:
            return
        self.operator.run_engineered_ball_compensation_wizard()
        self.refresh()

    def _show_result(self) -> None:
        if self.calibration is None or not self.calibration.projection.is_valid:
            return
        self.operator.show_projector_calibration_result(self.calibration)
        self.operator.show_projector_residual_overlay(self.calibration)

    def _browse_holdout(self) -> None:
        start = str(Path(self.holdout_path.text()).parent) if self.holdout_path.text().strip() else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择独立 Holdout JSON", start, "JSON (*.json)")
        if path:
            self.holdout_path.setText(path)

    def _verify_holdout(self) -> None:
        if self.calibration is None or not self.calibration.projection.is_valid:
            return
        path = self.holdout_path.text().strip()
        if not path:
            QtWidgets.QMessageBox.information(self, "尚未选择文件", "请先选择独立 Holdout JSON。")
            return
        try:
            report = verify_holdout_file(path, self.calibration)
            self.report.setPlainText(format_holdout_report(report))
        except Exception as exc:
            self.report.setPlainText(f"Holdout 验收失败：{exc}")

    def closeEvent(self, event) -> None:
        self._preview_timer.stop()
        super().closeEvent(event)
