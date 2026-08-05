from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..calibration import (
    BallCompensationSample,
    BallCompensationCheckpoint,
    BallCompensationValidationError,
    HoldoutCheckpointObservation,
    MIN_BALL_COMPENSATION_ACCEPTED_SAMPLES,
    MIN_BALL_COMPENSATION_HOLDOUT_SAMPLES,
    MIN_BALL_COMPENSATION_TRAINING_SAMPLES,
    aggregate_ball_compensation_holdout_repeats,
    ball_compensation_checkpoint_compatibility_errors,
    ball_holdout_geometry_is_formal,
    build_ball_compensation_model,
    build_engineered_ball_sampling_grid,
    create_setting_aware_calibration_service,
    delete_ball_compensation_checkpoint,
    evaluate_ball_compensation_training_residuals,
    fit_and_validate_ball_compensation,
    fit_with_reused_holdout_diagnostic,
    load_ball_compensation_checkpoint,
    load_reusable_holdout_source,
    make_ball_compensation_checkpoint,
    restart_ball_compensation_checkpoint_from_holdout,
    save_ball_compensation_checkpoint,
    select_ball_compensation_holdout_targets,
    start_calibration_audit,
    update_calibration_table_boundaries_from_geometry_frame,
)
from ..calibration.ball_sampling_detection import (
    BALL_SAMPLING_DETECTION_VERSION,
    ball_sampling_delta_is_plausible,
    ball_sampling_geometry_is_usable,
    ball_sampling_target_distance_limit_px,
)
from ..calibration.ball_compensation_resampling import (
    aggregate_training_repeats,
    assess_holdout_repeatability,
    plan_failed_holdout_recovery,
    prepare_high_residual_training_resampling,
)
from ..calibration.quality_standards import FORMAL_TABLE_ERROR_MEDIAN_MM
from ..capture import create_capture_service
from ..geometry_runtime import load_validated_table_geometry
from ..geometry_contract import calibration_context
from ..paths import PROJECT_ROOT
from ..perception import build_detection_region_policy, create_detector, filter_detections_by_region
from ..schemas import Detection
from ..table_boundaries import EdgeInsets
from ..utils import group_from_class
from .ball_compensation_residual_view import (
    render_projector_ball_residual_view,
    render_training_residual_bubble_chart,
    save_residual_view,
)

TIMESTAMPED_BALL_COMPENSATION_FILE_RE = re.compile(r"^(?P<base>.*?)(?:_\d{8}_\d{6})?$")
DEFAULT_BALL_COMPENSATION_OUTPUT_DIR = PROJECT_ROOT / "local_settings" / "calibrations"
BALL_COMPENSATION_CHECKPOINT_PATH = PROJECT_ROOT / "local_settings" / "checkpoints" / "ball_compensation_checkpoint.json"
ENGINEERED_SAMPLING_COLS = 8
ENGINEERED_SAMPLING_ROWS = 7
ENGINEERED_TRAINING_REPEATS = 3
ENGINEERED_SAMPLE_SETTLE_DELAY_SECONDS = 3.0
ENGINEERED_SAMPLE_TIMEOUT_SECONDS = 28.0
ENGINEERED_SAMPLE_DROPOUT_GRACE_SECONDS = 0.45
ENGINEERED_CENTER_PLAYABLE_SAFE_INSET_MM = 6.0
ENGINEERED_HOLDOUT_LOCATION_COUNT = 10
ENGINEERED_HOLDOUT_REPEATS = 3
ENGINEERED_HOLDOUT_MIN_GEOMETRY_QUALITY = 0.70
ENGINEERED_HOLDOUT_REPEATABILITY_MAX_DEVIATION_MM = 2.0
ENGINEERED_FAILED_REGION_ERROR_MM = FORMAL_TABLE_ERROR_MEDIAN_MM
ENGINEERED_FAILED_REGION_NEIGHBOR_COUNT = 8


class _SamplingAborted(RuntimeError):
    pass


def resolve_ball_sampling_region(table, *, boundaries_ready: bool) -> tuple[np.ndarray, float, str]:
    """Return the polygon, extra inset and diagnostic name used for ball sampling."""

    using_center_playable = bool(boundaries_ready and table.center_playable_polygon_mm)
    polygon = np.asarray(
        table.center_playable_polygon_mm or table.inner_polygon_mm,
        dtype=np.float32,
    ).reshape((-1, 2))
    safe_inset_mm = (
        ENGINEERED_CENTER_PLAYABLE_SAFE_INSET_MM
        if using_center_playable
        else 0.5 * float(table.ball_diameter_mm)
    )
    return (
        polygon,
        safe_inset_mm,
        "center_playable" if using_center_playable else "inner_polygon_fallback",
    )


def ball_compensation_path_or_default(path_value: Optional[str]) -> Path:
    current = (path_value or "").strip()
    if current:
        candidate = Path(current)
        if candidate.exists() and candidate.is_dir():
            return candidate / "engineered_ball_compensation.json"
        if not candidate.suffix and candidate.name.lower() in {"calibrations", "calibration", "local_settings"}:
            return candidate / "engineered_ball_compensation.json"
        return candidate
    return DEFAULT_BALL_COMPENSATION_OUTPUT_DIR / "engineered_ball_compensation.json"


def ball_compensation_path_from_input(raw_value: str, current_value: Optional[str]) -> Path:
    raw = (raw_value or "").strip()
    if not raw:
        return ball_compensation_path_or_default(current_value).resolve()
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    base_dir = ball_compensation_path_or_default(current_value).parent
    return (base_dir / candidate).resolve()


def timestamped_ball_compensation_output_path(path_value: Optional[str]) -> Path:
    template = ball_compensation_path_or_default(path_value)
    stem_match = TIMESTAMPED_BALL_COMPENSATION_FILE_RE.match(template.stem)
    base_stem = stem_match.group("base") if stem_match else template.stem
    safe_stem = (base_stem or "engineered_ball_compensation").strip() or "engineered_ball_compensation"
    suffix = template.suffix or ".json"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_BALL_COMPENSATION_OUTPUT_DIR / f"{safe_stem}_{stamp}{suffix}"


class EngineeredBallCompensationWizardDialog(QtWidgets.QDialog):
    def __init__(
        self,
        operator,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        auto_start: bool = False,
        auto_close_on_success: bool = False,
        resample_residual_threshold_mm: float | None = None,
    ):
        super().__init__(parent or operator)
        self.operator = operator
        self._busy = False
        self._abort_requested = False
        self._saved_path: Optional[Path] = None
        self._samples: list[BallCompensationSample] = []
        self._auto_close_on_success = bool(auto_close_on_success)
        self._resample_residual_threshold_mm = (
            None
            if resample_residual_threshold_mm is None
            else max(0.0, float(resample_residual_threshold_mm))
        )
        self.setWindowTitle("工程球体补偿自动采样向导")
        self.resize(980, 820)

        layout = QtWidgets.QVBoxLayout(self)

        if self._resample_residual_threshold_mm is not None:
            intro_text = (
                f"当前为高残差定点重采模式：系统会从当前启用的 56 点结果中自动筛选训练残差不低于 "
                f"{self._resample_residual_threshold_mm:.1f} mm 的点。每个新样本替换原编号样本，其余训练数据保持不变；"
                "定点重采后可选择重新采集一套全新的 30 次正式 Holdout，或沿用现有 30 次数据进行临时诊断。"
            )
        else:
            intro_text = (
                "该向导会先按 8×7、共 56 个全桌目标点完成训练采样；每个目标自动采集 3 个稳定批次并取中位数，"
                "第一遍有效样本全部用于拟合，不会再扣除边角点。随后会在 10 个空间分散位置进行 3 轮独立 Holdout 复采。"
                "请确保投影平面校准文件与球检测模型都有效。"
                f"采样时建议清空台面，仅保留一颗球；允许跳过异常点，但至少要获得 {MIN_BALL_COMPENSATION_ACCEPTED_SAMPLES} 个有效样本。"
            )
        intro = QtWidgets.QLabel(intro_text)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        steps = QtWidgets.QPlainTextEdit()
        steps.setReadOnly(True)
        if self._resample_residual_threshold_mm is not None:
            step_lines = [
                "1. 确认当前启用的是需要修正的 56 点球心补偿文件，且相机、投影和检测模型没有改变。",
                "2. 清空台面，仅保留一颗标准球。",
                "3. 点击“开始重采高残差点”；日志会列出自动筛出的原始点号与残差。",
                f"4. 每个待重采位置自动完成 {ENGINEERED_TRAINING_REPEATS} 个稳定批次并取中位数，不允许跳过。",
                f"5. 完成后选择：采集 {ENGINEERED_HOLDOUT_LOCATION_COUNT} 个位置 × "
                f"{ENGINEERED_HOLDOUT_REPEATS} 轮全新独立 Holdout（正式验收），或沿用现有 30 次数据（仅诊断）。",
            ]
        else:
            step_lines = [
                    "1. 在设置中确认当前投影平面校准文件可加载。",
                    "2. 启用可用的球检测后端；向导严格沿用当前工业相机畸变校正开关，不会自行开启校正。",
                    "3. 清空台面，仅保留一颗标准球。",
                    "4. 点击“开始自动采样”，按提示把球逐个移动到贴近球边缘的短弧目标圈。",
                    f"5. 每个训练位置自动完成 {ENGINEERED_TRAINING_REPEATS} 个稳定批次并取中位数；第一遍至少保留 "
                    f"{MIN_BALL_COMPENSATION_TRAINING_SAMPLES} 个训练样本；随后按提示完成 "
                    f"{ENGINEERED_HOLDOUT_LOCATION_COUNT} 个位置 × {ENGINEERED_HOLDOUT_REPEATS} 轮独立 Holdout。",
                    "6. 正式 Holdout 只接受轮廓椭圆球心；若画面只能得到 bbox，向导会等待重试而不会写入验收数据。",
                    f"7. 30次完成后先按每个位置最大复采离散度 {ENGINEERED_HOLDOUT_REPEATABILITY_MAX_DEVIATION_MM:.1f} mm 预检；"
                    "只重采超限位置。正式验收失败时，失败区域会标红，并可重采对应训练点及周边一圈。",
                ]
        steps.setPlainText("\n".join(step_lines))
        layout.addWidget(steps)

        file_box = QtWidgets.QGroupBox("输出文件模板")
        file_layout = QtWidgets.QVBoxLayout(file_box)
        self.output_path_edit = QtWidgets.QLineEdit(self.operator.config.calibration.engineered_ball_compensation_file or "")
        self.output_path_edit.setPlaceholderText("未设置时将保存到 local_settings/calibrations/engineered_ball_compensation_时间戳.json")
        file_layout.addWidget(self.output_path_edit)
        file_actions = QtWidgets.QHBoxLayout()
        browse_btn = QtWidgets.QPushButton("浏览")
        browse_btn.clicked.connect(self._browse_output_path)
        file_actions.addWidget(browse_btn)
        file_actions.addStretch(1)
        file_layout.addLayout(file_actions)
        layout.addWidget(file_box)

        self.preview = QtWidgets.QLabel("等待开始采样")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(300)
        self.preview.setStyleSheet("background:#111; border:1px solid #333;")
        layout.addWidget(self.preview)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.summary = QtWidgets.QLabel("尚未开始工程球体补偿采样。")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box, 1)

        button_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton(
            "开始重采高残差点"
            if self._resample_residual_threshold_mm is not None
            else "开始自动采样"
        )
        self.start_btn.clicked.connect(self.run_sampling)
        button_row.addWidget(self.start_btn)
        self.stop_btn = QtWidgets.QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.request_stop)
        button_row.addWidget(self.stop_btn)
        button_row.addStretch(1)
        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)
        if auto_start:
            QtCore.QTimer.singleShot(0, self.run_sampling)

    def reject(self) -> None:
        if self._busy:
            self.request_stop()
            return
        super().reject()

    def _browse_output_path(self) -> None:
        current = str(ball_compensation_path_or_default(self.output_path_edit.text().strip() or self.operator.config.calibration.engineered_ball_compensation_file))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "选择工程球体补偿文件模板",
            current,
            "JSON (*.json);;所有文件 (*.*)",
        )
        if path:
            self.output_path_edit.setText(path)

    def _append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{stamp}] {text}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        self.operator._append_log(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.start_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)

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

    def request_stop(self) -> None:
        self._abort_requested = True
        self._append_log("已请求停止当前采样流程，等待本轮检测退出。")

    def _output_template_path(self) -> Path:
        raw = self.output_path_edit.text().strip()
        return ball_compensation_path_from_input(raw, self.operator.config.calibration.engineered_ball_compensation_file)

    def _safe_ball_compensation_output_path(self, calibration) -> Path:
        template_path = self._output_template_path()
        plane_path = Path(calibration.projection.source_path).resolve() if calibration.projection.source_path else None
        try:
            template_resolved = template_path.resolve()
        except Exception:
            template_resolved = template_path
        if plane_path is not None and str(template_resolved).lower() == str(plane_path).lower():
            fallback = template_resolved.parent / "engineered_ball_compensation.json"
            self._append_log(
                "检测到工程球体补偿输出路径与工程平面校准文件相同，已自动改用独立补偿文件模板: "
                f"{fallback}"
            )
            self.output_path_edit.setText(str(fallback))
            return fallback
        return template_resolved

    def _checkpoint_context(
        self,
        calibration,
        *,
        frame_width: int,
        frame_height: int,
        camera_coordinate_domain: str,
    ) -> dict:
        return {
            "projection_file": calibration.projection.source_path,
            "detector_model_file": self.operator.config.detector.model_path,
            "camera_coordinate_domain": str(camera_coordinate_domain),
            "frame_rotation_degrees": int(self.operator.config.camera.frame_rotation_degrees),
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "ball_diameter_mm": float(calibration.table.ball_diameter_mm),
            "sampling_detection": BALL_SAMPLING_DETECTION_VERSION,
        }

    def _prompt_checkpoint_action(
        self,
        checkpoint: BallCompensationCheckpoint,
        compatibility_errors: list[str],
    ) -> str:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("发现球心补偿暂存进度")
        box.setIcon(QtWidgets.QMessageBox.Question if not compatibility_errors else QtWidgets.QMessageBox.Warning)
        progress_text = (
            f"训练进度：{checkpoint.training_cursor}/{len(checkpoint.sampling_grid_table_mm)}\n"
            f"Holdout 进度：{checkpoint.holdout_completed_count}/"
            f"{ENGINEERED_HOLDOUT_LOCATION_COUNT * ENGINEERED_HOLDOUT_REPEATS}\n"
            f"保存时间：{checkpoint.saved_at}"
        )
        if compatibility_errors:
            box.setText(
                progress_text
                + "\n\n该暂存与当前标定环境不兼容，不能安全继承：\n- "
                + "\n- ".join(compatibility_errors)
            )
            discard_btn = box.addButton("放弃旧暂存并重新开始", QtWidgets.QMessageBox.DestructiveRole)
            cancel_btn = box.addButton("取消", QtWidgets.QMessageBox.RejectRole)
            box.exec_()
            return "discard" if box.clickedButton() is discard_btn else "cancel"
        holdout_total = ENGINEERED_HOLDOUT_LOCATION_COUNT * ENGINEERED_HOLDOUT_REPEATS
        training_complete = checkpoint.training_cursor == len(checkpoint.sampling_grid_table_mm)
        holdout_complete = checkpoint.holdout_completed_count >= holdout_total
        if training_complete and holdout_complete:
            box.setText(
                progress_text
                + "\n\n上一轮 56 点训练已完成，但 30 次 Holdout 已采完后仍未通过验收。"
                "请选择本轮重新采集范围："
            )
            restart_holdout_btn = box.addButton(
                "保留 56 点，重新采集 30 点",
                QtWidgets.QMessageBox.AcceptRole,
            )
            restart_all_btn = box.addButton(
                "清空全部，重新采集 56+30",
                QtWidgets.QMessageBox.DestructiveRole,
            )
            cancel_btn = box.addButton("取消", QtWidgets.QMessageBox.RejectRole)
            del cancel_btn
            box.exec_()
            if box.clickedButton() is restart_holdout_btn:
                return "restart_holdout"
            if box.clickedButton() is restart_all_btn:
                return "discard"
            return "cancel"
        box.setText(progress_text + "\n\n是否从该进度继续？")
        resume_btn = box.addButton("继续暂存进度", QtWidgets.QMessageBox.AcceptRole)
        discard_btn = box.addButton("不继承，重新开始", QtWidgets.QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("取消", QtWidgets.QMessageBox.RejectRole)
        del cancel_btn
        box.exec_()
        if box.clickedButton() is resume_btn:
            return "resume"
        if box.clickedButton() is discard_btn:
            return "discard"
        return "cancel"

    def _write_sampling_checkpoint(
        self,
        *,
        context: dict,
        sample_points: np.ndarray,
        training_cursor: int,
        holdout_observations: list[HoldoutCheckpointObservation],
        pending_training_resample_indices: list[int] | tuple[int, ...] = (),
        failed_holdout_targets_table_mm: list[tuple[float, float]] | tuple[tuple[float, float], ...] = (),
        holdout_generation: int = 0,
    ) -> None:
        checkpoint = make_ball_compensation_checkpoint(
            context=context,
            sampling_grid_table_mm=sample_points,
            training_cursor=training_cursor,
            training_samples=self._samples,
            holdout_observations=holdout_observations,
            pending_training_resample_indices=pending_training_resample_indices,
            failed_holdout_targets_table_mm=failed_holdout_targets_table_mm,
            holdout_generation=holdout_generation,
        )
        save_ball_compensation_checkpoint(BALL_COMPENSATION_CHECKPOINT_PATH, checkpoint)

    @staticmethod
    def _hydrate_resumed_training_samples(samples: list[BallCompensationSample], calibration) -> None:
        for sample in samples:
            target = np.asarray(sample.target_table_mm, dtype=np.float32).reshape((1, 2))
            projected = calibration.table_mm_to_projector_px(target)[0]
            expected = calibration.table_mm_to_camera_px(target)[0]
            observed = calibration.camera_px_to_table_mm(
                np.asarray([sample.detected_camera_px], dtype=np.float32)
            )[0]
            sample.projected_target_px = (float(projected[0]), float(projected[1]))
            sample.expected_camera_px = (float(expected[0]), float(expected[1]))
            sample.observed_table_mm = (float(observed[0]), float(observed[1]))
            sample.delta_table_mm = (
                float(sample.target_table_mm[0] - observed[0]),
                float(sample.target_table_mm[1] - observed[1]),
            )
            if sample.detected_radius_px <= 0.0:
                sample.detected_radius_px = float(
                    _expected_camera_ball_radius(calibration, np.asarray(sample.target_table_mm, dtype=np.float32))
                )

    @QtCore.pyqtSlot()
    def run_sampling(self) -> None:
        if self._busy:
            return
        capture = None
        audit = start_calibration_audit(
            self.operator.config.logging.directory,
            "ball_center_compensation",
            context={
                "camera_backend": self.operator.config.camera.backend,
                "camera_id": self.operator.config.camera.camera_id,
                "camera_device_index": self.operator.config.camera.device_index,
                "nori_device_id": self.operator.config.camera.nori_device_id,
                "detector_backend": self.operator.config.detector.backend,
                "detector_model": self.operator.config.detector.model_path,
                "projection_file": self.operator.config.calibration.projection_file,
                "sampling_grid": [ENGINEERED_SAMPLING_COLS, ENGINEERED_SAMPLING_ROWS],
                "sampling_detection": BALL_SAMPLING_DETECTION_VERSION,
            },
        )
        self._abort_requested = False
        self._saved_path = None
        self._samples = []
        self._set_busy(True)
        self.progress.setValue(0)
        self.summary.setText("正在准备工程球体补偿采样。")
        self._append_log("工程球体补偿自动采样开始，正在校验校准和检测链路。")
        try:
            self.operator._sync_config_from_controls()
            self.operator._save_user_settings()
            if self.operator.pipeline is not None:
                self._append_log("检测到实时采集正在运行，先自动停止以释放相机。")
                self.operator.stop_pipeline()
            capture = create_capture_service(self.operator.config.camera)
            capture_info = capture.info()
            calibration = create_setting_aware_calibration_service(
                self.operator.config.calibration,
                self.operator.config.camera,
                frame_undistorted=bool(capture.frame_distortion_corrected),
                detector_config=self.operator.config.detector,
                projection_config=self.operator.config.projection,
                actual_frame_size=(int(capture_info.width), int(capture_info.height)),
            )
            audit.event(
                "calibration_loaded",
                metrics={
                    "frame_width": int(capture_info.width),
                    "frame_height": int(capture_info.height),
                    "projection_valid": calibration.projection.is_valid,
                    "previous_ball_model_valid": calibration.ball_compensation_model.is_valid,
                },
                details={
                    "projection_source": calibration.projection.source_path,
                    "ball_model_source": calibration.ball_compensation_model.source_path,
                    "geometry_quality": calibration.geometry_quality_report,
                },
            )
            self._append_log(
                "camera_coordinate_domain="
                f"{'undistorted' if calibration.distortion_correction_enabled else 'raw'}, "
                "严格遵循当前工业相机畸变校正开关"
            )
            if not calibration.projection.is_valid:
                plane_path = self.operator.config.calibration.projection_file
                ball_path = self.operator.config.calibration.engineered_ball_compensation_file
                same_path_hint = ""
                if plane_path and ball_path:
                    try:
                        if str(Path(plane_path).resolve()).lower() == str(Path(ball_path).resolve()).lower():
                            same_path_hint = (
                                "\n检测到工程平面校准文件与工程球体补偿文件指向同一路径。"
                                "这通常意味着平面校准 JSON 已被球体补偿 JSON 覆盖，请重新指定一份有效的工程平面校准文件。"
                            )
                    except Exception:
                        pass
                raise RuntimeError(
                    "当前投影平面校准文件无效，无法继续进行球体补偿采样。"
                    f"{same_path_hint}"
                )
            detector = create_detector(self.operator.config.detector)
            if getattr(detector, "version", "disabled") == "disabled":
                raise RuntimeError("工程球体补偿向导需要启用球检测后端，当前 detector.backend=disabled。")
            geometry = load_validated_table_geometry(
                self.operator.config.geometry.outline_path,
                self.operator.config.geometry.inline_path,
                self.operator.config.geometry.pocket_path,
                allow_empty=False,
            )
            self.operator.ensure_projection_window_for_operator()
            prime_frame = _prime_capture_frame(capture)
            if prime_frame is not None:
                self._set_preview_image(prime_frame)
            boundaries_ready = False
            if prime_frame is not None:
                boundaries_ready = update_calibration_table_boundaries_from_geometry_frame(
                    calibration,
                    geometry,
                    prime_frame.shape,
                    projection_visible_insets=EdgeInsets(
                        top_mm=float(self.operator.config.calibration.projection_visible_inset_top_mm),
                        right_mm=float(self.operator.config.calibration.projection_visible_inset_right_mm),
                        bottom_mm=float(self.operator.config.calibration.projection_visible_inset_bottom_mm),
                        left_mm=float(self.operator.config.calibration.projection_visible_inset_left_mm),
                    ),
                    physical_rail_insets=EdgeInsets(
                        top_mm=float(self.operator.config.calibration.physical_rail_inset_top_mm),
                        right_mm=float(self.operator.config.calibration.physical_rail_inset_right_mm),
                        bottom_mm=float(self.operator.config.calibration.physical_rail_inset_bottom_mm),
                        left_mm=float(self.operator.config.calibration.physical_rail_inset_left_mm),
                    ),
                    physical_middle_pocket_relief_top_mm=float(self.operator.config.calibration.physical_middle_pocket_relief_top_mm),
                    physical_middle_pocket_relief_bottom_mm=float(self.operator.config.calibration.physical_middle_pocket_relief_bottom_mm),
                    center_reachable_extra_margin_mm=float(self.operator.config.calibration.center_reachable_extra_margin_mm),
                )
            if boundaries_ready:
                self._append_log("已根据 inline/pocket 几何刷新球心可达区，采样点将只落在 center playable 内圈。")
            else:
                self._append_log("未能从 inline/pocket 几何刷新球心可达区，将回退到矩形安全采样区。")
            sample_polygon, edge_safe_inset_mm, sampling_region_name = resolve_ball_sampling_region(
                calibration.table,
                boundaries_ready=boundaries_ready,
            )
            sample_points = build_engineered_ball_sampling_grid(
                calibration.table.width_mm,
                calibration.table.height_mm,
                calibration.table.ball_diameter_mm,
                cols=ENGINEERED_SAMPLING_COLS,
                rows=ENGINEERED_SAMPLING_ROWS,
                preferred_polygon_mm=sample_polygon,
                extra_safe_inset_mm=edge_safe_inset_mm,
                priority_points_mm=np.asarray(calibration.table.pockets_mm, dtype=np.float32),
            )
            if len(sample_points) < MIN_BALL_COMPENSATION_ACCEPTED_SAMPLES:
                raise RuntimeError(
                    f"当前安全采样区只能生成 {len(sample_points)} 个点，"
                    f"第一遍至少需要 {MIN_BALL_COMPENSATION_ACCEPTED_SAMPLES} 个有效训练位置。"
                )
            self._append_log(
                "sampling_region="
                f"{sampling_region_name}, "
                f"grid={ENGINEERED_SAMPLING_COLS}x{ENGINEERED_SAMPLING_ROWS}, "
                f"edge_safe_inset={edge_safe_inset_mm:.1f}mm"
            )
            audit.event(
                "sampling_grid_built",
                metrics={
                    "sample_count": len(sample_points),
                    "columns": ENGINEERED_SAMPLING_COLS,
                    "rows": ENGINEERED_SAMPLING_ROWS,
                    "edge_safe_inset_mm": edge_safe_inset_mm,
                },
                details={"sampling_region": sampling_region_name},
            )
            frame_height, frame_width = prime_frame.shape[:2] if prime_frame is not None else (
                int(self.operator.config.camera.height),
                int(self.operator.config.camera.width),
            )
            coordinate_domain = (
                "undistorted"
                if calibration.distortion_correction_enabled and calibration.camera.is_valid
                else "raw"
            )
            model_context = calibration_context(
                frame_width=frame_width,
                frame_height=frame_height,
                frame_rotation_degrees=int(self.operator.config.camera.frame_rotation_degrees),
                camera_coordinate_domain=coordinate_domain,
                distortion_file=(
                    self.operator.config.camera.distortion_correction_file
                    if coordinate_domain == "undistorted"
                    else None
                ),
                projection_file=calibration.projection.source_path,
                detector_model_file=self.operator.config.detector.model_path,
                ball_diameter_mm=float(calibration.table.ball_diameter_mm),
            )
            checkpoint_context = self._checkpoint_context(
                calibration,
                frame_width=frame_width,
                frame_height=frame_height,
                camera_coordinate_domain=coordinate_domain,
            )
            training_cursor = 0
            checkpoint_holdout_observations: list[HoldoutCheckpointObservation] = []
            pending_training_resample_indices: list[int] = []
            failed_holdout_targets_table_mm: list[tuple[float, float]] = []
            holdout_generation = 0
            targeted_resampling = self._resample_residual_threshold_mm is not None
            if targeted_resampling:
                ball_path = self.operator.config.calibration.engineered_ball_compensation_file
                if not ball_path:
                    raise RuntimeError("当前未启用球心补偿文件，无法计算高残差重采点。")
                task = prepare_high_residual_training_resampling(
                    ball_path,
                    current_sampling_grid_table_mm=sample_points,
                    expected_calibration_context=model_context,
                    threshold_mm=float(self._resample_residual_threshold_mm),
                )
                source = task.source
                plan = task.plan
                sample_points = np.asarray(source.sampling_grid_table_mm, dtype=np.float32).reshape((-1, 2))
                self._samples = list(source.samples)
                training_cursor = len(sample_points)
                pending_training_resample_indices = list(plan.training_sample_indices)
                checkpoint_holdout_observations.clear()
                holdout_generation = 0
                self._hydrate_resumed_training_samples(self._samples, calibration)
                residual_by_index = dict(zip(plan.sample_indices, plan.residuals_mm))
                selected_text = ", ".join(
                    f"{index + 1}({residual_by_index[index]:.2f}mm)"
                    for index in plan.training_sample_indices
                )
                self._append_log(
                    f"已从当前校准筛出 {len(plan.training_sample_indices)} 个训练残差 ≥ "
                    f"{plan.threshold_mm:.1f} mm 的点：{selected_text}。"
                )
                self._append_log("定点重采使用原校准文件保存的 56 点目标坐标，不依赖当前网格生成顺序。")
                self._append_log(
                    "其余训练样本保持不变；重采完成后可选择全新 30 次正式 Holdout，"
                    "或沿用现有 30 次数据作临时诊断。"
                )
                self._write_sampling_checkpoint(
                    context=checkpoint_context,
                    sample_points=sample_points,
                    training_cursor=training_cursor,
                    holdout_observations=[],
                    pending_training_resample_indices=pending_training_resample_indices,
                    failed_holdout_targets_table_mm=[],
                    holdout_generation=holdout_generation,
                )
                audit.event(
                    "high_residual_training_resample_started",
                    metrics={
                        "threshold_mm": plan.threshold_mm,
                        "selected_count": len(plan.training_sample_indices),
                    },
                    details={
                        "source_path": str(source.source_path),
                        "sample_indices": [index + 1 for index in plan.training_sample_indices],
                        "residuals_mm": [residual_by_index[index] for index in plan.training_sample_indices],
                    },
                )
            else:
                try:
                    saved_checkpoint = load_ball_compensation_checkpoint(BALL_COMPENSATION_CHECKPOINT_PATH)
                except Exception as exc:
                    choice = QtWidgets.QMessageBox.question(
                        self,
                        "球心补偿暂存损坏",
                        f"暂存文件无法读取：{exc}\n\n是否删除损坏暂存并重新开始？",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                        QtWidgets.QMessageBox.Yes,
                    )
                    if choice != QtWidgets.QMessageBox.Yes:
                        raise _SamplingAborted()
                    delete_ball_compensation_checkpoint(BALL_COMPENSATION_CHECKPOINT_PATH)
                    saved_checkpoint = None
                if saved_checkpoint is not None:
                    checkpoint_context_errors = ball_compensation_checkpoint_compatibility_errors(
                        saved_checkpoint,
                        context=checkpoint_context,
                        sampling_grid_table_mm=saved_checkpoint.sampling_grid_table_mm,
                    )
                    if not checkpoint_context_errors:
                        sample_points = np.asarray(
                            saved_checkpoint.sampling_grid_table_mm,
                            dtype=np.float32,
                        ).reshape((-1, 2))
                        self._append_log(
                            "断点继续使用暂存文件保存的 56 点目标坐标，"
                            "避免网格生成算法变化导致已采样点错位。"
                        )
                    compatibility_errors = ball_compensation_checkpoint_compatibility_errors(
                        saved_checkpoint,
                        context=checkpoint_context,
                        sampling_grid_table_mm=sample_points,
                    )
                    checkpoint_action = self._prompt_checkpoint_action(saved_checkpoint, compatibility_errors)
                    if checkpoint_action == "cancel":
                        raise _SamplingAborted()
                    if checkpoint_action == "discard":
                        delete_ball_compensation_checkpoint(BALL_COMPENSATION_CHECKPOINT_PATH)
                        audit.event("checkpoint_discarded", details={"path": str(BALL_COMPENSATION_CHECKPOINT_PATH)})
                    else:
                        if checkpoint_action == "restart_holdout":
                            saved_checkpoint = restart_ball_compensation_checkpoint_from_holdout(saved_checkpoint)
                            save_ball_compensation_checkpoint(BALL_COMPENSATION_CHECKPOINT_PATH, saved_checkpoint)
                            audit.event(
                                "checkpoint_holdout_restarted",
                                metrics={
                                    "training_cursor": saved_checkpoint.training_cursor,
                                    "training_sample_count": len(saved_checkpoint.training_samples),
                                },
                                details={"path": str(BALL_COMPENSATION_CHECKPOINT_PATH)},
                            )
                            self._append_log("已保留第一遍 56 点训练记录，并清空旧 Holdout；本轮将重新采集 30 点。")
                        self._samples = list(saved_checkpoint.training_samples)
                        training_cursor = int(saved_checkpoint.training_cursor)
                        checkpoint_holdout_observations = list(saved_checkpoint.holdout_observations)
                        pending_training_resample_indices = list(saved_checkpoint.pending_training_resample_indices)
                        failed_holdout_targets_table_mm = list(saved_checkpoint.failed_holdout_targets_table_mm)
                        holdout_generation = int(saved_checkpoint.holdout_generation)
                        self._hydrate_resumed_training_samples(self._samples, calibration)
                        self.progress.setValue(
                            int(round(training_cursor / max(1, len(sample_points)) * 70.0))
                        )
                        audit.event(
                            "checkpoint_resumed",
                            metrics={
                                "training_cursor": training_cursor,
                                "training_sample_count": len(self._samples),
                                "holdout_completed_count": len(checkpoint_holdout_observations),
                            },
                            details={"path": str(BALL_COMPENSATION_CHECKPOINT_PATH)},
                        )
                        self._append_log(
                            f"已继承球心补偿暂存：训练 {training_cursor}/{len(sample_points)}，"
                            f"Holdout {len(checkpoint_holdout_observations)}/"
                            f"{ENGINEERED_HOLDOUT_LOCATION_COUNT * ENGINEERED_HOLDOUT_REPEATS}。"
                        )
            self._append_log(f"已生成 {len(sample_points)} 个工程采样点，请按投影目标圈移动单颗球。")

            total = max(1, len(sample_points))
            for idx in range(training_cursor, len(sample_points)):
                target_table_mm = sample_points[idx]
                if self._abort_requested:
                    raise _SamplingAborted()
                percent = int(round(idx / total * 70.0))
                self.progress.setValue(percent)
                target_arr = np.asarray([target_table_mm], dtype=np.float32)
                target_proj = calibration.table_mm_to_projector_px(target_arr)[0]
                expected_cam = calibration.table_mm_to_camera_px(target_arr)[0]
                target_ellipse = calibration.table_circle_to_projector_ellipse(
                    target_table_mm.astype(np.float32),
                    0.5 * float(calibration.table.ball_diameter_mm),
                )
                self._show_target(target_ellipse, idx + 1, total)
                self.summary.setText(
                    f"正在等待第 {idx + 1}/{total} 个点稳定。请把单颗球移到目标圈中央，系统检测到稳定后会自动跳到下一点。"
                )
                self._append_log(
                    f"切换到采样点 {idx + 1}/{total}: table=({target_table_mm[0]:.1f}, {target_table_mm[1]:.1f}) mm"
                )
                outcome = self._collect_training_repeats(
                    capture,
                    detector,
                    calibration,
                    geometry,
                    idx,
                    total,
                    target_table_mm.astype(np.float32),
                    target_proj.astype(np.float32),
                    expected_cam.astype(np.float32),
                )
                if outcome is None:
                    audit.event(
                        "sample_skipped",
                        status="warning",
                        metrics={"sample_index": idx + 1, "total": total},
                        details={"target_table_mm": target_table_mm},
                    )
                    remaining_count = len(sample_points) - idx - 1
                    maximum_accepted = len(self._samples) + remaining_count
                    if maximum_accepted < MIN_BALL_COMPENSATION_ACCEPTED_SAMPLES:
                        raise RuntimeError(
                            f"跳过当前点后最多只能获得 {maximum_accepted} 个有效样本，"
                            f"低于最低要求 {MIN_BALL_COMPENSATION_ACCEPTED_SAMPLES} 个；"
                            "已提前结束，避免继续无效采样。"
                        )
                    training_cursor = idx + 1
                    self._write_sampling_checkpoint(
                        context=checkpoint_context,
                        sample_points=sample_points,
                        training_cursor=training_cursor,
                        holdout_observations=checkpoint_holdout_observations,
                        pending_training_resample_indices=pending_training_resample_indices,
                        failed_holdout_targets_table_mm=failed_holdout_targets_table_mm,
                        holdout_generation=holdout_generation,
                    )
                    continue
                self._samples.append(outcome)
                training_cursor = idx + 1
                audit.event(
                    "sample_accepted",
                    metrics={
                        "sample_index": idx + 1,
                        "total": total,
                        "detection_confidence": outcome.detection_confidence,
                        "stability_spread_px": outcome.stability_spread_px,
                        "geometry_quality": outcome.geometry_quality,
                        "delta_norm_mm": float(np.linalg.norm(outcome.delta_table_mm)),
                    },
                    details={
                        "target_table_mm": outcome.target_table_mm,
                        "detected_camera_px": outcome.detected_camera_px,
                        "delta_table_mm": outcome.delta_table_mm,
                        "geometry_method": outcome.geometry_method,
                    },
                )
                self._append_log(
                    "采样成功: "
                    f"camera=({outcome.detected_camera_px[0]:.1f}, {outcome.detected_camera_px[1]:.1f}) px, "
                    f"delta=({outcome.delta_table_mm[0]:.2f}, {outcome.delta_table_mm[1]:.2f}) mm"
                )
                self._write_sampling_checkpoint(
                    context=checkpoint_context,
                    sample_points=sample_points,
                    training_cursor=training_cursor,
                    holdout_observations=checkpoint_holdout_observations,
                    pending_training_resample_indices=pending_training_resample_indices,
                    failed_holdout_targets_table_mm=failed_holdout_targets_table_mm,
                    holdout_generation=holdout_generation,
                )

            if len(self._samples) < MIN_BALL_COMPENSATION_ACCEPTED_SAMPLES:
                raise RuntimeError(
                    f"有效采样点只有 {len(self._samples)} 个，至少需要 "
                    f"{MIN_BALL_COMPENSATION_ACCEPTED_SAMPLES} 个第一遍训练样本。"
                )
            reusable_holdout_source = None
            try:
                reusable_holdout_source = load_reusable_holdout_source(
                    self.operator.config.calibration.engineered_ball_compensation_file,
                    expected_sampling_grid_table_mm=sample_points,
                    expected_calibration_context=model_context,
                )
            except (OSError, TypeError, ValueError) as exc:
                self._append_log(f"现有 Holdout 无法安全读取，将只提供全新 30 次采集：{exc}")
            reused_diagnostic = None
            holdout_strategy_prompted = False
            validated = None
            while validated is None and reused_diagnostic is None:
                if pending_training_resample_indices:
                    checkpoint_holdout_observations.clear()
                    holdout_generation += 1
                    self._resample_training_regions(
                        capture=capture,
                        detector=detector,
                        calibration=calibration,
                        geometry=geometry,
                        sample_points=sample_points,
                        training_cursor=training_cursor,
                        checkpoint_context=checkpoint_context,
                        pending_indices=pending_training_resample_indices,
                        failed_targets=failed_holdout_targets_table_mm,
                        holdout_generation=holdout_generation,
                        audit=audit,
                    )
                    failed_holdout_targets_table_mm.clear()

                # The complete first pass remains the only training set. Failed
                # Holdout observations are diagnostic and are never promoted.
                build_ball_compensation_model(
                    self._samples,
                    ball_diameter_mm=float(calibration.table.ball_diameter_mm),
                    table_width_mm=float(calibration.table.width_mm),
                    table_height_mm=float(calibration.table.height_mm),
                )
                if (
                    not checkpoint_holdout_observations
                    and reusable_holdout_source is not None
                    and not holdout_strategy_prompted
                ):
                    holdout_strategy_prompted = True
                    strategy = self._prompt_holdout_strategy(reusable_holdout_source)
                    if strategy == "cancel":
                        raise _SamplingAborted()
                    if strategy == "reuse":
                        reused_diagnostic = fit_with_reused_holdout_diagnostic(
                            self._samples,
                            calibration,
                            reusable_holdout_source,
                            calibration_context=model_context,
                        )
                        audit.event(
                            "existing_holdout_reused_for_diagnostic",
                            status="warning",
                            metrics={
                                "training_count": len(self._samples),
                                "holdout_location_count": len(reusable_holdout_source.samples),
                                "holdout_observation_count": reusable_holdout_source.observation_count,
                            },
                            details={
                                "source_path": str(reusable_holdout_source.source_path),
                                "holdout_quality": reused_diagnostic.holdout_report,
                            },
                        )
                        self._append_log(
                            "已沿用现有 30 次 Holdout 数据重新评估新 56 点模型；"
                            "该结果只作诊断，不计为正式独立验收。"
                        )
                        break
                try:
                    validated = self._run_holdout_cycle(
                        capture=capture,
                        detector=detector,
                        calibration=calibration,
                        geometry=geometry,
                        sample_points=sample_points,
                        training_cursor=training_cursor,
                        checkpoint_context=checkpoint_context,
                        checkpoint_observations=checkpoint_holdout_observations,
                        model_context=model_context,
                        holdout_generation=holdout_generation,
                        audit=audit,
                    )
                except BallCompensationValidationError as exc:
                    audit.event(
                        "holdout_evaluated",
                        status="failed",
                        metrics={
                            "training_count": exc.training_count,
                            "holdout_count": exc.holdout_count,
                        },
                        details={"holdout_quality": exc.report},
                    )
                    recovery = plan_failed_holdout_recovery(
                        self._samples,
                        exc.report,
                        failure_error_mm=ENGINEERED_FAILED_REGION_ERROR_MM,
                        surrounding_neighbor_count=ENGINEERED_FAILED_REGION_NEIGHBOR_COUNT,
                    )
                    failed_holdout_targets_table_mm[:] = list(recovery.failed_targets_table_mm)
                    self._show_failed_regions(calibration, recovery.failed_targets_table_mm)
                    if recovery.training_sample_indices and self._prompt_failed_holdout_recovery(recovery):
                        pending_training_resample_indices[:] = list(recovery.training_sample_indices)
                        checkpoint_holdout_observations.clear()
                        self._write_sampling_checkpoint(
                            context=checkpoint_context,
                            sample_points=sample_points,
                            training_cursor=training_cursor,
                            holdout_observations=checkpoint_holdout_observations,
                            pending_training_resample_indices=pending_training_resample_indices,
                            failed_holdout_targets_table_mm=failed_holdout_targets_table_mm,
                            holdout_generation=holdout_generation,
                        )
                        validated = None
                        continue
                    self._write_sampling_checkpoint(
                        context=checkpoint_context,
                        sample_points=sample_points,
                        training_cursor=training_cursor,
                        holdout_observations=checkpoint_holdout_observations,
                        failed_holdout_targets_table_mm=failed_holdout_targets_table_mm,
                        holdout_generation=holdout_generation,
                    )
                    raise
            diagnostic_only = reused_diagnostic is not None
            if diagnostic_only:
                model = reused_diagnostic.model
                training_samples = list(self._samples)
                holdout_samples = list(reusable_holdout_source.samples)
                holdout_report = reused_diagnostic.holdout_report
                training_residual_report = reused_diagnostic.training_report
                model.quality_report = {
                    **model.quality_report,
                    "temporary_activation": True,
                    "formal_holdout_gate_waived": True,
                    "formal_holdout_quality_gate_passed": False,
                }
                calibration.ball_compensation_model = model
                calibration._rebuild_geometry()
            else:
                model = validated.model
                training_samples = list(validated.training_samples)
                holdout_samples = list(validated.holdout_samples)
                holdout_report = validated.holdout_report
                training_residual_report = evaluate_ball_compensation_training_residuals(
                    training_samples,
                    calibration,
                )
            audit.event(
                (
                    "training_and_reused_holdout_diagnostic_prepared"
                    if diagnostic_only
                    else "training_and_independent_holdout_prepared"
                ),
                metrics={
                    "accepted_count": len(self._samples),
                    "training_count": len(training_samples),
                    "holdout_count": len(holdout_samples),
                },
            )
            audit.event(
                "reused_holdout_evaluated" if diagnostic_only else "holdout_evaluated",
                status=(
                    "warning"
                    if diagnostic_only
                    else "ok" if bool(holdout_report.get("quality_gate_passed", False)) else "failed"
                ),
                details={
                    "training_quality": model.quality_report,
                    "holdout_quality": holdout_report,
                },
            )
            save_template = self._safe_ball_compensation_output_path(calibration)
            save_path = (
                DEFAULT_BALL_COMPENSATION_OUTPUT_DIR
                / f"engineered_ball_compensation_{time.strftime('%Y%m%d_%H%M%S')}_temp_reused_holdout.json"
                if diagnostic_only
                else timestamped_ball_compensation_output_path(str(save_template))
            )
            residual_view_path = save_path.with_name(f"{save_path.stem}_residuals.png")
            residual_view = render_training_residual_bubble_chart(
                float(calibration.table.width_mm),
                float(calibration.table.height_mm),
                training_residual_report,
            )
            save_residual_view(residual_view_path, residual_view)
            extra_data = {
                "ball_diameter_mm": float(calibration.table.ball_diameter_mm),
                "projection_file": calibration.projection.source_path,
                "settle_delay_seconds": float(ENGINEERED_SAMPLE_SETTLE_DELAY_SECONDS),
                "samples": [sample.to_dict() for sample in training_samples],
                "holdout_samples": [sample.to_dict() for sample in holdout_samples],
                "sampling_grid_table_mm": np.asarray(sample_points, dtype=np.float64).reshape((-1, 2)).tolist(),
                "training_residual_report": training_residual_report,
                "residual_view_file": str(residual_view_path),
            }
            if diagnostic_only:
                extra_data.update(
                    {
                        "temporary_activation": True,
                        "formal_holdout_status": "not_run_after_training_resample",
                        "formal_holdout_gate_waived": True,
                        "diagnostic_reused_holdout_source_file": str(reusable_holdout_source.source_path),
                        "diagnostic_reused_holdout_samples": [sample.to_dict() for sample in holdout_samples],
                        "diagnostic_reused_holdout_repeatability": dict(reusable_holdout_source.repeatability),
                        "diagnostic_reused_holdout_quality_report": holdout_report,
                    }
                )
            else:
                extra_data["holdout_quality_report"] = holdout_report
            model.save_json(
                save_path,
                extra_data=extra_data,
            )
            self._saved_path = save_path
            self.operator.config.calibration.engineered_ball_compensation_file = str(save_path)
            self.operator._sync_controls_from_config()
            self.operator._save_user_settings()
            self.operator._update_module_status()
            if diagnostic_only:
                audit.event(
                    "checkpoint_retained_for_formal_holdout",
                    details={"path": str(BALL_COMPENSATION_CHECKPOINT_PATH)},
                )
            else:
                delete_ball_compensation_checkpoint(BALL_COMPENSATION_CHECKPOINT_PATH)
                audit.event(
                    "checkpoint_consumed",
                    details={"path": str(BALL_COMPENSATION_CHECKPOINT_PATH)},
                )
            self.progress.setValue(100)
            if diagnostic_only:
                self.summary.setText(
                    _ball_compensation_diagnostic_summary(
                        training_report=training_residual_report,
                        holdout_report=holdout_report,
                        save_path=save_path,
                        residual_view_path=residual_view_path,
                    )
                )
            else:
                self.summary.setText(
                    _ball_compensation_completion_summary(
                        accepted_count=len(self._samples),
                        total_count=len(sample_points),
                        training_count=len(training_samples),
                        holdout_count=len(holdout_samples),
                        training_report=model.quality_report,
                        holdout_report=holdout_report,
                        save_path=save_path,
                    )
                )
            self.output_path_edit.setText(str(save_path))
            self._append_log(f"工程球体补偿文件已生成并写回当前设置: {save_path}")
            self._append_log(f"全桌残差视图已保存: {residual_view_path}")
            self._show_residual_view(
                calibration,
                training_residual_report,
                holdout_report=holdout_report,
            )
            audit_path = audit.finish(
                "success",
                quality={
                    "training": model.quality_report,
                    "training_residuals": training_residual_report,
                    "holdout_diagnostic" if diagnostic_only else "holdout": holdout_report,
                },
                artifacts=[save_path, residual_view_path],
            )
            self._append_log(f"校准审计报告: {audit_path}")
            if self._auto_close_on_success:
                QtCore.QTimer.singleShot(0, self.accept)
        except _SamplingAborted:
            audit_path = audit.finish(
                "aborted",
                quality={"accepted_sample_count": len(self._samples)},
            )
            self._append_log(f"校准审计报告: {audit_path}")
            self.summary.setText("工程球体补偿采样已停止，未生成新的补偿文件。")
            self._append_log("工程球体补偿采样已停止。")
        except Exception as exc:
            audit_path = audit.finish(
                "failed",
                quality={"accepted_sample_count": len(self._samples)},
                error=exc,
            )
            self._append_log(f"校准审计报告: {audit_path}")
            self.summary.setText(f"工程球体补偿采样失败: {exc}")
            self._append_log(f"工程球体补偿采样失败: {exc}")
            QtWidgets.QMessageBox.critical(self, "工程球体补偿采样失败", str(exc))
        finally:
            if capture is not None:
                capture.release()
            self._set_busy(False)

    def _resample_training_regions(
        self,
        *,
        capture,
        detector,
        calibration,
        geometry,
        sample_points: np.ndarray,
        training_cursor: int,
        checkpoint_context: dict,
        pending_indices: list[int],
        failed_targets: list[tuple[float, float]],
        holdout_generation: int,
        audit,
    ) -> None:
        total = len(pending_indices)
        for ordinal, sample_index in enumerate(list(pending_indices), start=1):
            target_table_mm = np.asarray(sample_points[int(sample_index)], dtype=np.float32)
            target_arr = target_table_mm.reshape((1, 2))
            target_proj = calibration.table_mm_to_projector_px(target_arr)[0]
            expected_cam = calibration.table_mm_to_camera_px(target_arr)[0]
            target_ellipse = calibration.table_circle_to_projector_ellipse(
                target_table_mm,
                0.5 * float(calibration.table.ball_diameter_mm),
            )
            self._show_target(target_ellipse, ordinal, total, alert=True)
            self.summary.setText(
                f"正在重采高残差/失败区域训练点 {ordinal}/{total}（原第 {sample_index + 1}/56 点）。"
            )
            outcome = self._collect_training_repeats(
                capture,
                detector,
                calibration,
                geometry,
                int(sample_index),
                len(sample_points),
                target_table_mm,
                target_proj.astype(np.float32),
                expected_cam.astype(np.float32),
            )
            if outcome is None:
                raise RuntimeError("高残差/失败区域训练点重采不可跳过；请重试当前点或结束向导。")
            replaced = False
            for position, existing in enumerate(self._samples):
                if int(existing.sample_index) == int(sample_index):
                    self._samples[position] = outcome
                    replaced = True
                    break
            if not replaced:
                raise RuntimeError(f"无法找到待重采的训练样本索引: {sample_index}")
            pending_indices.remove(sample_index)
            audit.event(
                "training_region_resampled",
                metrics={
                    "sample_index": int(sample_index) + 1,
                    "remaining_count": len(pending_indices),
                    "repeat_count": ENGINEERED_TRAINING_REPEATS,
                },
                details={"target_table_mm": outcome.target_table_mm},
            )
            self._write_sampling_checkpoint(
                context=checkpoint_context,
                sample_points=sample_points,
                training_cursor=training_cursor,
                holdout_observations=[],
                pending_training_resample_indices=pending_indices,
                failed_holdout_targets_table_mm=failed_targets,
                holdout_generation=holdout_generation,
            )

    def _run_holdout_cycle(
        self,
        *,
        capture,
        detector,
        calibration,
        geometry,
        sample_points: np.ndarray,
        training_cursor: int,
        checkpoint_context: dict,
        checkpoint_observations: list[HoldoutCheckpointObservation],
        model_context: dict,
        holdout_generation: int,
        audit,
    ):
        holdout_targets = select_ball_compensation_holdout_targets(
            self._samples,
            count=ENGINEERED_HOLDOUT_LOCATION_COUNT,
            require_formal_geometry=True,
            minimum_geometry_quality=ENGINEERED_HOLDOUT_MIN_GEOMETRY_QUALITY,
        )
        repeated_holdout: list[list[BallCompensationSample]] = [[] for _ in holdout_targets]
        completed_keys: set[tuple[int, int]] = set()
        resumed_samples = [item.sample for item in checkpoint_observations]
        self._hydrate_resumed_training_samples(resumed_samples, calibration)
        for item in checkpoint_observations:
            key = (int(item.repeat_index), int(item.location_index))
            if key in completed_keys:
                raise RuntimeError(f"球心补偿暂存包含重复 Holdout 记录: repeat/location={key}")
            if not (0 <= item.repeat_index < ENGINEERED_HOLDOUT_REPEATS):
                raise RuntimeError(f"球心补偿暂存的 Holdout 轮次越界: {item.repeat_index}")
            if not (0 <= item.location_index < len(holdout_targets)):
                raise RuntimeError(f"球心补偿暂存的 Holdout 位置越界: {item.location_index}")
            expected_target = np.asarray(holdout_targets[item.location_index].target_table_mm, dtype=np.float64)
            saved_target = np.asarray(item.sample.target_table_mm, dtype=np.float64)
            if float(np.linalg.norm(expected_target - saved_target)) > 0.5:
                raise RuntimeError("球心补偿暂存的 Holdout 选点与当前选点策略不兼容，请重新开始30次验证。")
            completed_keys.add(key)
            repeated_holdout[item.location_index].append(item.sample)

        total = len(holdout_targets) * ENGINEERED_HOLDOUT_REPEATS
        self._append_log(
            f"开始第 {holdout_generation + 1} 套独立 Holdout："
            f"{len(holdout_targets)} 个位置 × {ENGINEERED_HOLDOUT_REPEATS} 轮。"
        )
        audit.event(
            "holdout_sampling_started",
            metrics={
                "training_count": len(self._samples),
                "location_count": len(holdout_targets),
                "repeat_count": ENGINEERED_HOLDOUT_REPEATS,
                "holdout_generation": holdout_generation,
            },
            details={
                "selected_targets": [
                    {
                        "training_sample_index": int(sample.sample_index) + 1,
                        "target_table_mm": list(sample.target_table_mm),
                        "first_pass_geometry_method": str(sample.geometry_method),
                        "first_pass_geometry_quality": float(sample.geometry_quality),
                    }
                    for sample in holdout_targets
                ]
            },
        )
        schedule: list[tuple[int, int]] = []
        for repeat_index in range(ENGINEERED_HOLDOUT_REPEATS):
            order = list(range(len(holdout_targets)))
            rotation = (repeat_index * 3) % max(1, len(order))
            order = order[rotation:] + order[:rotation]
            schedule.extend((repeat_index, location_index) for location_index in order)
        for observation_index, (repeat_index, location_index) in enumerate(schedule):
            if (repeat_index, location_index) in completed_keys:
                continue
            outcome = self._collect_holdout_observation(
                capture=capture,
                detector=detector,
                calibration=calibration,
                geometry=geometry,
                target_sample=holdout_targets[location_index],
                location_index=location_index,
                repeat_index=repeat_index,
                display_index=observation_index,
                total=total,
                alert=False,
            )
            outcome.sample_index = len(self._samples) + observation_index
            repeated_holdout[location_index].append(outcome)
            checkpoint_observations.append(HoldoutCheckpointObservation(location_index, repeat_index, outcome))
            completed_keys.add((repeat_index, location_index))
            self._audit_and_checkpoint_holdout(
                outcome=outcome,
                location_index=location_index,
                repeat_index=repeat_index,
                audit=audit,
                checkpoint_context=checkpoint_context,
                sample_points=sample_points,
                training_cursor=training_cursor,
                checkpoint_observations=checkpoint_observations,
                holdout_generation=holdout_generation,
            )

        precheck_round = 0
        while True:
            assessment = assess_holdout_repeatability(
                repeated_holdout,
                maximum_deviation_mm=ENGINEERED_HOLDOUT_REPEATABILITY_MAX_DEVIATION_MM,
                minimum_repeats=ENGINEERED_HOLDOUT_REPEATS,
            )
            audit.event(
                "holdout_repeatability_prechecked",
                status="ok" if assessment.passed else "failed",
                metrics={
                    "unstable_location_count": len(assessment.unstable_location_indices),
                    "maximum_deviation_mm": assessment.maximum_deviation_mm,
                    "precheck_round": precheck_round,
                },
                details={
                    "locations": [
                        {
                            "location_index": item.location_index + 1,
                            "repeat_count": item.repeat_count,
                            "maximum_deviation_mm": item.maximum_deviation_mm,
                            "passed": item.passed,
                        }
                        for item in assessment.locations
                    ]
                },
            )
            if assessment.passed:
                break
            precheck_round += 1
            self._append_log(
                "Holdout 重复性预检未通过，仅重采位置："
                + ", ".join(str(index + 1) for index in assessment.unstable_location_indices)
            )
            for location_index in assessment.unstable_location_indices:
                repeated_holdout[location_index] = []
                checkpoint_observations[:] = [
                    item for item in checkpoint_observations if int(item.location_index) != location_index
                ]
                for repeat_index in range(ENGINEERED_HOLDOUT_REPEATS):
                    outcome = self._collect_holdout_observation(
                        capture=capture,
                        detector=detector,
                        calibration=calibration,
                        geometry=geometry,
                        target_sample=holdout_targets[location_index],
                        location_index=location_index,
                        repeat_index=repeat_index,
                        display_index=repeat_index,
                        total=ENGINEERED_HOLDOUT_REPEATS,
                        alert=True,
                    )
                    outcome.sample_index = len(self._samples) + location_index * ENGINEERED_HOLDOUT_REPEATS + repeat_index
                    repeated_holdout[location_index].append(outcome)
                    checkpoint_observations.append(
                        HoldoutCheckpointObservation(location_index, repeat_index, outcome)
                    )
                    self._audit_and_checkpoint_holdout(
                        outcome=outcome,
                        location_index=location_index,
                        repeat_index=repeat_index,
                        audit=audit,
                        checkpoint_context=checkpoint_context,
                        sample_points=sample_points,
                        training_cursor=training_cursor,
                        checkpoint_observations=checkpoint_observations,
                        holdout_generation=holdout_generation,
                        resampled=True,
                    )

        holdout_samples, holdout_repeatability = aggregate_ball_compensation_holdout_repeats(
            repeated_holdout,
            minimum_repeats=ENGINEERED_HOLDOUT_REPEATS,
        )
        audit.event(
            "holdout_repeats_aggregated",
            metrics={
                "location_count": len(holdout_samples),
                "observation_count": holdout_repeatability["observation_count"],
            },
            details={"repeatability": holdout_repeatability},
        )
        return fit_and_validate_ball_compensation(
            self._samples,
            calibration,
            holdout_samples=holdout_samples,
            holdout_repeatability=holdout_repeatability,
            calibration_context=model_context,
        )

    def _collect_holdout_observation(
        self,
        *,
        capture,
        detector,
        calibration,
        geometry,
        target_sample: BallCompensationSample,
        location_index: int,
        repeat_index: int,
        display_index: int,
        total: int,
        alert: bool,
    ) -> BallCompensationSample:
        if self._abort_requested:
            raise _SamplingAborted()
        target_table_mm = np.asarray(target_sample.target_table_mm, dtype=np.float32)
        target_arr = target_table_mm.reshape((1, 2))
        target_proj = calibration.table_mm_to_projector_px(target_arr)[0]
        expected_cam = calibration.table_mm_to_camera_px(target_arr)[0]
        target_ellipse = calibration.table_circle_to_projector_ellipse(
            target_table_mm,
            0.5 * float(calibration.table.ball_diameter_mm),
        )
        self._show_target(target_ellipse, display_index + 1, total, alert=alert)
        self.summary.setText(
            f"Holdout位置 {location_index + 1}，第 {repeat_index + 1}/{ENGINEERED_HOLDOUT_REPEATS} 次。"
            + ("该位置重复性超限，正在局部重采；只判断三次一致性。" if alert else "请重新移动球到目标圈中心。")
        )
        outcome = self._collect_sample_for_target(
            capture,
            detector,
            calibration,
            geometry,
            display_index,
            total,
            target_table_mm,
            target_proj.astype(np.float32),
            expected_cam.astype(np.float32),
            formal_geometry_required=True,
            allow_skip=False,
        )
        if outcome is None:
            raise RuntimeError("正式 Holdout 复采不可跳过；请重试当前点或结束向导。")
        return outcome

    def _audit_and_checkpoint_holdout(
        self,
        *,
        outcome: BallCompensationSample,
        location_index: int,
        repeat_index: int,
        audit,
        checkpoint_context: dict,
        sample_points: np.ndarray,
        training_cursor: int,
        checkpoint_observations: list[HoldoutCheckpointObservation],
        holdout_generation: int,
        resampled: bool = False,
    ) -> None:
        audit.event(
            "holdout_sample_resampled" if resampled else "holdout_sample_accepted",
            metrics={
                "location_index": location_index + 1,
                "repeat_index": repeat_index + 1,
                "geometry_quality": outcome.geometry_quality,
                "stability_spread_px": outcome.stability_spread_px,
                "holdout_generation": holdout_generation,
            },
            details={
                "target_table_mm": outcome.target_table_mm,
                "detected_camera_px": outcome.detected_camera_px,
                "geometry_method": outcome.geometry_method,
            },
        )
        self._write_sampling_checkpoint(
            context=checkpoint_context,
            sample_points=sample_points,
            training_cursor=training_cursor,
            holdout_observations=checkpoint_observations,
            holdout_generation=holdout_generation,
        )

    def _prompt_failed_holdout_recovery(self, recovery) -> bool:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("球心补偿固定失败区域")
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setText(
            f"正式Holdout仍有 {len(recovery.failed_holdout_location_indices)} 个位置误差不低于 "
            f"{ENGINEERED_FAILED_REGION_ERROR_MM:.1f} mm，已在投影画面标红。\n\n"
            f"可以重采这些位置对应的56点训练点及周边一圈，共 "
            f"{len(recovery.training_sample_indices)} 个训练点；旧Holdout只保留为诊断，"
            "随后必须重新采集一套全新的30次Holdout。"
        )
        resample_btn = box.addButton("重采标红区域训练点及周边", QtWidgets.QMessageBox.AcceptRole)
        finish_btn = box.addButton("保留当前56点并结束", QtWidgets.QMessageBox.RejectRole)
        del finish_btn
        box.exec_()
        return box.clickedButton() is resample_btn

    def _prompt_holdout_strategy(self, source) -> str:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("选择 30 次 Holdout 的处理方式")
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setText(
            "56 点训练数据已经准备完成。请选择下一步：\n\n"
            f"• 重新采集全新 30 次：可作为正式独立 Holdout 验收（推荐）。\n"
            f"• 沿用现有 {source.observation_count} 次：立即评估并临时启用，但因这些观测早于本次训练点重采，"
            "只属于诊断结果，不能证明正式验收通过。\n\n"
            "选择沿用时会保留当前 56 点检查点，之后仍可回来重新采集正式 30 次。"
        )
        fresh_btn = box.addButton("重新采集全新 30 次（正式验收）", QtWidgets.QMessageBox.AcceptRole)
        reuse_btn = box.addButton("沿用现有 30 次（仅诊断）", QtWidgets.QMessageBox.ActionRole)
        cancel_btn = box.addButton("暂不处理", QtWidgets.QMessageBox.RejectRole)
        del cancel_btn
        box.setDefaultButton(fresh_btn)
        box.exec_()
        if box.clickedButton() is fresh_btn:
            return "fresh"
        if box.clickedButton() is reuse_btn:
            return "reuse"
        return "cancel"

    def _show_residual_view(self, calibration, training_report, *, holdout_report=None) -> None:
        if self.operator.projection_window is None:
            return
        image = render_projector_ball_residual_view(
            calibration,
            training_report,
            projector_size=(
                int(self.operator.config.projection.projector_width),
                int(self.operator.config.projection.projector_height),
            ),
            holdout_report=holdout_report,
        )
        self.operator.projection_window.set_image(image)
        self._pump_ui()

    def _show_failed_regions(self, calibration, targets: tuple[tuple[float, float], ...]) -> None:
        if self.operator.projection_window is None or not targets:
            return
        ellipses = [
            calibration.table_circle_to_projector_ellipse(
                np.asarray(target, dtype=np.float32),
                0.5 * float(calibration.table.ball_diameter_mm),
            )
            for target in targets
        ]
        image = _render_failed_regions_image(
            (int(self.operator.config.projection.projector_width), int(self.operator.config.projection.projector_height)),
            ellipses,
        )
        self.operator.projection_window.set_image(image)
        self._pump_ui()

    def _show_target(self, target_ellipse, index: int, total: int, *, alert: bool = False) -> None:
        if self.operator.projection_window is None:
            return
        image = _render_target_image(
            (int(self.operator.config.projection.projector_width), int(self.operator.config.projection.projector_height)),
            target_ellipse,
            index,
            total,
            alert=alert,
        )
        self.operator.projection_window.set_image(image)
        self._pump_ui()

    def _collect_training_repeats(
        self,
        capture,
        detector,
        calibration,
        geometry,
        sample_index: int,
        total_count: int,
        target_table_mm: np.ndarray,
        target_proj: np.ndarray,
        expected_cam: np.ndarray,
    ) -> Optional[BallCompensationSample]:
        repeats: list[BallCompensationSample] = []
        for repeat_index in range(ENGINEERED_TRAINING_REPEATS):
            self._append_log(
                f"训练点 {sample_index + 1}/{total_count}，稳定批次 "
                f"{repeat_index + 1}/{ENGINEERED_TRAINING_REPEATS}。"
            )
            sample = self._collect_sample_for_target(
                capture,
                detector,
                calibration,
                geometry,
                sample_index,
                total_count,
                target_table_mm,
                target_proj,
                expected_cam,
            )
            if sample is None:
                return None
            repeats.append(sample)
        return aggregate_training_repeats(repeats, minimum_repeats=ENGINEERED_TRAINING_REPEATS)

    def _collect_sample_for_target(
        self,
        capture,
        detector,
        calibration,
        geometry,
        sample_index: int,
        total_count: int,
        target_table_mm: np.ndarray,
        target_proj: np.ndarray,
        expected_cam: np.ndarray,
        *,
        formal_geometry_required: bool = False,
        allow_skip: bool = True,
    ) -> Optional[BallCompensationSample]:
        while True:
            sample = self._wait_for_stable_sample(
                capture,
                detector,
                calibration,
                geometry,
                sample_index,
                total_count,
                target_table_mm,
                target_proj,
                expected_cam,
                formal_geometry_required=formal_geometry_required,
            )
            if sample is not None:
                return sample
            if self._abort_requested:
                raise _SamplingAborted()
            timeout_instruction = (
                "你可以重试当前点、跳过当前点，或结束本次向导。"
                if allow_skip
                else "正式 Holdout 不允许跳过；你可以重试当前点或结束本次向导。"
            )
            choice = QtWidgets.QMessageBox.question(
                self,
                "采样超时",
                f"第 {sample_index + 1}/{total_count} 个采样点在限定时间内未达到稳定条件。\n"
                f"{timeout_instruction}",
                (
                    QtWidgets.QMessageBox.Retry | QtWidgets.QMessageBox.Ignore | QtWidgets.QMessageBox.Cancel
                    if allow_skip
                    else QtWidgets.QMessageBox.Retry | QtWidgets.QMessageBox.Cancel
                ),
                QtWidgets.QMessageBox.Retry,
            )
            if choice == QtWidgets.QMessageBox.Retry:
                self._append_log(f"重试采样点 {sample_index + 1}/{total_count}。")
                continue
            if allow_skip and choice == QtWidgets.QMessageBox.Ignore:
                self._append_log(f"跳过采样点 {sample_index + 1}/{total_count}。")
                return None
            raise _SamplingAborted()

    def _wait_for_stable_sample(
        self,
        capture,
        detector,
        calibration,
        geometry,
        sample_index: int,
        total_count: int,
        target_table_mm: np.ndarray,
        target_proj: np.ndarray,
        expected_cam: np.ndarray,
        *,
        formal_geometry_required: bool = False,
    ) -> Optional[BallCompensationSample]:
        deadline = time.perf_counter() + ENGINEERED_SAMPLE_TIMEOUT_SECONDS
        history: list[tuple[np.ndarray, float, float, float, str]] = []
        detection_regions = None
        settle_started_at: float | None = None
        last_candidate_at: float | None = None
        last_diagnostic_log_at = 0.0
        while time.perf_counter() < deadline:
            if self._abort_requested:
                raise _SamplingAborted()
            packet = capture.read()
            if packet is None or packet.image is None:
                self._pump_ui()
                time.sleep(0.03)
                continue
            frame = packet.image
            if detection_regions is None:
                detection_regions = _detection_regions_for_frame(frame, geometry, calibration)
            mask_polygon = detection_regions.global_polygon if detection_regions is not None else None
            raw_detections = detector.detect(frame, mask_polygon=mask_polygon)
            raw_ball_count = sum(1 for detection in raw_detections if _looks_like_ball_detection(detection))
            detections = filter_detections_by_region(raw_detections, detection_regions)
            region_ball_count = sum(1 for detection in detections if _looks_like_ball_detection(detection))
            expected_radius_px = _expected_camera_ball_radius(calibration, target_table_mm)
            selection = _select_ball_candidate(
                detections,
                expected_cam,
                expected_radius_px=expected_radius_px,
            )
            candidate = selection.candidate
            candidate_count = selection.size_accepted_count
            distance_px = selection.nearest_distance_px
            if candidate is None:
                now = time.perf_counter()
                if _candidate_dropout_expired(last_candidate_at, now):
                    history.clear()
                    settle_started_at = None
                diagnostic = _ball_candidate_diagnostic_text(
                    selection,
                    raw_ball_count=raw_ball_count,
                    region_ball_count=region_ball_count,
                    expected_radius_px=expected_radius_px,
                )
                self.summary.setText(
                    f"第 {sample_index + 1}/{total_count} 个点等待中：{diagnostic}"
                )
                self._set_preview_image(_annotate_preview(frame, expected_cam, None, candidate_count, distance_px))
                self._pump_ui()
                if now - last_diagnostic_log_at >= 5.0:
                    self._append_log(f"采样点 {sample_index + 1}/{total_count} 候选诊断：{diagnostic}")
                    last_diagnostic_log_at = now
                time.sleep(0.03)
                continue

            center = np.asarray(candidate.center, dtype=np.float32)
            last_candidate_at = time.perf_counter()
            radius = float(candidate.radius_px)
            conf = float(candidate.conf)
            geometry_quality = float(candidate.geometry_quality)
            geometry_method = str(candidate.geometry_method)
            if formal_geometry_required and not ball_holdout_geometry_is_formal(geometry_method):
                history.clear()
                settle_started_at = None
                last_candidate_at = None
                now = time.perf_counter()
                self.summary.setText(
                    f"第 {sample_index + 1}/{total_count} 个 Holdout 点只检测到 bbox 球框；"
                    "正式验收需要完整球心轮廓，正在继续等待。"
                )
                self._set_preview_image(_annotate_preview(frame, expected_cam, candidate, candidate_count, distance_px))
                self._pump_ui()
                if now - last_diagnostic_log_at >= 5.0:
                    self._append_log(
                        f"Holdout 点 {sample_index + 1}/{total_count} 拒绝 bbox："
                        f"geometry_quality={geometry_quality:.2f}。请避免反光或适当降低曝光。"
                    )
                    last_diagnostic_log_at = now
                time.sleep(0.03)
                continue
            if history and np.linalg.norm(center - history[-1][0]) > max(32.0, radius * 1.6):
                history.clear()
                settle_started_at = None
            history.append((center, radius, conf, geometry_quality, geometry_method))
            history = history[-7:]
            stable = _stable_measurement(history)
            self.summary.setText(
                f"第 {sample_index + 1}/{total_count} 个点等待稳定：候选球距目标 {distance_px:.1f}px，已连续观测 {len(history)} 帧。"
            )
            self._set_preview_image(_annotate_preview(frame, expected_cam, candidate, candidate_count, distance_px))
            self._pump_ui()
            if stable is None:
                settle_started_at = None
                time.sleep(0.03)
                continue

            if settle_started_at is None:
                settle_started_at = time.perf_counter()
            remaining_s = ENGINEERED_SAMPLE_SETTLE_DELAY_SECONDS - (time.perf_counter() - settle_started_at)
            if remaining_s > 0.0:
                self.summary.setText(
                    f"第 {sample_index + 1}/{total_count} 个点正在进行 3 秒稳定倒计时，"
                    f"剩余 {remaining_s:.1f}s，当前距离 {distance_px:.1f}px。"
                )
                self._pump_ui()
                time.sleep(0.03)
                continue

            stable_center, stable_radius, stable_confidence, spread_px, stable_quality, stable_method = stable
            observed_table_mm = calibration.camera_px_to_table_mm(np.asarray([stable_center], dtype=np.float32))[0]
            delta_table_mm = target_table_mm - observed_table_mm
            if not ball_sampling_delta_is_plausible(delta_table_mm, calibration.table.ball_diameter_mm):
                delta_norm_mm = float(np.linalg.norm(delta_table_mm))
                history.clear()
                settle_started_at = None
                last_candidate_at = None
                self.summary.setText(
                    f"第 {sample_index + 1}/{total_count} 个点检测到的球仍位于上一目标附近："
                    f"补偿量 {delta_norm_mm:.1f} mm 超过一个球直径，请把球移动到当前目标圈。"
                )
                self._append_log(
                    f"采样点 {sample_index + 1}/{total_count} 拒绝跨点候选："
                    f"delta={delta_norm_mm:.1f} mm。"
                )
                time.sleep(0.03)
                continue
            captured_preview = _annotate_preview(
                frame,
                expected_cam,
                candidate,
                candidate_count,
                distance_px,
                captured=True,
            )
            self._set_preview_image(captured_preview)
            self.summary.setText(
                f"第 {sample_index + 1}/{total_count} 个点采样成功：spread={spread_px:.2f}px，"
                f"delta=({float(delta_table_mm[0]):.2f}, {float(delta_table_mm[1]):.2f}) mm。"
            )
            self._pump_ui()
            time.sleep(0.18)
            return BallCompensationSample(
                sample_index=int(sample_index),
                target_table_mm=(float(target_table_mm[0]), float(target_table_mm[1])),
                detected_camera_px=(float(stable_center[0]), float(stable_center[1])),
                projected_target_px=(float(target_proj[0]), float(target_proj[1])),
                expected_camera_px=(float(expected_cam[0]), float(expected_cam[1])),
                observed_table_mm=(float(observed_table_mm[0]), float(observed_table_mm[1])),
                delta_table_mm=(float(delta_table_mm[0]), float(delta_table_mm[1])),
                detected_radius_px=float(stable_radius),
                detection_confidence=float(stable_confidence),
                stability_spread_px=float(spread_px),
                geometry_quality=float(stable_quality),
                geometry_method=str(stable_method),
                detector_version=str(getattr(detector, "version", "unknown")),
            )
        return None


def _prime_capture_frame(capture, attempts: int = 12) -> Optional[np.ndarray]:
    for _ in range(max(1, int(attempts))):
        packet = capture.read()
        if packet is not None and packet.image is not None and packet.image.size > 0:
            return packet.image
        time.sleep(0.03)
    return None


def _render_target_image(
    projector_size: tuple[int, int],
    target_ellipse,
    index: int,
    total: int,
    *,
    alert: bool = False,
) -> np.ndarray:
    width, height = int(projector_size[0]), int(projector_size[1])
    image = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
    image[:] = (8, 14, 10)
    center = (int(round(float(target_ellipse.center_px[0]))), int(round(float(target_ellipse.center_px[1]))))
    # Keep projected light off the ball itself. A cross through the center
    # changes a solid ball's appearance and can make both YOLO segmentation and
    # contour refinement fail. The surrounding ring and ticks still provide a
    # precise physical centering guide.
    radius_x, radius_y = _target_guide_radii(target_ellipse)
    guide_color = (40, 40, 255) if alert else (80, 255, 160)
    for start, end in ((-12.0, 12.0), (78.0, 102.0), (168.0, 192.0), (258.0, 282.0)):
        cv2.ellipse(
            image,
            center,
            (radius_x, radius_y),
            float(target_ellipse.rotation_deg),
            start,
            end,
            guide_color,
            3,
            cv2.LINE_AA,
        )
    cx, cy = center
    tick_gap = 7
    tick_length = 14
    tick_color = (255, 255, 255)
    cv2.line(image, (cx - radius_x - tick_gap - tick_length, cy), (cx - radius_x - tick_gap, cy), tick_color, 2, cv2.LINE_AA)
    cv2.line(image, (cx + radius_x + tick_gap, cy), (cx + radius_x + tick_gap + tick_length, cy), tick_color, 2, cv2.LINE_AA)
    cv2.line(image, (cx, cy - radius_y - tick_gap - tick_length), (cx, cy - radius_y - tick_gap), tick_color, 2, cv2.LINE_AA)
    cv2.line(image, (cx, cy + radius_y + tick_gap), (cx, cy + radius_y + tick_gap + tick_length), tick_color, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"Sample {index}/{total}",
        (32, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "RESAMPLE FAILED REGION" if alert else "Move the single ball into the ring",
        (32, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 80, 255) if alert else (180, 255, 220),
        2,
        cv2.LINE_AA,
    )
    return image


def _render_failed_regions_image(projector_size: tuple[int, int], target_ellipses) -> np.ndarray:
    width, height = int(projector_size[0]), int(projector_size[1])
    image = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
    image[:] = (8, 10, 18)
    for index, ellipse in enumerate(target_ellipses, start=1):
        center = (int(round(float(ellipse.center_px[0]))), int(round(float(ellipse.center_px[1]))))
        cv2.ellipse(
            image,
            center,
            _target_guide_radii(ellipse),
            float(ellipse.rotation_deg),
            0.0,
            360.0,
            (30, 30, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(index),
            (center[0] + 12, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        "FAILED HOLDOUT REGIONS",
        (32, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (40, 40, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _target_guide_radii(target_ellipse) -> tuple[int, int]:
    """Keep the centering guide close to the projected physical ball edge."""

    return (
        int(round(max(6.0, float(target_ellipse.radius_x_px) + 2.0))),
        int(round(max(6.0, float(target_ellipse.radius_y_px) + 2.0))),
    )


def _ball_compensation_completion_summary(
    *,
    accepted_count: int,
    total_count: int,
    training_count: int,
    holdout_count: int,
    training_report: dict,
    holdout_report: dict,
    save_path: str | Path,
) -> str:
    mapping = training_report.get("mapping_cross_validation", {})
    delta = training_report.get("delta_norm_mm", {})
    holdout_error = holdout_report.get("error_mm", {})
    repeatability_error = holdout_report.get("repeatability", {}).get("error_mm", {})
    mean_vector = np.asarray(holdout_report.get("mean_vector_mm", [0.0, 0.0]), dtype=np.float64).reshape((-1,))
    bias_mm = float(np.linalg.norm(mean_vector[:2])) if mean_vector.size >= 2 else 0.0
    repeatability_line = ""
    if repeatability_error:
        repeatability_line = (
            f"独立复采离散度 median={float(repeatability_error.get('median', 0.0)):.2f} mm | "
            f"P95={float(repeatability_error.get('p95', 0.0)):.2f} mm\n"
        )
    return (
        "工程球体补偿采样完成。\n"
        f"有效采样: {accepted_count}/{total_count} | 训练={training_count} 独立Holdout={holdout_count}\n"
        f"模型={mapping.get('model_kind', 'unknown')} | "
        f"训练CV P95={float(mapping.get('p95_mm', 0.0)):.2f}/"
        f"{float(mapping.get('maximum_p95_mm', 0.0)):.2f} mm\n"
        f"Holdout P95={float(holdout_error.get('p95', 0.0)):.2f}/"
        f"{float(holdout_report.get('maximum_p95_mm', 0.0)):.2f} mm | "
        f"平均偏差={bias_mm:.2f}/{float(holdout_report.get('maximum_mean_bias_mm', 0.0)):.2f} mm | "
        f"验收={'通过' if holdout_report.get('quality_gate_passed') else '未通过'}\n"
        f"{repeatability_line}"
        f"原始补偿量 delta P95={float(delta.get('p95', 0.0)):.2f} mm\n"
        f"保存路径: {save_path}"
    )


def _ball_compensation_diagnostic_summary(
    *,
    training_report: dict,
    holdout_report: dict,
    save_path: str | Path,
    residual_view_path: str | Path,
) -> str:
    training_error = dict(training_report.get("error_mm", {}))
    holdout_error = dict(holdout_report.get("error_mm", {}))
    observation_count = int(holdout_report.get("observation_count", holdout_report.get("sample_count", 0)))
    return (
        "新 7 点已并入 56 点模型，现有 30 次 Holdout 已用于诊断。\n"
        "注意：旧 Holdout 早于本次训练点重采，因此不能作为正式独立验收；当前 56 点检查点已保留。\n"
        f"训练残差 mean={float(training_error.get('mean', 0.0)):.2f} mm | "
        f"median={float(training_error.get('median', 0.0)):.2f} mm | "
        f"P95={float(training_error.get('p95', 0.0)):.2f} mm | "
        f"max={float(training_error.get('max', 0.0)):.2f} mm\n"
        f"训练点 ≥1/≥2/≥5 mm：{int(training_report.get('count_ge_1mm', 0))}/"
        f"{int(training_report.get('count_ge_2mm', 0))}/{int(training_report.get('count_ge_5mm', 0))}\n"
        f"沿用 Holdout={observation_count} 次/{int(holdout_report.get('sample_count', 0))} 个位置 | "
        f"median={float(holdout_error.get('median', 0.0)):.2f} mm | "
        f"P95={float(holdout_error.get('p95', 0.0)):.2f} mm | "
        f"max={float(holdout_error.get('max', 0.0)):.2f} mm\n"
        f"临时补偿文件: {save_path}\n"
        f"全桌残差视图: {residual_view_path}"
    )


def _detection_regions_for_frame(frame_bgr: np.ndarray, geometry, calibration):
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    fallback_polygon = None
    poly = getattr(getattr(calibration, "projection", None), "table_polygon_cam", None)
    if poly is not None:
        poly_np = np.asarray(poly, dtype=np.float32).reshape((-1, 2))
        if poly_np.shape[0] >= 3:
            fallback_polygon = poly_np.astype(np.float32)
    policy = build_detection_region_policy(frame_bgr.shape, geometry, fallback_polygon=fallback_polygon)
    if policy.global_polygon is None and policy.ball_polygon is None and policy.cue_stick_polygon is None:
        return None
    return policy


def _expected_camera_ball_radius(calibration, target_table_mm: np.ndarray) -> float:
    center = np.asarray(target_table_mm, dtype=np.float32).reshape((2,))
    radius_mm = 0.5 * float(calibration.table.ball_diameter_mm)
    references = np.asarray(
        [center, center + np.asarray([radius_mm, 0.0], dtype=np.float32), center + np.asarray([0.0, radius_mm], dtype=np.float32)],
        dtype=np.float32,
    )
    camera = calibration.table_mm_to_camera_px(references)
    radii = [float(np.linalg.norm(camera[1] - camera[0])), float(np.linalg.norm(camera[2] - camera[0]))]
    return max(2.0, float(np.median(radii)))


@dataclass(frozen=True)
class BallCandidateSelection:
    candidate: Optional[Detection]
    ball_class_count: int
    geometry_accepted_count: int
    size_accepted_count: int
    nearest_distance_px: float


def _select_ball_candidate(
    detections: list[Detection],
    expected_cam: np.ndarray,
    *,
    expected_radius_px: float,
) -> BallCandidateSelection:
    ball_detections = [detection for detection in detections if _looks_like_ball_detection(detection)]
    geometry_accepted = [
        detection
        for detection in ball_detections
        if ball_sampling_geometry_is_usable(detection, expected_cam, expected_radius_px)
    ]
    size_accepted = [
        detection
        for detection in geometry_accepted
        if 0.50 <= float(detection.radius_px) / max(2.0, float(expected_radius_px)) <= 1.70
    ]
    if not size_accepted:
        return BallCandidateSelection(
            candidate=None,
            ball_class_count=len(ball_detections),
            geometry_accepted_count=len(geometry_accepted),
            size_accepted_count=0,
            nearest_distance_px=float("inf"),
        )
    expected = np.asarray(expected_cam, dtype=np.float32).reshape((2,))
    best = None
    best_distance = float("inf")
    for detection in size_accepted:
        center = np.asarray(detection.center, dtype=np.float32)
        distance = float(np.linalg.norm(center - expected))
        if distance < best_distance:
            best = detection
            best_distance = distance
    max_distance = ball_sampling_target_distance_limit_px(expected_radius_px)
    if best_distance > max_distance:
        best = None
    return BallCandidateSelection(
        candidate=best,
        ball_class_count=len(ball_detections),
        geometry_accepted_count=len(geometry_accepted),
        size_accepted_count=len(size_accepted),
        nearest_distance_px=best_distance,
    )


def _pick_ball_candidate(
    detections: list[Detection],
    expected_cam: np.ndarray,
    *,
    expected_radius_px: float,
) -> tuple[Optional[Detection], int, float]:
    selection = _select_ball_candidate(
        detections,
        expected_cam,
        expected_radius_px=expected_radius_px,
    )
    return selection.candidate, selection.size_accepted_count, selection.nearest_distance_px


def _ball_candidate_diagnostic_text(
    selection: BallCandidateSelection,
    *,
    raw_ball_count: int,
    region_ball_count: int,
    expected_radius_px: float,
) -> str:
    prefix = (
        f"YOLO球={raw_ball_count}，区域内={region_ball_count}，"
        f"几何可用={selection.geometry_accepted_count}，尺寸可用={selection.size_accepted_count}"
    )
    if raw_ball_count <= 0:
        return prefix + "；模型没有识别到球，请确认球体未被投影亮线覆盖并检查检测模型。"
    if region_ball_count <= 0:
        return prefix + "；球被台面检测区域过滤，请检查几何标注。"
    if selection.geometry_accepted_count <= 0:
        return prefix + "；球框存在但中心/轮廓质量不足。"
    if selection.size_accepted_count <= 0:
        return prefix + f"；检测球半径与预期 {expected_radius_px:.1f}px 不匹配。"
    if np.isfinite(selection.nearest_distance_px):
        return prefix + f"；最近球距预期点 {selection.nearest_distance_px:.1f}px，尚未进入采样范围。"
    return prefix + "；没有可用候选球。"


def _looks_like_ball_detection(detection: Detection) -> bool:
    cls_name = str(detection.cls_name or "").strip().lower()
    if not cls_name:
        return False
    if group_from_class(cls_name) not in {"cue", "solid", "stripe", "black"}:
        return False
    x1, y1, x2, y2 = detection.bbox
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    return width >= 4.0 and height >= 4.0


def _stable_measurement(
    history: list[tuple[np.ndarray, float, float, float, str]],
) -> Optional[tuple[np.ndarray, float, float, float, float, str]]:
    if len(history) < 5:
        return None
    recent = history[-7:]
    recent_methods = [str(item[4]).strip().lower() for item in recent]
    robust_bbox_window = len(recent) >= 7 and all(method.startswith("bbox") for method in recent_methods)
    if robust_bbox_window:
        # Edge locations can produce a good YOLO box with an occasional box
        # regression jump even while the ball is stationary.  Training already
        # downweights bbox geometry; trim only the two farthest frames so one
        # transient jump cannot restart the three-second settle timer forever.
        all_centers = np.asarray([item[0] for item in recent], dtype=np.float32).reshape((-1, 2))
        provisional_center = np.median(all_centers, axis=0)
        distances = np.linalg.norm(all_centers - provisional_center.reshape((1, 2)), axis=1)
        kept = np.argsort(distances, kind="stable")[:5]
        window = [recent[int(index)] for index in kept]
    else:
        window = history[-5:]
    centers = np.asarray([item[0] for item in window], dtype=np.float32).reshape((-1, 2))
    radii = np.asarray([item[1] for item in window], dtype=np.float32).reshape((-1,))
    confidences = np.asarray([item[2] for item in window], dtype=np.float32).reshape((-1,))
    qualities = np.asarray([item[3] for item in window], dtype=np.float32).reshape((-1,))
    methods = [str(item[4]) for item in window]
    median_center = np.median(centers, axis=0).astype(np.float32)
    spread = float(np.max(np.linalg.norm(centers - median_center.reshape((1, 2)), axis=1)))
    allowed_spread = max(3.5, 0.18 * float(np.median(radii)))
    if spread > allowed_spread:
        return None
    stable_method = max(dict.fromkeys(methods), key=methods.count)
    return (
        median_center,
        float(np.median(radii)),
        float(np.mean(confidences)),
        spread,
        float(np.median(qualities)),
        stable_method,
    )


def _candidate_dropout_expired(last_candidate_at: float | None, now: float) -> bool:
    """Reset stability only after a real detection gap, not one missed frame."""

    if last_candidate_at is None:
        return True
    return float(now) - float(last_candidate_at) > ENGINEERED_SAMPLE_DROPOUT_GRACE_SECONDS


def _annotate_preview(
    frame_bgr: np.ndarray,
    expected_cam: np.ndarray,
    candidate: Optional[Detection],
    candidate_count: int,
    distance_px: float,
    captured: bool = False,
) -> np.ndarray:
    image = frame_bgr.copy()
    expected = (int(round(float(expected_cam[0]))), int(round(float(expected_cam[1]))))
    cv2.drawMarker(image, expected, (0, 255, 255), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
    if candidate is not None:
        center = candidate.center
        radius = int(round(float(candidate.radius_px)))
        color = (0, 220, 80) if captured else (0, 180, 255)
        cv2.circle(image, (int(round(center[0])), int(round(center[1]))), max(8, radius), color, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"expected=({expected[0]}, {expected[1]})  balls={candidate_count}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if np.isfinite(distance_px):
        cv2.putText(
            image,
            f"distance={distance_px:.1f}px" + ("  captured" if captured else ""),
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (120, 255, 200) if captured else (255, 220, 120),
            2,
            cv2.LINE_AA,
        )
    return image
