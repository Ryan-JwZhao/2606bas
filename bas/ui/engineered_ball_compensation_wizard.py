from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..calibration import (
    BallCompensationSample,
    build_ball_compensation_model,
    build_engineered_ball_sampling_grid,
    create_calibration_service,
    update_calibration_table_boundaries_from_geometry_frame,
)
from ..capture import create_capture_service
from ..geometry import TableGeometryLoader
from ..paths import PROJECT_ROOT
from ..perception import create_detector
from ..schemas import Detection
from ..table_boundaries import EdgeInsets

TIMESTAMPED_BALL_COMPENSATION_FILE_RE = re.compile(r"^(?P<base>.*?)(?:_\d{8}_\d{6})?$")
DEFAULT_BALL_COMPENSATION_OUTPUT_DIR = PROJECT_ROOT / "local_settings" / "calibrations"


class _SamplingAborted(RuntimeError):
    pass


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
    def __init__(self, operator, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent or operator)
        self.operator = operator
        self._busy = False
        self._abort_requested = False
        self._saved_path: Optional[Path] = None
        self._samples: list[BallCompensationSample] = []
        self.setWindowTitle("工程球体补偿自动采样向导")
        self.resize(980, 820)

        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "该向导会按预设采样网格自动投影目标圈。请先切到工程模式，并确保工程平面校准文件与球检测模型都有效。"
            "采样时建议清空台面，仅保留一颗球；把球移动到目标圈后，系统会自动等待位置稳定并记录样本。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        steps = QtWidgets.QPlainTextEdit()
        steps.setReadOnly(True)
        steps.setPlainText(
            "\n".join(
                [
                    "1. 在设置中切换到工程模式，并确认当前工程平面校准文件可加载。",
                    "2. 启用可用的球检测后端，并确保当前相机画面已经完成去畸变。",
                    "3. 清空台面，仅保留一颗标准球。",
                    "4. 点击“开始自动采样”，按提示把球逐个移动到投影目标圈。",
                    "5. 采样完成后，程序会自动生成新的工程球体补偿 JSON，并写回当前设置。",
                ]
            )
        )
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
        self.start_btn = QtWidgets.QPushButton("开始自动采样")
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

    @QtCore.pyqtSlot()
    def run_sampling(self) -> None:
        if self._busy:
            return
        capture = None
        self._abort_requested = False
        self._saved_path = None
        self._samples = []
        self._set_busy(True)
        self.progress.setValue(0)
        self.summary.setText("正在准备工程球体补偿采样。")
        self._append_log("工程球体补偿自动采样开始，正在校验工程模式和检测链路。")
        try:
            self.operator._sync_config_from_controls()
            self.operator._save_user_settings()
            if self.operator.config.calibration.normalized_projection_mode() != "engineered":
                raise RuntimeError("请先在设置中切换到工程模式，再运行工程球体补偿向导。")
            if self.operator.pipeline is not None:
                self._append_log("检测到实时采集正在运行，先自动停止以释放相机。")
                self.operator.stop_pipeline()
            capture = create_capture_service(self.operator.config.camera)
            calibration = create_calibration_service(
                self.operator.config.calibration,
                frame_undistorted=bool(capture.frame_distortion_corrected),
            )
            if not calibration.projection.is_valid:
                plane_path = self.operator.config.calibration.engineered_plane_projection_file or self.operator.config.calibration.active_projection_file()
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
                    "当前工程模式的平面投影校准文件无效，无法继续进行球体补偿采样。"
                    f"{same_path_hint}"
                )
            detector = create_detector(self.operator.config.detector)
            if getattr(detector, "version", "disabled") == "disabled":
                raise RuntimeError("工程球体补偿向导需要启用球检测后端，当前 detector.backend=disabled。")
            geometry = TableGeometryLoader.load_optional(
                self.operator.config.geometry.outline_path,
                self.operator.config.geometry.inline_path,
                self.operator.config.geometry.pocket_path,
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
            sample_polygon = np.asarray(
                calibration.table.center_playable_polygon_mm or calibration.table.inner_polygon_mm,
                dtype=np.float32,
            ).reshape((-1, 2))
            sample_points = build_engineered_ball_sampling_grid(
                calibration.table.width_mm,
                calibration.table.height_mm,
                calibration.table.ball_diameter_mm,
                preferred_polygon_mm=sample_polygon,
                extra_safe_inset_mm=0.5 * float(calibration.table.ball_diameter_mm),
            )
            self._append_log(f"已生成 {len(sample_points)} 个工程采样点，请按投影目标圈移动单颗球。")

            total = max(1, len(sample_points))
            for idx, target_table_mm in enumerate(sample_points):
                if self._abort_requested:
                    raise _SamplingAborted()
                percent = int(round(idx / total * 100.0))
                self.progress.setValue(percent)
                target_arr = np.asarray([target_table_mm], dtype=np.float32)
                target_proj = calibration.table_mm_to_projector_px(target_arr)[0]
                expected_cam = calibration.table_mm_to_camera_px(target_arr)[0]
                target_radius_px = calibration.table_radius_to_projector_px(
                    target_table_mm.astype(np.float32),
                    0.5 * float(calibration.table.ball_diameter_mm),
                )
                self._show_target(target_proj, target_radius_px, idx + 1, total)
                self.summary.setText(
                    f"正在等待第 {idx + 1}/{total} 个点稳定。请把单颗球移到目标圈中央，系统检测到稳定后会自动跳到下一点。"
                )
                self._append_log(
                    f"切换到采样点 {idx + 1}/{total}: table=({target_table_mm[0]:.1f}, {target_table_mm[1]:.1f}) mm"
                )
                outcome = self._collect_sample_for_target(
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
                    continue
                self._samples.append(outcome)
                self._append_log(
                    "采样成功: "
                    f"camera=({outcome.detected_camera_px[0]:.1f}, {outcome.detected_camera_px[1]:.1f}) px, "
                    f"delta=({outcome.delta_table_mm[0]:.2f}, {outcome.delta_table_mm[1]:.2f}) mm"
                )

            if len(self._samples) < 4:
                raise RuntimeError(f"有效采样点只有 {len(self._samples)} 个，至少需要 4 个点才能生成稳定补偿文件。")

            model = build_ball_compensation_model(
                self._samples,
                ball_diameter_mm=calibration.table.ball_diameter_mm,
            )
            save_template = self._safe_ball_compensation_output_path(calibration)
            save_path = timestamped_ball_compensation_output_path(str(save_template))
            model.save_json(
                save_path,
                extra_data={
                    "ball_diameter_mm": float(calibration.table.ball_diameter_mm),
                    "projection_mode": "engineered",
                    "projection_file": calibration.projection.source_path,
                    "samples": [sample.to_dict() for sample in self._samples],
                    "sampling_grid_table_mm": np.asarray(sample_points, dtype=np.float64).reshape((-1, 2)).tolist(),
                },
            )
            self._saved_path = save_path
            self.operator.config.calibration.engineered_ball_compensation_file = str(save_path)
            self.operator.config.calibration.sync_projection_file_alias()
            self.operator._sync_controls_from_config()
            self.operator._save_user_settings()
            self.operator._update_module_status()
            delta_report = model.quality_report.get("delta_norm_mm", {})
            self.progress.setValue(100)
            self.summary.setText(
                "工程球体补偿采样完成。\n"
                f"有效采样: {len(self._samples)}/{len(sample_points)}\n"
                f"delta p95={float(delta_report.get('p95', 0.0)):.2f} mm, "
                f"delta max={float(delta_report.get('max', 0.0)):.2f} mm\n"
                f"保存路径: {save_path}"
            )
            self.output_path_edit.setText(str(save_path))
            self._append_log(f"工程球体补偿文件已生成并写回当前设置: {save_path}")
        except _SamplingAborted:
            self.summary.setText("工程球体补偿采样已停止，未生成新的补偿文件。")
            self._append_log("工程球体补偿采样已停止。")
        except Exception as exc:
            self.summary.setText(f"工程球体补偿采样失败: {exc}")
            self._append_log(f"工程球体补偿采样失败: {exc}")
            QtWidgets.QMessageBox.critical(self, "工程球体补偿采样失败", str(exc))
        finally:
            if capture is not None:
                capture.release()
            self._set_busy(False)

    def _show_target(self, target_proj: np.ndarray, target_radius_px: float, index: int, total: int) -> None:
        if self.operator.projection_window is None:
            return
        image = _render_target_image(
            (int(self.operator.config.projection.projector_width), int(self.operator.config.projection.projector_height)),
            target_proj,
            target_radius_px,
            index,
            total,
        )
        self.operator.projection_window.set_image(image)
        self._pump_ui()

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
            )
            if sample is not None:
                return sample
            if self._abort_requested:
                raise _SamplingAborted()
            choice = QtWidgets.QMessageBox.question(
                self,
                "采样超时",
                f"第 {sample_index + 1}/{total_count} 个采样点在限定时间内未达到稳定条件。\n"
                "你可以重试当前点、跳过当前点，或结束本次向导。",
                QtWidgets.QMessageBox.Retry | QtWidgets.QMessageBox.Ignore | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Retry,
            )
            if choice == QtWidgets.QMessageBox.Retry:
                self._append_log(f"重试采样点 {sample_index + 1}/{total_count}。")
                continue
            if choice == QtWidgets.QMessageBox.Ignore:
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
    ) -> Optional[BallCompensationSample]:
        deadline = time.perf_counter() + 16.0
        history: list[tuple[np.ndarray, float, float]] = []
        detection_mask = None
        while time.perf_counter() < deadline:
            if self._abort_requested:
                raise _SamplingAborted()
            packet = capture.read()
            if packet is None or packet.image is None:
                self._pump_ui()
                time.sleep(0.03)
                continue
            frame = packet.image
            if detection_mask is None:
                detection_mask = _table_mask_for_frame(frame, geometry, calibration)
            detections = detector.detect(frame, mask_polygon=detection_mask)
            candidate, candidate_count, distance_px = _pick_ball_candidate(detections, expected_cam)
            if candidate is None:
                history.clear()
                self.summary.setText(
                    f"第 {sample_index + 1}/{total_count} 个点等待中：未检测到接近目标圈的球。"
                )
                self._set_preview_image(_annotate_preview(frame, expected_cam, None, candidate_count, distance_px))
                self._pump_ui()
                time.sleep(0.03)
                continue

            center = np.asarray(candidate.center, dtype=np.float32)
            radius = float(candidate.radius_px)
            conf = float(candidate.conf)
            if history and np.linalg.norm(center - history[-1][0]) > max(32.0, radius * 1.6):
                history.clear()
            history.append((center, radius, conf))
            history = history[-5:]
            stable = _stable_measurement(history)
            self.summary.setText(
                f"第 {sample_index + 1}/{total_count} 个点等待稳定：候选球距目标 {distance_px:.1f}px，已连续观测 {len(history)} 帧。"
            )
            self._set_preview_image(_annotate_preview(frame, expected_cam, candidate, candidate_count, distance_px))
            self._pump_ui()
            if stable is None:
                time.sleep(0.03)
                continue

            stable_center, stable_radius, stable_confidence, spread_px = stable
            observed_table_mm = calibration.camera_px_to_table_mm(np.asarray([stable_center], dtype=np.float32))[0]
            delta_table_mm = target_table_mm - observed_table_mm
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
    target_proj: np.ndarray,
    target_radius_px: float,
    index: int,
    total: int,
) -> np.ndarray:
    width, height = int(projector_size[0]), int(projector_size[1])
    image = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
    image[:] = (8, 14, 10)
    center = (int(round(float(target_proj[0]))), int(round(float(target_proj[1]))))
    outer_radius = int(round(max(18.0, float(target_radius_px))))
    inner_radius = max(6, int(round(outer_radius * 0.35)))
    cv2.circle(image, center, outer_radius, (80, 255, 160), 3, cv2.LINE_AA)
    cv2.circle(image, center, inner_radius, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(image, center, (255, 255, 255), cv2.MARKER_CROSS, max(22, outer_radius + 10), 2, cv2.LINE_AA)
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
        "Move the single ball into the ring",
        (32, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 255, 220),
        2,
        cv2.LINE_AA,
    )
    return image


def _table_mask_for_frame(frame_bgr: np.ndarray, geometry, calibration) -> Optional[np.ndarray]:
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    height, width = frame_bgr.shape[:2]
    if geometry is not None and not geometry.is_empty:
        outer, _, _ = geometry.scaled(width, height)
        if outer.shape[0] >= 3:
            return outer.astype(np.float32)
    poly = np.asarray(calibration.projection.table_polygon_cam, dtype=np.float32).reshape((-1, 2))
    if poly.shape[0] >= 3:
        return poly.astype(np.float32)
    return None


def _pick_ball_candidate(detections: list[Detection], expected_cam: np.ndarray) -> tuple[Optional[Detection], int, float]:
    ball_detections = [detection for detection in detections if _looks_like_ball_detection(detection)]
    if not ball_detections:
        return None, 0, float("inf")
    expected = np.asarray(expected_cam, dtype=np.float32).reshape((2,))
    best = None
    best_distance = float("inf")
    for detection in ball_detections:
        center = np.asarray(detection.center, dtype=np.float32)
        distance = float(np.linalg.norm(center - expected))
        if distance < best_distance:
            best = detection
            best_distance = distance
    if best is None:
        return None, len(ball_detections), float("inf")
    max_distance = max(75.0, 4.0 * float(best.radius_px))
    if best_distance > max_distance:
        return None, len(ball_detections), best_distance
    return best, len(ball_detections), best_distance


def _looks_like_ball_detection(detection: Detection) -> bool:
    cls_name = str(detection.cls_name or "").strip().lower()
    if not cls_name:
        return False
    if "stick" in cls_name:
        return False
    x1, y1, x2, y2 = detection.bbox
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    return width >= 4.0 and height >= 4.0


def _stable_measurement(history: list[tuple[np.ndarray, float, float]]) -> Optional[tuple[np.ndarray, float, float, float]]:
    if len(history) < 3:
        return None
    centers = np.asarray([item[0] for item in history[-3:]], dtype=np.float32).reshape((-1, 2))
    radii = np.asarray([item[1] for item in history[-3:]], dtype=np.float32).reshape((-1,))
    confidences = np.asarray([item[2] for item in history[-3:]], dtype=np.float32).reshape((-1,))
    median_center = np.median(centers, axis=0).astype(np.float32)
    spread = float(np.max(np.linalg.norm(centers - median_center.reshape((1, 2)), axis=1)))
    allowed_spread = max(3.5, 0.18 * float(np.median(radii)))
    if spread > allowed_spread:
        return None
    return median_center, float(np.median(radii)), float(np.mean(confidences)), spread


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
