from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PyQt5 import QtCore, QtWidgets


@dataclass(frozen=True)
class CompleteCalibrationRunResult:
    stage: str
    projection_completed: bool
    ball_compensation_completed: bool

    @property
    def completed(self) -> bool:
        return self.projection_completed and self.ball_compensation_completed


def run_complete_calibration_steps(
    run_projection: Callable[[], bool],
    confirm_ball_setup: Callable[[], bool],
    run_ball_compensation: Callable[[], bool],
) -> CompleteCalibrationRunResult:
    """Run the two independent calibration phases as one operator workflow."""
    if not bool(run_projection()):
        return CompleteCalibrationRunResult("projection_failed", False, False)
    if not bool(confirm_ball_setup()):
        return CompleteCalibrationRunResult("ball_setup_cancelled", True, False)
    if not bool(run_ball_compensation()):
        return CompleteCalibrationRunResult("ball_compensation_failed", True, False)
    return CompleteCalibrationRunResult("complete", True, True)


class CompleteCalibrationWizardDialog(QtWidgets.QDialog):
    """Single entry point for projection and ball-center calibration."""

    def __init__(self, operator, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent or operator)
        self.operator = operator
        self._busy = False
        self.result_state: Optional[CompleteCalibrationRunResult] = None
        self.setWindowTitle("完整几何校准向导")
        self.resize(720, 520)

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("相机—投影仪联合校准 + 球心补偿")
        title.setObjectName("title")
        layout.addWidget(title)

        intro = QtWidgets.QLabel(
            "向导会连续完成两个校准阶段。联合校准建立台面平面映射；"
            "球心补偿随后使用真实球修正球体高度和检测中心偏差。"
            "两套模型仍会独立计算、保存和执行质量验收。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        phase_box = QtWidgets.QGroupBox("流程")
        phase_layout = QtWidgets.QGridLayout(phase_box)
        phase_layout.addWidget(QtWidgets.QLabel("1. 联合校准"), 0, 0)
        self.projection_state = QtWidgets.QLabel("等待开始")
        phase_layout.addWidget(self.projection_state, 0, 1)
        phase_layout.addWidget(QtWidgets.QLabel("2. 球心补偿"), 1, 0)
        self.ball_state = QtWidgets.QLabel("等待联合校准")
        phase_layout.addWidget(self.ball_state, 1, 1)
        layout.addWidget(phase_box)

        preparation = QtWidgets.QPlainTextEdit()
        preparation.setReadOnly(True)
        preparation.setMaximumHeight(130)
        preparation.setPlainText(
            "开始前：\n"
            "1. 固定相机、投影仪和球桌，确认检测模型已经启用。\n"
            "2. 联合校准阶段保持台面无遮挡。\n"
            "3. 联合校准完成后，向导会暂停并提示清空台面，只保留一颗标准球。"
        )
        layout.addWidget(preparation)

        self.summary = QtWidgets.QLabel("尚未开始完整校准。")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        layout.addStretch(1)

        buttons = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("开始完整校准")
        self.start_btn.clicked.connect(self.run_calibration)
        buttons.addWidget(self.start_btn)
        buttons.addStretch(1)
        self.close_btn = QtWidgets.QPushButton("关闭")
        self.close_btn.clicked.connect(self.reject)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.start_btn.setEnabled(not self._busy)
        self.close_btn.setEnabled(not self._busy)

    def reject(self) -> None:
        if self._busy:
            return
        super().reject()

    def _confirm_ball_setup(self) -> bool:
        self.ball_state.setText("等待现场准备")
        answer = QtWidgets.QMessageBox.question(
            self,
            "准备球心补偿",
            "联合校准已经完成。\n\n"
            "请移除校准物，清空台面并只保留一颗标准球。准备完成后选择“是”，"
            "向导将自动开始球心补偿；选择“否”可稍后从工作台单独继续。",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        return answer == QtWidgets.QMessageBox.Yes

    @QtCore.pyqtSlot()
    def run_calibration(self) -> None:
        if self._busy:
            return
        detector_backend = str(getattr(self.operator.config.detector, "backend", "disabled") or "disabled").lower()
        if detector_backend == "disabled":
            QtWidgets.QMessageBox.information(
                self,
                "检测模型未启用",
                "完整校准需要球检测模型。请先在设置中启用检测后端。",
            )
            return

        self._set_busy(True)
        self.result_state = None
        self.projection_state.setText("进行中")
        self.ball_state.setText("等待联合校准")
        self.summary.setText("正在进行相机—投影仪联合校准。")
        try:
            def run_projection() -> bool:
                succeeded = bool(self.operator.run_linked_projector_calibration(auto_start=True))
                self.projection_state.setText("已完成" if succeeded else "失败或取消")
                return succeeded

            def run_ball() -> bool:
                self.ball_state.setText("进行中")
                self.summary.setText("正在进行球心补偿，请按投影目标移动标准球。")
                succeeded = bool(self.operator.run_engineered_ball_compensation_wizard(auto_start=True))
                self.ball_state.setText("已完成" if succeeded else "失败或取消")
                return succeeded

            self.result_state = run_complete_calibration_steps(
                run_projection,
                self._confirm_ball_setup,
                run_ball,
            )
            if self.result_state.completed:
                self.summary.setText("完整几何校准已完成：联合校准和球心补偿均已通过并保存。")
                QtWidgets.QMessageBox.information(self, "校准完成", self.summary.text())
            elif self.result_state.stage == "ball_setup_cancelled":
                self.ball_state.setText("稍后继续")
                self.summary.setText("联合校准已保存；球心补偿尚未执行，可稍后从校准工作台单独继续。")
            elif self.result_state.stage == "projection_failed":
                self.summary.setText("联合校准未完成，完整校准已经停止。")
            else:
                self.summary.setText("联合校准已保存，但球心补偿未完成；请检查提示后重试球心补偿。")
        except Exception as exc:
            self.summary.setText(f"完整校准中止：{exc}")
            QtWidgets.QMessageBox.critical(self, "完整校准中止", str(exc))
        finally:
            self._set_busy(False)

