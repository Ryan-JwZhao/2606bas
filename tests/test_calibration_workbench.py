from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from bas.config import AppConfig
from bas import runtime_env

runtime_env.preload_torch_for_backend = lambda _backend: None

from bas.ui.calibration_workbench import CalibrationWorkbenchDialog, assess_calibration_workbench


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _calibration(*, projection_valid: bool, ball_valid: bool):
    return SimpleNamespace(
        distortion_correction_enabled=False,
        projection=SimpleNamespace(
            is_valid=projection_valid,
            compatibility_errors=() if projection_valid else ("frame_rotation_degrees",),
        ),
        ball_compensation_model=SimpleNamespace(
            is_valid=ball_valid,
            compatibility_errors=() if ball_valid else ("projection_fingerprint",),
        ),
    )


def test_workbench_keeps_camera_and_joint_calibration_available_without_projection() -> None:
    status = assess_calibration_workbench(
        _calibration(projection_valid=False, ball_valid=False),
        detector_enabled=True,
        camera_running=False,
    )

    assert status.stage == "joint_calibration"
    assert status.projection_valid is False
    assert status.ball_calibration_enabled is False
    assert status.verification_enabled is False
    assert "联合校准" in status.next_action


def test_workbench_gui_disables_only_geometry_dependent_actions_before_calibration() -> None:
    app = _app()
    calls: list[str] = []
    operator = SimpleNamespace(
        config=AppConfig(),
        pipeline=None,
        start_pipeline=lambda: calls.append("start"),
        stop_pipeline=lambda: calls.append("stop"),
        open_settings=lambda: calls.append("settings"),
        run_linked_projector_calibration=lambda: calls.append("joint"),
        run_engineered_ball_compensation_wizard=lambda: calls.append("ball"),
        show_projector_calibration_result=lambda _calibration: calls.append("show"),
        show_projector_residual_overlay=lambda _calibration: calls.append("residual"),
    )
    parent = QtWidgets.QWidget()
    dialog = CalibrationWorkbenchDialog(
        operator,
        parent,
        service_factory=lambda: _calibration(projection_valid=False, ball_valid=False),
    )

    assert dialog.start_camera_btn.isEnabled() is True
    assert dialog.joint_btn.isEnabled() is True
    assert dialog.ball_btn.isEnabled() is False
    assert dialog.verify_btn.isEnabled() is False
    assert "180" not in dialog.projection_state.text()

    dialog.close()
    parent.close()
    app.processEvents()


def test_workbench_enables_ball_and_holdout_after_joint_calibration() -> None:
    status = assess_calibration_workbench(
        _calibration(projection_valid=True, ball_valid=False),
        detector_enabled=True,
        camera_running=True,
    )

    assert status.stage == "ball_compensation"
    assert status.ball_calibration_enabled is True
    assert status.verification_enabled is True
