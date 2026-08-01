from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from bas.calibration.projector import ProjectionCalibration
from bas.config import AppConfig
from bas.geometry_contract import projection_calibration_context
from bas import runtime_env

runtime_env.preload_torch_for_backend = lambda _backend: None

from bas.ui.calibration_workbench import CalibrationWorkbenchDialog, assess_calibration_workbench
from bas.ui.complete_calibration_wizard import run_complete_calibration_steps


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
        run_complete_calibration_wizard=lambda: calls.append("complete"),
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
    assert dialog.complete_btn.isEnabled() is True
    assert dialog.ball_btn.isEnabled() is False
    assert dialog.verify_btn.isEnabled() is False
    assert "180" not in dialog.projection_state.text()

    dialog.complete_btn.click()
    assert calls == ["complete"]

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


def test_workbench_reuses_last_actual_capture_size_after_pipeline_stops(tmp_path) -> None:
    app = _app()
    config = AppConfig()
    config.camera.width = 1920
    config.camera.height = 1080
    config.camera.distortion_correction_enabled = False
    config.detector.backend = "debug_color"
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float64),
        np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float64),
        projector_size=(1280, 800),
    )
    projection.calibration_context = projection_calibration_context(
        frame_width=1280,
        frame_height=720,
        frame_rotation_degrees=0,
        camera_coordinate_domain="raw",
        distortion_file=None,
        projector_width=1280,
        projector_height=800,
    )
    path = tmp_path / "projection.json"
    projection.save(path)
    config.calibration.projection_file = str(path)
    operator = SimpleNamespace(
        config=config,
        pipeline=None,
        calibration_actual_frame_size=lambda: (1280, 720),
        start_pipeline=lambda: None,
        stop_pipeline=lambda: None,
        open_settings=lambda: None,
        run_linked_projector_calibration=lambda: None,
        run_engineered_ball_compensation_wizard=lambda: None,
        run_complete_calibration_wizard=lambda: None,
        show_projector_calibration_result=lambda _calibration: None,
        show_projector_residual_overlay=lambda _calibration: None,
    )
    parent = QtWidgets.QWidget()
    dialog = CalibrationWorkbenchDialog(operator, parent)

    assert dialog.status is not None
    assert dialog.status.projection_valid is True
    assert dialog.status.ball_calibration_enabled is True

    dialog.close()
    parent.close()
    app.processEvents()


def test_complete_calibration_runs_both_phases_in_order() -> None:
    calls: list[str] = []

    result = run_complete_calibration_steps(
        lambda: calls.append("projection") or True,
        lambda: calls.append("prepare_ball") or True,
        lambda: calls.append("ball") or True,
    )

    assert result.completed is True
    assert calls == ["projection", "prepare_ball", "ball"]


def test_complete_calibration_stops_when_projection_fails() -> None:
    calls: list[str] = []

    result = run_complete_calibration_steps(
        lambda: calls.append("projection") or False,
        lambda: calls.append("prepare_ball") or True,
        lambda: calls.append("ball") or True,
    )

    assert result.stage == "projection_failed"
    assert calls == ["projection"]


def test_complete_calibration_can_pause_after_projection_for_table_setup() -> None:
    calls: list[str] = []

    result = run_complete_calibration_steps(
        lambda: calls.append("projection") or True,
        lambda: calls.append("prepare_ball") or False,
        lambda: calls.append("ball") or True,
    )

    assert result.stage == "ball_setup_cancelled"
    assert result.projection_completed is True
    assert calls == ["projection", "prepare_ball"]
