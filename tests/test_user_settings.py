from __future__ import annotations

import json

from bas.config import AppConfig
from bas.user_settings import UserSettings


def test_user_settings_can_clear_default_paths_and_save_false_values(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(
        json.dumps(
            {
                "nori_sdk_root": None,
                "model_path": None,
                "class_file_path": None,
                "distortion_correction_enabled": False,
                "distortion_correction_file": None,
                "exposure_auto": True,
                "white_balance_auto": False,
                "white_balance_value": 5200,
                "learning_ranker_model_path": None,
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig()
    cfg.camera.nori_sdk_root = "C:/sdk"
    cfg.detector.model_path = "C:/model.pt"
    cfg.detector.class_file_path = "C:/classes.txt"
    cfg.camera.distortion_correction_enabled = True
    cfg.camera.distortion_correction_file = "C:/intrinsics.yaml"
    cfg.camera.exposure_auto = False
    cfg.camera.white_balance_auto = True
    cfg.camera.white_balance_value = 4600
    cfg.learning.ranker_model_path = "C:/ranker.json"

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.camera.nori_sdk_root is None
    assert cfg.detector.model_path is None
    assert cfg.detector.class_file_path is None
    assert cfg.camera.distortion_correction_enabled is False
    assert cfg.camera.distortion_correction_file is None
    assert cfg.camera.exposure_auto is True
    assert cfg.camera.white_balance_auto is False
    assert cfg.camera.white_balance_value == 5200
    assert cfg.learning.ranker_model_path is None


def test_user_settings_save_does_not_persist_internal_loaded_key_state(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    settings = UserSettings(exposure_auto=True)
    settings._provided_keys.add("exposure_auto")
    settings._loaded_from_file = True

    settings.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert "_provided_keys" not in saved
    assert "_loaded_from_file" not in saved
    assert saved["exposure_auto"] is True


def test_user_settings_applies_white_balance_value_with_clamp(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(
        json.dumps(
            {
                "white_balance_auto": False,
                "white_balance_value": 999999,
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig()
    cfg.camera.white_balance_auto = True
    cfg.camera.white_balance_value = 4600

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.camera.white_balance_auto is False
    assert cfg.camera.white_balance_value == 15000


def test_missing_user_settings_does_not_clear_default_config_paths(tmp_path) -> None:
    cfg = AppConfig()
    cfg.detector.model_path = "C:/model.pt"
    cfg.detector.class_file_path = "C:/classes.txt"
    cfg.camera.distortion_correction_file = "C:/intrinsics.yaml"

    UserSettings.load(tmp_path / "missing.json").apply_to_config(cfg)

    assert cfg.detector.model_path == "C:/model.pt"
    assert cfg.detector.class_file_path == "C:/classes.txt"
    assert cfg.camera.distortion_correction_file == "C:/intrinsics.yaml"


def test_user_settings_applies_boundary_inset_configuration(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(
        json.dumps(
            {
                "projection_visible_inset_bottom_mm": 12.0,
                "physical_rail_inset_top_mm": 10.0,
                "physical_middle_pocket_relief_top_mm": 6.0,
                "center_reachable_extra_margin_mm": 3.0,
                "ball_center_compensation_enabled": True,
                "ball_center_compensation_auto_reference": True,
                "ball_center_compensation_scale_x_pct": 2.5,
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig()

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.calibration.projection_visible_inset_bottom_mm == 12.0
    assert cfg.calibration.physical_rail_inset_top_mm == 10.0
    assert cfg.calibration.physical_middle_pocket_relief_top_mm == 6.0
    assert cfg.calibration.center_reachable_extra_margin_mm == 3.0
    assert cfg.calibration.ball_center_compensation_enabled is True
    assert cfg.calibration.ball_center_compensation_auto_reference is True
    assert cfg.calibration.ball_center_compensation_scale_x_pct == 2.5


def test_user_settings_legacy_manual_ball_reference_disables_auto_reference(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(
        json.dumps(
            {
                "ball_center_compensation_ref_x_px": 120.0,
                "ball_center_compensation_ref_y_px": -240.0,
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig()

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.calibration.ball_center_compensation_auto_reference is False
    assert cfg.calibration.ball_center_compensation_ref_x_px == 120.0
    assert cfg.calibration.ball_center_compensation_ref_y_px == -240.0
