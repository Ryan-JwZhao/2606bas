from __future__ import annotations

import json

from bas.config import AppConfig
from bas.user_settings import UserSettings


def test_web_control_settings_default_and_roundtrip(tmp_path) -> None:
    config = AppConfig()
    assert config.web_control.host == "0.0.0.0"
    assert config.web_control.port == 17070

    path = tmp_path / "user_settings.json"
    UserSettings(web_control_host="127.0.0.1", web_control_port=18080).save(path)
    loaded = UserSettings.load(path).apply_to_config(config)

    assert loaded.web_control.host == "127.0.0.1"
    assert loaded.web_control.port == 18080


def test_exposure_control_mode_roundtrips_through_user_settings(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    cfg = AppConfig()
    cfg.camera.exposure_control = "uvc"

    UserSettings.from_config(cfg).save(path)
    restored = UserSettings.load(path).apply_to_config(AppConfig())

    assert restored.camera.exposure_control == "uvc"
    assert json.loads(path.read_text(encoding="utf-8"))["exposure_control"] == "uvc"


def test_invalid_exposure_control_mode_falls_back_to_auto(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps({"exposure_control": "unknown"}), encoding="utf-8")

    restored = UserSettings.load(path).apply_to_config(AppConfig())

    assert restored.camera.exposure_control == "auto"


def test_camera_frame_rotation_roundtrips_through_user_settings(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    cfg = AppConfig()
    cfg.camera.frame_rotation_degrees = 180

    UserSettings.from_config(cfg).save(path)
    restored = UserSettings.load(path).apply_to_config(AppConfig())

    assert restored.camera.frame_rotation_degrees == 180
    assert json.loads(path.read_text(encoding="utf-8"))["frame_rotation_degrees"] == 180


def test_video_rotation_roundtrips_through_user_settings(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    cfg = AppConfig()
    cfg.camera.video_rotation_degrees = 180

    UserSettings.from_config(cfg).save(path)
    restored = UserSettings.load(path).apply_to_config(AppConfig())

    assert restored.camera.video_rotation_degrees == 180
    assert json.loads(path.read_text(encoding="utf-8"))["video_rotation_degrees"] == 180


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


def test_user_settings_applies_single_projection_and_ball_compensation_paths(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(
        json.dumps(
            {
                "projection_calibration_file": "C:/projection.json",
                "engineered_ball_compensation_file": "C:/engineered_ball.json",
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig()

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.calibration.engineered_ball_compensation_file == "C:/engineered_ball.json"
    assert cfg.calibration.projection_file == "C:/projection.json"


def test_user_settings_applies_route_freeze_parameters(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(
        json.dumps(
            {
                "planner_route_freeze_enabled": True,
                "planner_route_freeze_enter_frames": 3,
                "planner_route_freeze_release_frames": 9,
                "planner_route_freeze_same_route_refresh_mm": 14.0,
                "planner_route_freeze_same_route_refresh_score_delta": 0.11,
                "planner_route_freeze_switch_confirm_frames": 4,
                "planner_route_freeze_switch_min_distance_mm": 36.0,
                "planner_route_freeze_switch_min_score_delta": 0.27,
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig()

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.planner.route_freeze_enabled is True
    assert cfg.planner.route_freeze_enter_frames == 3
    assert cfg.planner.route_freeze_release_frames == 9
    assert cfg.planner.route_freeze_same_route_refresh_mm == 14.0
    assert cfg.planner.route_freeze_same_route_refresh_score_delta == 0.11
    assert cfg.planner.route_freeze_switch_confirm_frames == 4
    assert cfg.planner.route_freeze_switch_min_distance_mm == 36.0
    assert cfg.planner.route_freeze_switch_min_score_delta == 0.27


def test_user_settings_round_trips_continuous_route_stability_parameters(tmp_path) -> None:
    cfg = AppConfig()
    cfg.planner.route_stability_enabled = True
    cfg.planner.route_stability_stationary_tau_ms = 2400.0
    cfg.planner.route_stability_motion_tau_ms = 55.0
    cfg.planner.route_stability_quiet_speed_mm_s = 11.0
    cfg.planner.route_stability_deadband_mm = 2.25
    cfg.planner.route_topology_switch_confirm_ms = 180.0
    cfg.planner.route_topology_switch_score_delta = 0.21
    path = tmp_path / "user_settings.json"

    UserSettings.from_config(cfg).save(path)
    restored = AppConfig()
    UserSettings.load(path).apply_to_config(restored)

    assert restored.planner.route_stability_enabled is True
    assert restored.planner.route_stability_stationary_tau_ms == 2400.0
    assert restored.planner.route_stability_motion_tau_ms == 55.0
    assert restored.planner.route_stability_quiet_speed_mm_s == 11.0
    assert restored.planner.route_stability_deadband_mm == 2.25
    assert restored.planner.route_topology_switch_confirm_ms == 180.0
    assert restored.planner.route_topology_switch_score_delta == 0.21


def test_user_settings_round_trips_pocket_corridor_and_rail_assist(tmp_path) -> None:
    cfg = AppConfig()
    cfg.planner.pocket_entry_safety_margin_mm = 2.5
    cfg.planner.pocket_entry_max_angle_deg = 52.0
    cfg.planner.rail_assist_enabled = True
    cfg.planner.rail_assist_max_center_distance_mm = 66.0
    cfg.planner.rail_assist_max_alignment_angle_deg = 16.0
    cfg.planner.rail_assist_max_deflection_deg = 58.0
    path = tmp_path / "user_settings.json"

    UserSettings.from_config(cfg).save(path)
    restored = UserSettings.load(path).apply_to_config(AppConfig())

    assert restored.planner.pocket_entry_safety_margin_mm == 2.5
    assert restored.planner.pocket_entry_max_angle_deg == 52.0
    assert restored.planner.rail_assist_enabled is True
    assert restored.planner.rail_assist_max_center_distance_mm == 66.0
    assert restored.planner.rail_assist_max_alignment_angle_deg == 16.0
    assert restored.planner.rail_assist_max_deflection_deg == 58.0


def test_user_settings_applies_cue_sector_correction_parameters(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(
        json.dumps(
            {
                "planner_cue_sector_correction_enabled": False,
                "planner_cue_sector_angle_deg": 18.0,
                "planner_cue_sector_edge_margin_deg": 1.5,
                "planner_cue_sector_corridor_width_px": 180.0,
                "planner_cue_sector_switch_confirm_frames": 5,
                "planner_cue_sector_switch_min_score_delta": 0.33,
                "planner_cue_sector_min_stick_quality": 0.4,
                "planner_cue_sector_require_balls_stationary": False,
                "planner_cue_sector_stationary_speed_mm_s": 11.0,
                "planner_cue_sector_stationary_speed_px_s": 33.0,
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig()

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.planner.cue_sector_correction_enabled is False
    assert cfg.planner.cue_sector_angle_deg == 18.0
    assert cfg.planner.cue_sector_edge_margin_deg == 1.5
    assert cfg.planner.cue_sector_corridor_width_px == 180.0
    assert cfg.planner.cue_sector_switch_confirm_frames == 5
    assert cfg.planner.cue_sector_switch_min_score_delta == 0.33
    assert cfg.planner.cue_sector_min_stick_quality == 0.4
    assert cfg.planner.cue_sector_require_balls_stationary is False
    assert cfg.planner.cue_sector_stationary_speed_mm_s == 11.0
    assert cfg.planner.cue_sector_stationary_speed_px_s == 33.0


def test_user_settings_applies_target_lock_parameters(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(
        json.dumps(
            {
                "planner_target_lock_enabled": False,
                "planner_target_lock_confirm_frames": 5,
                "planner_target_lock_switch_confirm_frames": 12,
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig()

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.planner.target_lock_enabled is False
    assert cfg.planner.target_lock_confirm_frames == 5
    assert cfg.planner.target_lock_switch_confirm_frames == 12


def test_user_settings_exports_target_lock_parameters() -> None:
    cfg = AppConfig()
    cfg.planner.target_lock_enabled = False
    cfg.planner.target_lock_confirm_frames = 4
    cfg.planner.target_lock_switch_confirm_frames = 10

    settings = UserSettings.from_config(cfg)

    assert settings.planner_target_lock_enabled is False
    assert settings.planner_target_lock_confirm_frames == 4
    assert settings.planner_target_lock_switch_confirm_frames == 10


def test_user_settings_migrates_legacy_target_shot_frames_to_hold_ms(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(
        json.dumps(
            {
                "planner_target_shot_enabled": False,
                "detect_fps_limit_hz": 15.0,
                "planner_target_shot_trigger_frames": 17,
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig()

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.planner.target_shot_enabled is False
    assert cfg.detector.detect_fps_limit_hz == 15.0
    assert cfg.planner.target_shot_trigger_frames is None
    assert cfg.planner.target_shot_activate_hold_ms == 1133


def test_user_settings_applies_target_shot_time_parameters(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(
        json.dumps(
            {
                "planner_target_shot_enabled": False,
                "planner_target_shot_activate_hold_ms": 900,
                "planner_target_shot_switch_hold_ms": 1600,
                "planner_target_shot_miss_grace_ms": 250,
                "planner_target_shot_release_confirm_ms": 350,
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig()

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.planner.target_shot_enabled is False
    assert cfg.planner.target_shot_activate_hold_ms == 900
    assert cfg.planner.target_shot_switch_hold_ms == 1600
    assert cfg.planner.target_shot_miss_grace_ms == 250
    assert cfg.planner.target_shot_release_confirm_ms == 350


def test_user_settings_exports_target_shot_time_parameters() -> None:
    cfg = AppConfig()
    cfg.planner.target_shot_enabled = False
    cfg.planner.target_shot_trigger_frames = None
    cfg.planner.target_shot_activate_hold_ms = 900
    cfg.planner.target_shot_switch_hold_ms = 1600
    cfg.planner.target_shot_miss_grace_ms = 250
    cfg.planner.target_shot_release_confirm_ms = 350

    settings = UserSettings.from_config(cfg)

    assert settings.planner_target_shot_enabled is False
    assert settings.planner_target_shot_trigger_frames is None
    assert settings.planner_target_shot_activate_hold_ms == 900
    assert settings.planner_target_shot_switch_hold_ms == 1600
    assert settings.planner_target_shot_miss_grace_ms == 250
    assert settings.planner_target_shot_release_confirm_ms == 350


def test_user_settings_applies_projection_interaction_toggles(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(
        json.dumps(
            {
                "projection_auto_pocket_animation_enabled": False,
                "projection_auto_victory_animation_enabled": False,
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig()

    UserSettings.load(path).apply_to_config(cfg)

    assert cfg.projection.auto_pocket_animation_enabled is False
    assert cfg.projection.auto_victory_animation_enabled is False


def test_user_settings_exports_projection_interaction_toggles() -> None:
    cfg = AppConfig()
    cfg.projection.auto_pocket_animation_enabled = False
    cfg.projection.auto_victory_animation_enabled = False

    settings = UserSettings.from_config(cfg)

    assert settings.projection_auto_pocket_animation_enabled is False
    assert settings.projection_auto_victory_animation_enabled is False


def test_user_settings_round_trips_training_projection_prompt_controls(tmp_path) -> None:
    path = tmp_path / "user_settings.json"
    path.write_text(
        json.dumps(
            {
                "projection_training_prompt_enabled": False,
                "projection_training_prompt_x_pct": 27.5,
                "projection_training_prompt_y_pct": 82.0,
                "projection_training_prompt_font_size_px": 52,
            }
        ),
        encoding="utf-8",
    )
    cfg = UserSettings.load(path).apply_to_config(AppConfig())

    assert cfg.projection.training_prompt_enabled is False
    assert cfg.projection.training_prompt_x_pct == 27.5
    assert cfg.projection.training_prompt_y_pct == 82.0
    assert cfg.projection.training_prompt_font_size_px == 52

    exported = UserSettings.from_config(cfg)
    assert exported.projection_training_prompt_enabled is False
    assert exported.projection_training_prompt_x_pct == 27.5
    assert exported.projection_training_prompt_y_pct == 82.0
    assert exported.projection_training_prompt_font_size_px == 52
