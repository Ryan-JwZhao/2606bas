import json

from bas.config import AppConfig
from bas.geometry_contract import projection_calibration_context
from bas.user_settings import UserSettings


def test_projector_has_no_independent_discrete_rotation_settings() -> None:
    config = AppConfig()

    assert not hasattr(config.projection, "calibration_rotation_degrees")
    assert not hasattr(config.projection, "output_rotation_degrees")


def test_legacy_projector_rotation_settings_are_ignored(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "frame_rotation_degrees": 180,
                "projection_calibration_rotation_degrees": 90,
                "projection_output_rotation_degrees": 180,
            }
        ),
        encoding="utf-8",
    )
    restored = UserSettings.load(path).apply_to_config(AppConfig())

    assert restored.camera.frame_rotation_degrees == 180
    assert not hasattr(restored.projection, "calibration_rotation_degrees")
    assert not hasattr(restored.projection, "output_rotation_degrees")


def test_projection_calibration_context_uses_one_projector_coordinate_domain() -> None:
    context = projection_calibration_context(
        frame_width=1920,
        frame_height=1080,
        frame_rotation_degrees=0,
        camera_coordinate_domain="raw",
        distortion_file=None,
        projector_width=1280,
        projector_height=800,
    )

    assert "projection_calibration_rotation_degrees" not in context
    assert "projection_output_rotation_degrees" not in context
