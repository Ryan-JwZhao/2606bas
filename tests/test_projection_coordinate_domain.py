import json

import numpy as np

from bas.config import AppConfig
from bas.geometry_contract import projection_calibration_context
from bas.projection.frame_transform import ProjectionFrameTransform
from bas.user_settings import UserSettings


def test_camera_rotation_supersedes_legacy_projector_rotation_settings(tmp_path) -> None:
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
    assert restored.projection.legacy_calibration_rotation_degrees == 0
    assert restored.projection.legacy_output_rotation_degrees == 0


def test_legacy_projector_output_rotation_is_used_until_camera_migration(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "projection_calibration_rotation_degrees": 0,
                "projection_output_rotation_degrees": 180,
            }
        ),
        encoding="utf-8",
    )

    restored = UserSettings.load(path).apply_to_config(AppConfig())

    assert restored.camera.frame_rotation_degrees == 0
    assert restored.projection.legacy_calibration_rotation_degrees == 0
    assert restored.projection.legacy_output_rotation_degrees == 180
    persisted = UserSettings.from_config(restored)
    assert persisted.projection_output_rotation_degrees == 180


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


def test_legacy_rotation_bridge_only_rotates_runtime_output() -> None:
    image = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))
    transform = ProjectionFrameTransform(
        calibration_rotation_degrees=0,
        output_rotation_degrees=180,
    )

    assert np.array_equal(transform.apply(image, calibration_mode=True), image)
    assert np.array_equal(
        transform.apply(image, calibration_mode=False),
        np.rot90(image, 2),
    )
