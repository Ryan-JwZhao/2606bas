from __future__ import annotations

import numpy as np

from bas.config import AppConfig
from bas.geometry_contract import projection_calibration_context
from bas.projection.frame_transform import ProjectionFrameTransform
from bas.user_settings import UserSettings


def _numbered_frame() -> np.ndarray:
    return np.array(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
        ],
        dtype=np.uint8,
    )


def test_calibration_and_output_rotations_are_independent() -> None:
    frame = _numbered_frame()
    transform = ProjectionFrameTransform(
        calibration_rotation_degrees=0,
        output_rotation_degrees=180,
    )

    calibration_frame = transform.apply(frame, calibration_mode=True)
    output_frame = transform.apply(frame, calibration_mode=False)

    np.testing.assert_array_equal(calibration_frame, frame)
    np.testing.assert_array_equal(output_frame, np.rot90(frame, 2))


def test_projection_rotations_round_trip_through_user_settings() -> None:
    config = AppConfig()
    config.camera.frame_rotation_degrees = 0
    config.projection.calibration_rotation_degrees = 90
    config.projection.output_rotation_degrees = 180

    stored = UserSettings.from_config(config)
    restored = AppConfig()
    stored.apply_to_config(restored)

    assert restored.camera.frame_rotation_degrees == 0
    assert restored.projection.calibration_rotation_degrees == 90
    assert restored.projection.output_rotation_degrees == 180


def test_projection_calibration_context_binds_only_the_calibration_rotation() -> None:
    context = projection_calibration_context(
        frame_width=1920,
        frame_height=1080,
        frame_rotation_degrees=0,
        camera_coordinate_domain="raw",
        distortion_file=None,
        projector_width=1280,
        projector_height=800,
        projection_calibration_rotation_degrees=90,
    )

    assert context["projection_calibration_rotation_degrees"] == 90
    assert "projection_output_rotation_degrees" not in context
