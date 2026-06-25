from __future__ import annotations

import cv2
import numpy as np
import pytest

from bas.calibration.charuco import CharucoBoardSpec, detect_charuco_corners, render_charuco_board
from bas.calibration.linked import (
    build_linked_patterns,
    match_linked_pattern_observation,
    projection_output_summary,
    solve_linked_projection_calibration,
)
from bas.geometry import TableGeometry


def _charuco_supported() -> bool:
    aruco = getattr(cv2, "aruco", None)
    return bool(aruco is not None and (hasattr(aruco, "CharucoDetector") or hasattr(aruco, "interpolateCornersCharuco")))


def _sample_geometry() -> TableGeometry:
    return TableGeometry(
        outer_norm=np.array([[0.02, 0.03], [0.98, 0.03], [0.98, 0.97], [0.02, 0.97]], dtype=np.float32),
        inner_norm=np.array(
            [
                [0.10, 0.10],
                [0.45, 0.10],
                [0.50, 0.08],
                [0.55, 0.10],
                [0.90, 0.10],
                [0.92, 0.45],
                [0.90, 0.90],
                [0.55, 0.90],
                [0.50, 0.92],
                [0.45, 0.90],
                [0.10, 0.90],
                [0.08, 0.45],
            ],
            dtype=np.float32,
        ),
        pockets_norm=[
            np.array([[0.08, 0.14], [0.10, 0.12], [0.12, 0.10]], dtype=np.float32),
            np.array([[0.46, 0.08], [0.50, 0.06], [0.54, 0.08]], dtype=np.float32),
            np.array([[0.88, 0.10], [0.90, 0.12], [0.92, 0.14]], dtype=np.float32),
            np.array([[0.92, 0.86], [0.90, 0.88], [0.88, 0.90]], dtype=np.float32),
            np.array([[0.54, 0.92], [0.50, 0.94], [0.46, 0.92]], dtype=np.float32),
            np.array([[0.12, 0.90], [0.10, 0.88], [0.08, 0.86]], dtype=np.float32),
        ],
    )


def test_build_linked_patterns_covers_full_and_focus_zones() -> None:
    patterns = build_linked_patterns(_sample_geometry(), (1280, 800))

    assert len(patterns) == 10
    assert patterns[0].collect_for_solver is False
    assert patterns[-1].collect_for_solver is False
    assert sum(1 for pattern in patterns if pattern.collect_for_solver) == 8
    assert any(pattern.emphasis_zone == "full" for pattern in patterns)
    assert any(pattern.emphasis_zone == "center" for pattern in patterns)
    assert any(pattern.emphasis_zone.startswith("pocket_") for pattern in patterns)


@pytest.mark.skipif(not _charuco_supported(), reason="OpenCV ChArUco detection support is unavailable.")
def test_render_charuco_board_handles_problematic_roi_dimensions() -> None:
    spec = CharucoBoardSpec(squares_x=8, squares_y=6, square_length_m=0.030, marker_length_m=0.022)
    image = render_charuco_board(spec, 384, 209)
    points, ids = detect_charuco_corners(image, spec)

    assert image.shape == (209, 384, 3)
    assert points.shape[0] >= 4
    assert ids.size == points.shape[0]


@pytest.mark.skipif(not _charuco_supported(), reason="OpenCV ChArUco detection support is unavailable.")
def test_linked_calibration_can_recover_projector_mapping_from_warped_patterns() -> None:
    projector_size = (1280, 800)
    patterns = [pattern for pattern in build_linked_patterns(_sample_geometry(), projector_size) if pattern.collect_for_solver][:3]
    camera_size = (1600, 1000)
    projector_quad = np.array([[0, 0], [1279, 0], [1279, 799], [0, 799]], dtype=np.float32)
    camera_quad = np.array([[140, 120], [1450, 80], [1500, 930], [100, 950]], dtype=np.float32)
    H_proj_to_cam = cv2.getPerspectiveTransform(projector_quad, camera_quad)

    observations = []
    for pattern in patterns:
        warped = cv2.warpPerspective(pattern.image, H_proj_to_cam, camera_size, flags=cv2.INTER_LINEAR)
        observation = match_linked_pattern_observation(pattern, warped)
        assert observation is not None
        observations.append(observation)

    table_polygon_cam = cv2.perspectiveTransform(projector_quad.reshape((-1, 1, 2)), H_proj_to_cam).reshape((-1, 2))
    result = solve_linked_projection_calibration(observations, projector_size, table_polygon_cam=table_polygon_cam)
    summary = projection_output_summary(result)

    assert "匹配角点" in summary
    assert result.projection.is_valid
    assert result.projection.table_polygon_proj.shape[0] == 4
    pred = result.projection.camera_to_projector_points(observations[0].camera_points)
    assert np.max(np.linalg.norm(pred - observations[0].projector_points, axis=1)) < 2.0
