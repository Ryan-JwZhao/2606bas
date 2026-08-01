from __future__ import annotations

import cv2
import numpy as np
import pytest

from bas.calibration.charuco import CharucoBoardSpec, detect_charuco_corners, render_charuco_board
from bas.calibration.linked import (
    LinkedCalibrationObservation,
    build_linked_patterns,
    collect_linked_pattern_observation,
    match_linked_pattern_observation,
    projection_output_summary,
    solve_linked_projection_calibration,
)
from bas.calibration.projector import ProjectionCalibration
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


def test_linked_focus_patterns_keep_full_board_size_at_projector_edges() -> None:
    patterns = build_linked_patterns(_sample_geometry(), (1280, 800))
    pocket_rois = [pattern.roi_proj for pattern in patterns if pattern.emphasis_zone.startswith("pocket_")]
    widths = [x2 - x1 for x1, _y1, x2, _y2 in pocket_rois]
    heights = [y2 - y1 for _x1, y1, _x2, y2 in pocket_rois]

    assert max(widths) - min(widths) <= 1
    assert max(heights) - min(heights) <= 1
    assert min(x1 for x1, _y1, _x2, _y2 in pocket_rois) >= 64
    assert min(y1 for _x1, y1, _x2, _y2 in pocket_rois) >= 64
    assert max(x2 for _x1, _y1, x2, _y2 in pocket_rois) <= 1216
    assert max(y2 for _x1, _y1, _x2, y2 in pocket_rois) <= 736


@pytest.mark.skipif(not _charuco_supported(), reason="OpenCV ChArUco detection support is unavailable.")
def test_all_linked_patterns_remain_detectable_after_camera_blur() -> None:
    patterns = [
        pattern
        for pattern in build_linked_patterns(_sample_geometry(), (1280, 800))
        if pattern.collect_for_solver
    ]

    matched_counts = []
    for pattern in patterns:
        blurred = cv2.GaussianBlur(pattern.image, (5, 5), 1.5)
        observation = match_linked_pattern_observation(pattern, blurred)
        matched_counts.append(0 if observation is None else observation.matched_count)

    assert len(matched_counts) == 8
    assert min(matched_counts) >= 6


@pytest.mark.skipif(not _charuco_supported(), reason="OpenCV ChArUco detection support is unavailable.")
def test_render_charuco_board_handles_problematic_roi_dimensions() -> None:
    spec = CharucoBoardSpec(squares_x=8, squares_y=6, square_length_m=0.030, marker_length_m=0.022)
    image = render_charuco_board(spec, 384, 209)
    points, ids = detect_charuco_corners(image, spec)

    assert image.shape == (209, 384, 3)
    assert points.shape[0] >= 4
    assert ids.size == points.shape[0]


@pytest.mark.skipif(not _charuco_supported(), reason="OpenCV ChArUco detection support is unavailable.")
def test_linked_pattern_collection_survives_delayed_camera_frames() -> None:
    pattern = next(
        pattern
        for pattern in build_linked_patterns(_sample_geometry(), (1280, 800))
        if pattern.emphasis_zone == "full"
    )
    stale = np.zeros_like(pattern.image)
    frames = iter([stale] * 8 + [pattern.image] * 4)

    capture = collect_linked_pattern_observation(
        pattern,
        lambda: next(frames, None),
        transition_frames=8,
        max_detection_frames=4,
    )

    assert capture.observation is not None
    assert capture.observation.matched_count >= 6
    assert capture.transition_frames_read == 8


@pytest.mark.skipif(not _charuco_supported(), reason="OpenCV ChArUco detection support is unavailable.")
def test_linked_calibration_can_recover_projector_mapping_from_warped_patterns(tmp_path) -> None:
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
    result = solve_linked_projection_calibration(
        observations,
        projector_size,
        table_polygon_cam=table_polygon_cam,
        minimum_pocket_zones=0,
    )
    summary = projection_output_summary(result)

    assert "匹配角点" in summary
    assert result.projection.is_valid
    assert result.projection.table_polygon_proj.shape[0] == 4
    assert result.projection.table_control_points_norm.shape[0] >= 12
    assert result.projection.table_control_points_proj.shape == result.projection.table_control_points_norm.shape
    assert result.summary["geometry_model"] == "independent_2d"
    assert result.summary["camera_extrinsics_used"] is False
    saved_path = tmp_path / "independent_projection.json"
    result.projection.save(saved_path)
    restored = ProjectionCalibration.load_json(saved_path)
    np.testing.assert_allclose(restored.table_control_points_norm, result.projection.table_control_points_norm)
    np.testing.assert_allclose(restored.table_control_points_proj, result.projection.table_control_points_proj)
    pred = result.projection.camera_to_projector_points(observations[0].camera_points)
    assert np.max(np.linalg.norm(pred - observations[0].projector_points, axis=1)) < 2.0


def test_linked_calibration_rejects_excessive_ransac_outliers() -> None:
    camera = np.asarray([[x, y] for y in (100.0, 300.0, 500.0) for x in (100.0, 300.0, 500.0, 700.0)], dtype=np.float32)
    projector = camera * np.asarray([1.2, 1.1], dtype=np.float32) + np.asarray([20.0, 30.0], dtype=np.float32)
    corrupted = projector.copy()
    corrupted[::2] += np.asarray([400.0, -300.0], dtype=np.float32)

    observations = [
        LinkedCalibrationObservation("clean", "clean", "center", camera, projector, np.arange(12), 12, 12),
        LinkedCalibrationObservation("bad", "bad", "pocket_lt", camera, corrupted, np.arange(12), 12, 12),
    ]

    with pytest.raises(RuntimeError, match="inlier"):
        solve_linked_projection_calibration(
            observations,
            (1280, 800),
            table_polygon_cam=np.asarray([[100, 100], [700, 100], [700, 500], [100, 500]], dtype=np.float32),
        )


def test_linked_calibration_requires_table_polygon_anchor() -> None:
    camera = np.asarray(
        [[x, y] for y in (100.0, 300.0, 500.0) for x in (100.0, 300.0, 500.0, 700.0)],
        dtype=np.float32,
    )
    projector = camera * np.asarray([1.2, 1.1], dtype=np.float32) + np.asarray([20.0, 30.0], dtype=np.float32)
    observations = [
        LinkedCalibrationObservation("full", "full", "full", camera, projector, np.arange(12), 12, 12),
        LinkedCalibrationObservation("center", "center", "center", camera, projector, np.arange(12), 12, 12),
    ]

    with pytest.raises(ValueError, match="table polygon"):
        solve_linked_projection_calibration(
            observations,
            (1280, 800),
            table_polygon_cam=None,
            minimum_pocket_zones=0,
        )
