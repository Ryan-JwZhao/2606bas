from __future__ import annotations

import cv2
import numpy as np
import pytest

from bas.calibration.charuco import CharucoBoardSpec, detect_charuco_corners, render_charuco_board
from bas.calibration.linked import (
    LinkedCalibrationObservation,
    build_linked_patterns,
    collect_linked_pattern_observation,
    linked_calibration_runtime_summary,
    match_linked_pattern_observation,
    linked_table_surface_polygon,
    projection_output_summary,
    solve_linked_projection_calibration,
)
from bas.calibration.projector import ProjectionCalibration, polygon_quad
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

    assert len(patterns) == 30
    assert patterns[0].collect_for_solver is False
    assert patterns[-1].collect_for_solver is False
    assert sum(1 for pattern in patterns if pattern.collect_for_solver) == 28
    assert sum(1 for pattern in patterns if pattern.pattern_id.startswith("coverage_")) == 20
    assert any(pattern.emphasis_zone == "full" for pattern in patterns)
    assert any(pattern.emphasis_zone == "center" for pattern in patterns)
    assert any(pattern.emphasis_zone.startswith("pocket_") for pattern in patterns)


def test_linked_coverage_anchor_uses_coplanar_inner_table_surface() -> None:
    geometry = _sample_geometry()
    surface = linked_table_surface_polygon(geometry, (1920, 1080))
    expected_inner = polygon_quad(geometry.inner_norm * np.asarray([1920.0, 1080.0], dtype=np.float32))

    assert surface.shape == (4, 2)
    assert np.allclose(surface, expected_inner, atol=1e-3)


def test_linked_calibration_runtime_summary_identifies_loaded_implementation() -> None:
    summary = linked_calibration_runtime_summary()

    assert "linked-geometry-v11" in summary
    assert "coverage=85%/85%/72%" in summary
    assert "core_grid=5x4:100%" in summary
    assert "edges=100%" in summary
    assert "dense_tiles=5x4" in summary
    assert "middle_pockets=edge_v2" in summary
    assert "corner_pockets=edge_v2" in summary
    assert "capture_retry=2" in summary
    assert "pattern_min=4" in summary
    assert "total_min=80" in summary
    assert "edge_contrast=raw_plus_clahe_v2" in summary
    assert "bas" in summary and "linked.py" in summary


def test_linked_focus_patterns_keep_full_board_size_at_projector_edges() -> None:
    patterns = build_linked_patterns(_sample_geometry(), (1280, 800))
    pocket_rois = [pattern.roi_proj for pattern in patterns if pattern.emphasis_zone.startswith("pocket_")]
    widths = [x2 - x1 for x1, _y1, x2, _y2 in pocket_rois]
    heights = [y2 - y1 for _x1, y1, _x2, y2 in pocket_rois]

    assert max(widths) - min(widths) <= 1
    assert max(heights) - min(heights) <= 1
    assert min(x1 for x1, _y1, _x2, _y2 in pocket_rois) >= 12
    assert min(y1 for _x1, y1, _x2, _y2 in pocket_rois) >= 8
    assert max(x2 for _x1, _y1, x2, _y2 in pocket_rois) <= 1268
    assert max(y2 for _x1, _y1, _x2, y2 in pocket_rois) <= 792


def test_middle_pocket_patterns_stay_centered_and_close_to_table_edges() -> None:
    patterns = build_linked_patterns(_sample_geometry(), (1280, 800))
    top = next(pattern for pattern in patterns if pattern.emphasis_zone == "pocket_mt")
    bottom = next(pattern for pattern in patterns if pattern.emphasis_zone == "pocket_mb")
    top_center = ((top.roi_proj[0] + top.roi_proj[2]) * 0.5, (top.roi_proj[1] + top.roi_proj[3]) * 0.5)
    bottom_center = ((bottom.roi_proj[0] + bottom.roi_proj[2]) * 0.5, (bottom.roi_proj[1] + bottom.roi_proj[3]) * 0.5)

    assert abs(top_center[0] - 640.0) <= 1.0
    assert top.roi_proj[1] <= 16
    assert abs(bottom_center[0] - 640.0) <= 1.0
    assert bottom.roi_proj[3] >= 784


def test_corner_pocket_patterns_reach_the_table_edge_band() -> None:
    patterns = build_linked_patterns(_sample_geometry(), (1280, 800))
    corners = [
        pattern
        for pattern in patterns
        if pattern.emphasis_zone in {"pocket_lt", "pocket_rt", "pocket_rb", "pocket_lb"}
    ]

    assert len(corners) == 4
    by_zone = {pattern.emphasis_zone: pattern.roi_proj for pattern in corners}
    assert by_zone["pocket_lt"][0] <= 16 and by_zone["pocket_lt"][1] <= 16
    assert by_zone["pocket_rt"][2] >= 1264 and by_zone["pocket_rt"][1] <= 16
    assert by_zone["pocket_rb"][2] >= 1264 and by_zone["pocket_rb"][3] >= 784
    assert by_zone["pocket_lb"][0] <= 16 and by_zone["pocket_lb"][3] >= 784


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

    assert len(matched_counts) == 28
    assert min(matched_counts) >= 4


def test_edge_patterns_survive_real_table_quad_clipping() -> None:
    """Regression for the 2026-08-02 audit: left column and bottom row were clipped.

    The camera sees the table up to the image boundary, while the previous
    projector calibration describes a slanted quadrilateral.  A board planned
    from only that quadrilateral's axis-aligned bbox spills beyond the visible
    table and loses the ChArUco border needed for interpolation.
    """

    projector_size = (1280, 800)
    table_quad_proj = np.asarray(
        [[29.5053, 141.1581], [1223.8042, 89.1743], [1249.3660, 771.2333], [49.0609, 816.2275]],
        dtype=np.float32,
    )
    prior = ProjectionCalibration(
        mode="linked_hybrid_charuco",
        homography=np.eye(3, dtype=np.float64),
        table_polygon_proj=table_quad_proj,
        projector_size=projector_size,
    )
    camera_size = (1920, 1080)
    camera_quad = np.asarray([[0, 0], [1919, 0], [1919, 1079], [0, 1079]], dtype=np.float32)
    projector_to_camera = cv2.getPerspectiveTransform(table_quad_proj, camera_quad)

    failures: dict[str, int] = {}
    for pattern in build_linked_patterns(_sample_geometry(), projector_size, prior_projection=prior):
        if not pattern.pattern_id.startswith("coverage_"):
            continue
        warped = cv2.warpPerspective(pattern.image, projector_to_camera, camera_size)
        observation = match_linked_pattern_observation(pattern, warped)
        count = 0 if observation is None else observation.matched_count
        if count < 4:
            failures[pattern.pattern_id] = count

    assert failures == {}


@pytest.mark.skipif(not _charuco_supported(), reason="OpenCV ChArUco detection support is unavailable.")
def test_dark_left_corner_pattern_uses_contrast_recovery() -> None:
    pattern = next(
        pattern
        for pattern in build_linked_patterns(_sample_geometry(), (1280, 800))
        if pattern.emphasis_zone == "pocket_lb"
    )
    gray = cv2.cvtColor(pattern.image, cv2.COLOR_BGR2GRAY).astype(np.float32) * 0.06
    noise = np.random.default_rng(123).normal(0.0, 1.0, gray.shape)
    dark = np.clip(gray + noise, 0, 255).astype(np.uint8)
    dark_bgr = cv2.cvtColor(dark, cv2.COLOR_GRAY2BGR)
    _raw_points, raw_ids = detect_charuco_corners(dark_bgr, pattern.board_spec)

    assert raw_ids.size < 4
    observation = match_linked_pattern_observation(pattern, dark_bgr)
    assert observation is not None
    assert observation.matched_count >= 6


def test_dark_edge_coverage_pattern_uses_contrast_recovery() -> None:
    pattern = next(
        pattern
        for pattern in build_linked_patterns(_sample_geometry(), (1280, 800))
        if pattern.emphasis_zone == "coverage_r4_c2"
    )
    gray = cv2.cvtColor(pattern.image, cv2.COLOR_BGR2GRAY).astype(np.float32) * 0.055
    dark_bgr = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    _raw_points, raw_ids = detect_charuco_corners(dark_bgr, pattern.board_spec)
    assert raw_ids.size < 4

    observation = match_linked_pattern_observation(pattern, dark_bgr)
    assert observation is not None
    assert observation.matched_count >= 4


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
def test_linked_pattern_collection_retries_a_required_pattern_after_an_empty_attempt() -> None:
    pattern = next(
        pattern
        for pattern in build_linked_patterns(_sample_geometry(), (1280, 800))
        if pattern.emphasis_zone == "full"
    )
    blank = np.zeros_like(pattern.image)
    frames = iter([blank] * (8 + 18) + [blank] * 8 + [pattern.image] * 4)

    capture = collect_linked_pattern_observation(
        pattern,
        lambda: next(frames, None),
        transition_frames=8,
        max_detection_frames=18,
        attempts=2,
    )

    assert capture.observation is not None
    assert capture.observation.matched_count >= 6
    assert capture.attempts == 2


@pytest.mark.skipif(not _charuco_supported(), reason="OpenCV ChArUco detection support is unavailable.")
def test_logged_missing_full_and_edge_patterns_recover_before_the_coverage_gate() -> None:
    projector_size = (1280, 800)
    camera_size = (1600, 1000)
    projector_quad = np.asarray([[0, 0], [1279, 0], [1279, 799], [0, 799]], dtype=np.float32)
    camera_quad = np.asarray([[140, 120], [1450, 80], [1500, 930], [100, 950]], dtype=np.float32)
    projector_to_camera = cv2.getPerspectiveTransform(projector_quad, camera_quad)
    initially_missing = {"full", "pocket_lt", "pocket_rb", "pocket_lb"}
    observations = []

    for pattern in build_linked_patterns(_sample_geometry(), projector_size):
        if not pattern.collect_for_solver:
            continue
        warped = cv2.warpPerspective(pattern.image, projector_to_camera, camera_size, flags=cv2.INTER_LINEAR)
        blank = np.zeros_like(warped)
        if pattern.emphasis_zone in initially_missing:
            frames = iter([blank] * (8 + 18 + 8) + [warped] * 18)
        else:
            frames = iter([blank] * 8 + [warped] * 18)
        capture = collect_linked_pattern_observation(
            pattern,
            lambda frames=frames: next(frames, None),
            transition_frames=8,
            max_detection_frames=18,
            attempts=2,
        )
        assert capture.observation is not None, pattern.emphasis_zone
        observations.append(capture.observation)

    table_polygon_cam = cv2.perspectiveTransform(
        projector_quad.reshape((-1, 1, 2)),
        projector_to_camera,
    ).reshape((-1, 2))
    result = solve_linked_projection_calibration(
        observations,
        projector_size,
        table_polygon_cam=table_polygon_cam,
    )

    assert result.projection.is_valid
    assert result.summary["pocket_zones_used"] == 6
    assert result.summary["spatial_coverage"]["hull_area_ratio"] >= 0.34


@pytest.mark.skipif(not _charuco_supported(), reason="OpenCV ChArUco detection support is unavailable.")
def test_linked_calibration_can_recover_projector_mapping_from_warped_patterns(tmp_path) -> None:
    projector_size = (1280, 800)
    patterns = [pattern for pattern in build_linked_patterns(_sample_geometry(), projector_size) if pattern.collect_for_solver]
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
    assert "覆盖率" in summary
    assert "跨图样CV" in summary
    assert result.projection.quality_report["pocket_zones_used"] == result.summary["pocket_zones_used"]
    assert f"袋口区域={result.summary['pocket_zones_used']}" in summary
    assert result.projection.is_valid
    assert result.projection.table_polygon_proj.shape[0] == 4
    assert result.projection.table_control_points_norm.shape[0] >= 12
    assert result.projection.table_control_points_norm.shape[0] <= 120
    assert result.projection.table_control_points_proj.shape == result.projection.table_control_points_norm.shape
    assert result.summary["geometry_model"] == "independent_2d"
    assert result.summary["camera_extrinsics_used"] is False
    saved_path = tmp_path / "independent_projection.json"
    result.projection.save(saved_path)
    restored = ProjectionCalibration.load_json(saved_path)
    assert restored.quality_report["pocket_zones_used"] == result.summary["pocket_zones_used"]
    np.testing.assert_allclose(restored.table_control_points_norm, result.projection.table_control_points_norm)
    np.testing.assert_allclose(restored.table_control_points_proj, result.projection.table_control_points_proj)
    pred = result.projection.camera_to_projector_points(observations[0].camera_points)
    assert np.max(np.linalg.norm(pred - observations[0].projector_points, axis=1)) < 2.0


def test_projection_summary_exposes_runtime_residual_support() -> None:
    result = type(
        "Result",
        (),
        {
            "projection": type(
                "Projection",
                (),
                {
                    "quality_report": {
                        "patterns_used": 8,
                        "patterns_total": 8,
                        "matched_points_total": 120,
                        "pocket_zones_used": 6,
                        "pattern_cv_p95_px": 0.52,
                        "spatial_coverage": {"width_ratio": 0.9, "height_ratio": 0.88, "hull_area_ratio": 0.72},
                    },
                    "calibration_error_stats": lambda self: {"mean_px": 0.2, "p95_px": 0.4, "max_px": 0.8},
                },
            )(),
        },
    )()

    summary = projection_output_summary(
        result,
        geometry_report={
            "projector_residual_support_grid_ratio": 0.82,
            "projector_residual_support_edge_ratio": 0.64,
        },
    )

    assert "全域=82.0%" in summary
    assert "边缘=64.0%" in summary


def test_linked_calibration_rejects_excessive_ransac_outliers() -> None:
    camera = np.asarray([[x, y] for y in (100.0, 300.0, 500.0) for x in (100.0, 300.0, 500.0, 700.0)], dtype=np.float32)
    projector = camera * np.asarray([1.2, 1.1], dtype=np.float32) + np.asarray([20.0, 30.0], dtype=np.float32)
    camera = np.tile(camera, (4, 1))
    projector = np.tile(projector, (4, 1))
    corrupted = projector.copy()
    corrupted[::2] += np.asarray([400.0, -300.0], dtype=np.float32)

    observations = [
        LinkedCalibrationObservation("clean", "clean", "center", camera, projector, np.arange(48), 48, 48),
        LinkedCalibrationObservation("bad", "bad", "pocket_lt", camera, corrupted, np.arange(48), 48, 48),
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


def test_linked_calibration_uses_actual_coverage_instead_of_named_pocket_zones() -> None:
    def observation(pattern_id: str, zone: str, camera: np.ndarray) -> LinkedCalibrationObservation:
        projector = camera * np.asarray([1.10, 1.05], dtype=np.float32) + np.asarray([20.0, 30.0], dtype=np.float32)
        count = int(camera.shape[0])
        return LinkedCalibrationObservation(pattern_id, pattern_id, zone, camera, projector, np.arange(count), count, count)

    dense = np.asarray(
        [[x, y] for y in np.linspace(24.0, 576.0, 8) for x in np.linspace(40.0, 960.0, 10)],
        dtype=np.float32,
    )
    result = solve_linked_projection_calibration(
        [observation("full", "full", dense[:40]), observation("center", "center", dense[40:])],
        (1280, 800),
        table_polygon_cam=np.asarray([[0, 0], [1000, 0], [1000, 600], [0, 600]], dtype=np.float32),
    )

    assert result.summary["pocket_zones_used"] == 0
    assert result.projection.is_valid


def test_linked_calibration_rejects_legacy_minimum_geometric_coverage() -> None:
    normalized = np.asarray(
        [
            [0.21, 0.475],
            [0.365, 0.20],
            [0.50, 0.10],
            [0.635, 0.20],
            [0.79, 0.475],
            [0.635, 0.75],
            [0.50, 0.85],
            [0.365, 0.75],
        ],
        dtype=np.float32,
    )
    camera = normalized * np.asarray([1000.0, 600.0], dtype=np.float32)
    projector = camera * np.asarray([1.10, 1.05], dtype=np.float32) + np.asarray([20.0, 30.0], dtype=np.float32)
    zones = ["full", "center", "pocket_lt", "pocket_rb"]
    camera = np.tile(camera, (3, 1))
    projector = np.tile(projector, (3, 1))
    observations = [
        LinkedCalibrationObservation(zone, zone, zone, camera, projector, np.arange(24), 24, 24)
        for zone in zones
    ]

    with pytest.raises(RuntimeError, match="incomplete full-table coverage"):
        solve_linked_projection_calibration(
            observations,
            (1280, 800),
            table_polygon_cam=np.asarray([[0, 0], [1000, 0], [1000, 600], [0, 600]], dtype=np.float32),
        )


def test_linked_calibration_rejects_named_zones_when_points_are_clustered() -> None:
    camera = np.asarray(
        [[x, y] for y in (250.0, 300.0, 350.0) for x in (400.0, 500.0, 600.0)],
        dtype=np.float32,
    )
    projector = camera * np.asarray([1.10, 1.05], dtype=np.float32) + np.asarray([20.0, 30.0], dtype=np.float32)
    camera = np.tile(camera, (2, 1))
    projector = np.tile(projector, (2, 1))
    zones = ["full", "center", "pocket_lt", "pocket_rt", "pocket_rb", "pocket_lb"]
    observations = [
        LinkedCalibrationObservation(zone, zone, zone, camera, projector, np.arange(18), 18, 18)
        for zone in zones
    ]

    with pytest.raises(RuntimeError, match="width coverage"):
        solve_linked_projection_calibration(
            observations,
            (1280, 800),
            table_polygon_cam=np.asarray([[0, 0], [1000, 0], [1000, 600], [0, 600]], dtype=np.float32),
        )
