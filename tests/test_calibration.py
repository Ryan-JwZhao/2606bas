from __future__ import annotations

import numpy as np
import cv2
import pytest

from bas.calibration.camera import CameraCalibration
from bas.calibration.charuco import CharucoBoardSpec, detect_charuco_corners, render_charuco_board
from bas.calibration.projector import ProjectionCalibration
from bas.calibration.service import BallCenterCompensation, CalibrationService
from bas.calibration.verification import format_holdout_report, verify_holdout_samples
from bas.schemas import TableModel


def test_projection_residual_maps_control_points_exactly() -> None:
    cam = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [8, 3], [2, 7], [6, 9], [3, 4], [9, 8], [1, 3], [7, 1]], dtype=np.float64)
    proj = cam * 2.0 + np.array([100.0, 50.0])
    proj[4:] += np.array([1.0, -0.5])
    calib = ProjectionCalibration.fit_from_correspondences(cam, proj, projector_size=(300, 200))
    pred = calib.camera_to_projector_points(cam)
    assert np.max(np.linalg.norm(pred - proj, axis=1)) < 1e-3


def test_projection_residual_ignores_ransac_outliers() -> None:
    cam_good = np.array(
        [
            [0, 0],
            [20, 0],
            [40, 0],
            [60, 0],
            [0, 20],
            [20, 20],
            [40, 20],
            [60, 20],
            [0, 40],
            [20, 40],
            [40, 40],
            [60, 40],
        ],
        dtype=np.float64,
    )
    proj_good = cam_good * 2.0 + np.array([100.0, 50.0])
    cam_bad = np.array(
        [
            [45, 4],
            [50, 4],
            [55, 4],
            [58, 8],
        ],
        dtype=np.float64,
    )
    proj_bad = cam_bad * 2.0 + np.array([100.0, 350.0])

    cam = np.vstack([cam_good, cam_bad])
    proj = np.vstack([proj_good, proj_bad])
    calib = ProjectionCalibration.fit_from_correspondences(cam, proj, projector_size=(400, 300))

    probe = np.array([[60, 0]], dtype=np.float64)
    pred = calib.camera_to_projector_points(probe)[0]
    expected = np.array([220.0, 50.0], dtype=np.float64)
    assert np.linalg.norm(pred - expected) < 2.0
    assert calib.quality_report["ransac_inliers"] == len(cam_good)
    assert calib.quality_report["ransac_outliers"] == len(cam_bad)
    assert calib.calibration_error_stats()["max_px"] > 250.0


def test_rendered_charuco_board_can_be_detected() -> None:
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or (not hasattr(aruco, "CharucoDetector") and not hasattr(aruco, "interpolateCornersCharuco")):
        pytest.skip("OpenCV ChArUco detection support is unavailable.")
    spec = CharucoBoardSpec(squares_x=6, squares_y=5, square_length_m=0.025, marker_length_m=0.018, dictionary_id=0)
    image = render_charuco_board(spec, width_px=1200, height_px=1000)
    corners, ids = detect_charuco_corners(image, spec)

    assert len(corners) > 0
    assert len(corners) == len(ids)


def test_table_mapping_uses_projection_table_bbox() -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64),
        np.array([[10, 20], [210, 20], [210, 120], [10, 120]], dtype=np.float64),
        projector_size=(220, 140),
    )
    projection.table_polygon_proj = np.array([[10, 20], [210, 20], [210, 120], [10, 120]], dtype=np.float64)
    service = CalibrationService(
        camera=CameraCalibration(metadata={}),
        projection=projection,
        table=TableModel(
            width_mm=2000,
            height_mm=1000,
            ball_diameter_mm=57.15,
            inner_polygon_mm=[(0, 0), (2000, 0), (2000, 1000), (0, 1000)],
            pockets_mm=[],
        ),
    )
    out = service.camera_px_to_table_mm(np.array([[50, 25]], dtype=np.float32))
    assert np.allclose(out[0], [1000, 500], atol=1e-3)


def test_table_mapping_uses_projection_table_quad() -> None:
    quad = np.array([[10, 20], [210, 35], [195, 118], [24, 110]], dtype=np.float64)
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64),
        quad,
        projector_size=(220, 140),
    )
    projection.table_polygon_proj = quad.copy()
    service = CalibrationService(
        camera=CameraCalibration(metadata={}),
        projection=projection,
        table=TableModel(
            width_mm=2000,
            height_mm=1000,
            ball_diameter_mm=57.15,
            inner_polygon_mm=[(0, 0), (2000, 0), (2000, 1000), (0, 1000)],
            pockets_mm=[],
        ),
    )

    center_mm = service.camera_px_to_table_mm(np.array([[50, 25]], dtype=np.float32))
    corners_proj = service.table_mm_to_projector_px(np.array([[0, 0], [2000, 0], [2000, 1000], [0, 1000]], dtype=np.float32))
    roundtrip_mm = service.projector_px_to_table_mm(corners_proj)

    assert np.allclose(center_mm[0], [1000, 500], atol=1e-3)
    assert np.allclose(corners_proj, quad, atol=1e-3)
    assert np.allclose(roundtrip_mm, np.array([[0, 0], [2000, 0], [2000, 1000], [0, 1000]], dtype=np.float32), atol=1e-3)


def test_ball_center_compensation_pulls_points_toward_reference() -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float64),
        np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float64),
        projector_size=(200, 100),
    )
    projection.table_polygon_proj = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float64)
    service = CalibrationService(
        camera=CameraCalibration(metadata={}),
        projection=projection,
        table=TableModel(
            width_mm=200,
            height_mm=100,
            ball_diameter_mm=57.15,
            inner_polygon_mm=[(0, 0), (200, 0), (200, 100), (0, 100)],
            pockets_mm=[],
        ),
        ball_center_compensation=BallCenterCompensation(
            enabled=True,
            ref_x_px=100.0,
            ref_y_px=50.0,
            scale_x_pct=10.0,
            scale_y_pct=10.0,
        ),
    )

    out = service.ball_camera_px_to_table_mm(np.array([[160, 80]], dtype=np.float32))

    assert np.allclose(out[0], [154.0, 77.0], atol=1e-3)


def test_holdout_verification_reports_mm_and_image_errors() -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64),
        np.array([[10, 20], [210, 20], [210, 120], [10, 120]], dtype=np.float64),
        projector_size=(220, 140),
    )
    projection.table_polygon_proj = np.array([[10, 20], [210, 20], [210, 120], [10, 120]], dtype=np.float64)
    service = CalibrationService(
        camera=CameraCalibration(metadata={}),
        projection=projection,
        table=TableModel(
            width_mm=2000,
            height_mm=1000,
            ball_diameter_mm=57.15,
            inner_polygon_mm=[(0, 0), (2000, 0), (2000, 1000), (0, 1000)],
            pockets_mm=[],
        ),
    )
    report = verify_holdout_samples(
        [
            {"camera_px": [50, 25], "projector_px": [110, 70], "world_mm": [1000, 500], "zone": "middle"},
            {"camera_px": [75, 25], "projector_px": [160, 70], "world_mm": [1500, 500], "zone": "middle"},
        ],
        service,
    )

    assert report["image_error_px"]["p95"] < 1e-3
    assert report["table_error_mm"]["p95"] < 1e-3
    assert report["verdict"]["formal"]
    assert "正式: 通过" in format_holdout_report(report)
