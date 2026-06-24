from __future__ import annotations

import numpy as np
import cv2
import pytest

from bas.calibration.camera import CameraCalibration
from bas.calibration.charuco import CharucoBoardSpec, detect_charuco_corners, render_charuco_board
from bas.calibration.projector import ProjectionCalibration
from bas.calibration.service import CalibrationService
from bas.calibration.verification import format_holdout_report, verify_holdout_samples
from bas.schemas import TableModel


def test_projection_residual_maps_control_points_exactly() -> None:
    cam = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [8, 3], [2, 7], [6, 9], [3, 4], [9, 8], [1, 3], [7, 1]], dtype=np.float64)
    proj = cam * 2.0 + np.array([100.0, 50.0])
    proj[4:] += np.array([1.0, -0.5])
    calib = ProjectionCalibration.fit_from_correspondences(cam, proj, projector_size=(300, 200))
    pred = calib.camera_to_projector_points(cam)
    assert np.max(np.linalg.norm(pred - proj, axis=1)) < 1e-3


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
