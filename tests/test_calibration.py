from __future__ import annotations

import json

import numpy as np
import cv2
import pytest

from bas.calibration.camera import CameraCalibration
from bas.calibration.ball_compensation import BallCompensationModel
from bas.calibration.charuco import CharucoBoardSpec, detect_charuco_corners, render_charuco_board
from bas.calibration.projector import ProjectionCalibration
from bas.calibration.service import (
    BallCenterCompensation,
    CalibrationService,
    create_calibration_service,
    create_setting_aware_calibration_service,
)
from bas.config import CalibrationConfig, CameraConfig, ProjectionConfig
from bas.geometry_contract import projection_calibration_context
from bas.calibration.verification import format_holdout_report, verify_holdout_samples
from bas.schemas import TableModel


def test_create_calibration_service_does_not_load_camera_file_when_correction_is_disabled(monkeypatch) -> None:
    def unexpected_load(_path):
        raise AssertionError("disabled correction must not load camera calibration")

    monkeypatch.setattr(CameraCalibration, "load_opencv_yaml", unexpected_load)

    service = create_calibration_service(
        CalibrationConfig(camera_file="must_not_load.yaml"),
        distortion_correction_enabled=False,
    )

    frame = np.ones((4, 6, 3), dtype=np.uint8)
    assert service.camera.is_valid is False
    assert service.undistort_frame(frame) is frame


def test_setting_aware_calibration_service_does_not_override_disabled_correction(monkeypatch) -> None:
    def unexpected_load(_path):
        raise AssertionError("disabled setting must apply to every calibration tool")

    monkeypatch.setattr(CameraCalibration, "load_opencv_yaml", unexpected_load)

    service = create_setting_aware_calibration_service(
        CalibrationConfig(camera_file="must_not_load.yaml"),
        CameraConfig(
            distortion_correction_enabled=False,
            distortion_correction_file="must_not_load.yaml",
        ),
    )
    points = np.asarray([[12.0, 34.0]], dtype=np.float32)

    assert service.camera.is_valid is False
    assert service.distortion_correction_enabled is False
    with pytest.raises(RuntimeError, match="Projection calibration"):
        service.camera_px_to_projector_px(points)


def test_setting_aware_service_uses_actual_capture_size_for_projection_context(tmp_path) -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float64),
        np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float64),
        projector_size=(1280, 800),
    )
    projection.calibration_context = projection_calibration_context(
        frame_width=1280,
        frame_height=720,
        frame_rotation_degrees=0,
        camera_coordinate_domain="raw",
        distortion_file=None,
        projector_width=1280,
        projector_height=800,
        projection_calibration_rotation_degrees=0,
    )
    path = tmp_path / "actual_capture_size_projection.json"
    projection.save(path)

    service = create_setting_aware_calibration_service(
        CalibrationConfig(projection_file=str(path)),
        CameraConfig(
            width=1920,
            height=1080,
            distortion_correction_enabled=False,
        ),
        projection_config=ProjectionConfig(
            projector_width=1280,
            projector_height=800,
            calibration_rotation_degrees=0,
        ),
        actual_frame_size=(1280, 720),
    )

    assert service.projection.is_valid is True
    assert service.projection.compatibility_errors == ()


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
            auto_reference=False,
            ref_x_px=100.0,
            ref_y_px=50.0,
            scale_x_pct=10.0,
            scale_y_pct=10.0,
        ),
    )

    out = service.ball_camera_px_to_table_mm(np.array([[160, 80]], dtype=np.float32))

    assert np.allclose(out[0], [154.0, 77.0], atol=1e-3)


def test_ball_center_compensation_auto_reference_uses_image_center() -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float64),
        np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float64),
        projector_size=(200, 100),
    )
    projection.table_polygon_proj = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float64)
    service = CalibrationService(
        camera=CameraCalibration(image_size=(200, 100), metadata={}),
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
            auto_reference=True,
            ref_x_px=-500.0,
            ref_y_px=-300.0,
            scale_x_pct=10.0,
            scale_y_pct=10.0,
        ),
    )

    out = service.ball_camera_px_to_table_mm(np.array([[160, 80]], dtype=np.float32))

    assert np.allclose(out[0], [154.0, 77.0], atol=1e-3)


def test_ball_center_compensation_can_sync_from_live_config() -> None:
    service = create_calibration_service(
        CalibrationConfig(),
        distortion_correction_enabled=False,
    )
    config = CalibrationConfig(
        ball_center_compensation_enabled=True,
        ball_center_compensation_auto_reference=False,
        ball_center_compensation_ref_x_px=321.0,
        ball_center_compensation_ref_y_px=654.0,
        ball_center_compensation_scale_x_pct=2.5,
        ball_center_compensation_scale_y_pct=-1.5,
    )

    service.sync_ball_center_compensation(config)

    assert service.ball_center_compensation == BallCenterCompensation(
        enabled=True,
        auto_reference=False,
        ref_x_px=321.0,
        ref_y_px=654.0,
        scale_x_pct=2.5,
        scale_y_pct=-1.5,
    )


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
    assert report["verdict"]["formal"] is False
    assert report["coverage"]["formal"] is False
    assert "正式: 未通过" in format_holdout_report(report)


def test_holdout_formal_verdict_requires_and_accepts_spatial_coverage() -> None:
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
    zones = [
        "center",
        "edge_top",
        "edge_bottom",
        "edge_left",
        "pocket_lt",
        "pocket_rt",
        "pocket_lb",
        "pocket_rb",
    ]
    samples = []
    for index in range(24):
        x = 10.0 + float((index % 6) * 15)
        y = 8.0 + float((index // 6) * 10)
        samples.append(
            {
                "camera_px": [x, y],
                "projector_px": [2.0 * x + 10.0, 2.0 * y + 20.0],
                "world_mm": [20.0 * x, 20.0 * y],
                "zone": zones[index % len(zones)],
            }
        )

    report = verify_holdout_samples(samples, service)

    assert report["coverage"]["formal"] is True
    assert report["verdict"]["formal"] is True


def test_holdout_ball_samples_use_ball_center_geometry() -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64),
        np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64),
        projector_size=(100, 50),
    )
    projection.table_polygon_cam = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64)
    projection.table_polygon_proj = projection.table_polygon_cam.copy()
    controls = np.asarray(
        [[0, 0], [100, 0], [0, 50], [100, 50], [50, 25], [25, 25]],
        dtype=np.float64,
    )
    targets = controls + np.asarray([2.0, 3.0], dtype=np.float64)
    service = CalibrationService(
        camera=CameraCalibration(metadata={}),
        projection=projection,
        table=TableModel(
            width_mm=100,
            height_mm=50,
            ball_diameter_mm=57.15,
            inner_polygon_mm=[(0, 0), (100, 0), (100, 50), (0, 50)],
            pockets_mm=[],
        ),
        projection_mode="engineered",
        ball_compensation_model=BallCompensationModel(
            mode="direct",
            control_points_camera_px=controls,
            delta_table_mm=np.zeros_like(controls),
            target_table_mm=targets,
        ),
    )

    report = verify_holdout_samples(
        [{"camera_px": [50, 25], "world_mm": [52, 28], "kind": "ball_center"}],
        service,
    )

    assert report["table_error_mm"]["p95"] < 1e-3
    assert report["geometry_model"]["camera_extrinsics_used"] == 0


def test_single_holdout_sample_cannot_receive_formal_verdict() -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64),
        np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64),
        projector_size=(100, 50),
    )
    projection.table_polygon_cam = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64)
    projection.table_polygon_proj = projection.table_polygon_cam.copy()
    service = CalibrationService(CameraCalibration(metadata={}), projection, TableModel(100, 50, 57.15, [(0, 0), (100, 0), (100, 50), (0, 50)], []))

    report = verify_holdout_samples(
        [{"camera_px": [50, 25], "world_mm": [50, 25], "zone": "center"}],
        service,
    )

    assert report["verdict"]["formal"] is False
    assert report["verdict"]["sample_coverage"] is False


def test_projection_loader_rejects_null_homography(tmp_path) -> None:
    path = tmp_path / "invalid_projection.json"
    path.write_text(json.dumps({"mode": "invalid", "homography": None}), encoding="utf-8")

    projection = ProjectionCalibration.load_json(path)

    assert projection.is_valid is False


def test_projection_context_mismatch_rejects_rotated_camera_artifact(tmp_path) -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64),
        np.array([[10, 20], [210, 20], [210, 120], [10, 120]], dtype=np.float64),
    )
    projection.calibration_context = {
        "frame_rotation_degrees": 0,
        "camera_coordinate_domain": "raw",
    }
    path = tmp_path / "projection_context.json"
    projection.save(path)

    loaded = ProjectionCalibration.load_json(
        path,
        expected_context={"frame_rotation_degrees": 180, "camera_coordinate_domain": "raw"},
    )

    assert loaded.is_valid is False
    assert "frame_rotation_degrees" in loaded.compatibility_errors


def test_engineered_calibration_service_loads_plane_and_ball_compensation(tmp_path) -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64),
        np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64),
        projector_size=(1000, 500),
    )
    projection.table_polygon_proj = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64)
    plane_path = tmp_path / "engineered_plane.json"
    projection.save(plane_path)
    ball_path = tmp_path / "engineered_ball.json"
    ball_path.write_text(
        json.dumps(
            {
                "mode": "engineered_ball_comp_v1",
                "control_camera_points": [[100.0, 200.0]],
                "delta_table_mm": [[10.0, -5.0]],
            }
        ),
        encoding="utf-8",
    )

    cfg = CalibrationConfig(
        projection_mode="engineered",
        engineered_plane_projection_file=str(plane_path),
        engineered_ball_compensation_file=str(ball_path),
        table_width_mm=1000.0,
        table_height_mm=500.0,
        ball_diameter_mm=57.15,
    )

    service = create_calibration_service(cfg, distortion_correction_enabled=True)
    out = service.ball_camera_px_to_table_mm(np.array([[100.0, 200.0]], dtype=np.float32))

    assert service.projection_mode == "engineered"
    assert service.projection.source_path == str(plane_path)
    assert service.ball_compensation_model.source_path == str(ball_path)
    assert np.allclose(out[0], [110.0, 195.0], atol=1e-3)
    assert service.ball_pixel_radius_to_mm((100.0, 200.0), 99.0) == pytest.approx(28.575, abs=1e-6)


def test_engineered_service_audits_and_rejects_bad_legacy_full_grid_model(tmp_path) -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.float64),
        np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.float64),
        projector_size=(1920, 1080),
    )
    projection.table_polygon_proj = np.array(
        [[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
        dtype=np.float64,
    )
    plane_path = tmp_path / "engineered_plane.json"
    projection.save(plane_path)

    controls = np.column_stack(
        [
            np.tile(np.linspace(160.0, 1760.0, 6), 5),
            np.repeat(np.linspace(140.0, 940.0, 5), 6),
        ]
    )
    targets = np.random.default_rng(20260801).uniform(
        [0.0, 0.0],
        [2540.0, 1270.0],
        size=(30, 2),
    )
    ball_path = tmp_path / "legacy_bad_ball.json"
    ball_path.write_text(
        json.dumps(
            {
                "mode": "engineered_ball_comp_v2",
                "control_camera_points": controls.tolist(),
                "target_table_mm": targets.tolist(),
            }
        ),
        encoding="utf-8",
    )
    cfg = CalibrationConfig(
        projection_mode="engineered",
        engineered_plane_projection_file=str(plane_path),
        engineered_ball_compensation_file=str(ball_path),
        table_width_mm=2540.0,
        table_height_mm=1270.0,
        ball_diameter_mm=57.15,
    )

    service = create_calibration_service(cfg, distortion_correction_enabled=False)

    assert service.ball_compensation_model.is_valid is False
    assert service.ball_compensation_model.quality_report["legacy_model_audited"] is True
    assert service.ball_compensation_model.quality_report["quality_gate_passed"] is False
    assert "cross-validation" in " ".join(
        service.ball_compensation_model.quality_report["quality_gate_errors"]
    )
