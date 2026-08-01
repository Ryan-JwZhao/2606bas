from __future__ import annotations

import cv2
import numpy as np

from bas.calibration.ball_compensation import BallCompensationModel
from bas.calibration.camera import CameraCalibration
from bas.calibration.projector import ProjectionCalibration, ResidualField
from bas.calibration.service import CalibrationService
from bas.schemas import TableModel


def _table() -> TableModel:
    return TableModel(
        width_mm=1000.0,
        height_mm=500.0,
        ball_diameter_mm=57.15,
        inner_polygon_mm=[(0.0, 0.0), (1000.0, 0.0), (1000.0, 500.0), (0.0, 500.0)],
        pockets_mm=[],
    )


def _projection(projector_polygon: np.ndarray) -> ProjectionCalibration:
    camera_polygon = np.asarray(
        [[100.0, 80.0], [1100.0, 80.0], [1100.0, 580.0], [100.0, 580.0]],
        dtype=np.float64,
    )
    projection = ProjectionCalibration.fit_from_correspondences(
        camera_polygon,
        projector_polygon,
        projector_size=(1280, 800),
    )
    projection.table_polygon_cam = camera_polygon
    projection.table_polygon_proj = projector_polygon.astype(np.float64)
    return projection


def test_camera_table_mapping_is_independent_from_projector_direction() -> None:
    first_projection = _projection(
        np.asarray([[20.0, 30.0], [1240.0, 70.0], [1190.0, 760.0], [45.0, 730.0]], dtype=np.float64)
    )
    second_projection = _projection(
        np.asarray([[160.0, 10.0], [1260.0, 180.0], [1050.0, 790.0], [5.0, 610.0]], dtype=np.float64)
    )
    first = CalibrationService(CameraCalibration(metadata={}), first_projection, _table())
    second = CalibrationService(CameraCalibration(metadata={}), second_projection, _table())
    camera_points = np.asarray([[200.0, 130.0], [600.0, 330.0], [1000.0, 530.0]], dtype=np.float32)

    first_table = first.camera_px_to_table_mm(camera_points)
    second_table = second.camera_px_to_table_mm(camera_points)

    np.testing.assert_allclose(first_table, second_table, atol=1e-4)
    np.testing.assert_allclose(first_table[1], [500.0, 250.0], atol=1e-4)
    assert first.geometry_quality_report["camera_extrinsics_used"] == 0


def test_table_projector_smooth_map_has_a_stable_inverse() -> None:
    projector_polygon = np.asarray(
        [[30.0, 20.0], [1230.0, 50.0], [1200.0, 760.0], [50.0, 730.0]],
        dtype=np.float64,
    )
    projection = _projection(projector_polygon)
    controls = np.asarray(
        [[200.0, 130.0], [600.0, 130.0], [1000.0, 130.0], [200.0, 330.0], [600.0, 330.0], [1000.0, 330.0],
         [200.0, 530.0], [600.0, 530.0], [1000.0, 530.0]],
        dtype=np.float64,
    )
    offsets = np.column_stack(
        [
            0.8 + 0.001 * (controls[:, 0] - 600.0),
            -0.5 + 0.0015 * (controls[:, 1] - 330.0),
        ]
    )
    projection.residual_field = ResidualField(control_points_cam=controls, offsets_proj=offsets)
    service = CalibrationService(CameraCalibration(metadata={}), projection, _table())
    table_points = np.asarray([[150.0, 100.0], [500.0, 250.0], [850.0, 400.0]], dtype=np.float32)

    projected = service.table_mm_to_projector_px(table_points)
    restored = service.projector_px_to_table_mm(projected)

    np.testing.assert_allclose(restored, table_points, atol=0.02)
    assert service.geometry_quality_report["projector_residual_cv_p95_px"] < 0.1


def test_ball_map_uses_direct_camera_to_known_table_targets() -> None:
    projection = _projection(
        np.asarray([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]], dtype=np.float64)
    )
    controls = np.asarray(
        [[200.0, 130.0], [600.0, 130.0], [1000.0, 130.0], [200.0, 330.0], [600.0, 330.0], [1000.0, 330.0],
         [200.0, 530.0], [600.0, 530.0], [1000.0, 530.0]],
        dtype=np.float64,
    )
    targets = np.column_stack(
        [
            0.97 * (controls[:, 0] - 100.0) + 4.0,
            0.96 * (controls[:, 1] - 80.0) - 3.0,
        ]
    )
    model = BallCompensationModel(
        mode="engineered_ball_comp_v2",
        control_points_camera_px=controls,
        delta_table_mm=np.zeros_like(controls),
        target_table_mm=targets,
        sample_weights=np.ones((controls.shape[0],), dtype=np.float64),
    )
    service = CalibrationService(
        CameraCalibration(metadata={}),
        projection,
        _table(),
        ball_compensation_model=model,
    )
    probe = np.asarray([[400.0, 230.0], [800.0, 430.0]], dtype=np.float32)
    expected = np.column_stack(
        [
            0.97 * (probe[:, 0] - 100.0) + 4.0,
            0.96 * (probe[:, 1] - 80.0) - 3.0,
        ]
    )

    actual = service.ball_camera_px_to_table_mm(probe)

    np.testing.assert_allclose(actual, expected, atol=0.05)
    assert service.geometry_quality_report["ball_map_cv_p95_mm"] < 0.1


def test_ball_map_blends_continuously_at_calibrated_domain_edge() -> None:
    projection = _projection(
        np.asarray([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]], dtype=np.float64)
    )
    controls = np.asarray(
        [[100.0, 100.0], [500.0, 100.0], [900.0, 100.0], [100.0, 250.0], [500.0, 250.0],
         [900.0, 250.0], [100.0, 400.0], [500.0, 400.0], [900.0, 400.0]],
        dtype=np.float64,
    )
    model = BallCompensationModel(
        mode="direct",
        control_points_camera_px=controls,
        delta_table_mm=np.zeros_like(controls),
        target_table_mm=controls + np.asarray([20.0, 0.0], dtype=np.float64),
        sample_weights=np.ones((controls.shape[0],), dtype=np.float64),
    )
    service = CalibrationService(
        CameraCalibration(metadata={}),
        projection,
        _table(),
        ball_compensation_model=model,
    )
    # The legacy rectangular support edge is x=36 px for this sample span.
    probes = np.asarray([[35.99, 250.0], [36.01, 250.0]], dtype=np.float32)

    mapped = service.ball_camera_px_to_table_mm(probes)

    assert float(np.linalg.norm(mapped[1] - mapped[0])) < 0.5


def test_invalid_projection_fails_closed_instead_of_returning_camera_pixels() -> None:
    service = CalibrationService(CameraCalibration(metadata={}), ProjectionCalibration(), _table())

    with np.testing.assert_raises(RuntimeError):
        service.camera_px_to_projector_px(np.asarray([[120.0, 80.0]], dtype=np.float32))


def test_projected_table_circle_preserves_directional_scale_as_ellipse() -> None:
    projection = _projection(
        np.asarray([[0.0, 0.0], [1000.0, 0.0], [900.0, 500.0], [100.0, 500.0]], dtype=np.float64)
    )
    service = CalibrationService(CameraCalibration(metadata={}), projection, _table())

    ellipse = service.table_circle_to_projector_ellipse((500.0, 250.0), 50.0)

    assert ellipse.radius_x_px > 0.0
    assert ellipse.radius_y_px > 0.0
    assert abs(ellipse.radius_x_px - ellipse.radius_y_px) > 1.0


def test_camera_distortion_roundtrip_does_not_require_extrinsics() -> None:
    camera = CameraCalibration(
        image_size=(640, 480),
        camera_matrix=np.asarray([[520.0, 0.0, 320.0], [0.0, 515.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        distortion_coefficients=np.asarray([[-0.08, 0.02, 0.001, -0.001, 0.0]], dtype=np.float64),
        metadata={},
    )
    raw = np.asarray([[120.0, 90.0], [320.0, 240.0], [530.0, 390.0]], dtype=np.float32)

    undistorted = camera.undistort_points(raw)
    restored = camera.distort_points(undistorted)

    np.testing.assert_allclose(restored, raw, atol=0.03)
    assert camera.rvec is None
    assert camera.tvec is None


def test_dense_table_outline_keeps_perspective_quad_instead_of_axis_aligned_bbox() -> None:
    camera_quad = np.asarray([[100.0, 80.0], [1120.0, 120.0], [1050.0, 590.0], [140.0, 540.0]], dtype=np.float64)
    projector_quad = np.asarray([[30.0, 20.0], [1230.0, 60.0], [1190.0, 760.0], [50.0, 720.0]], dtype=np.float64)
    projection = ProjectionCalibration.fit_from_correspondences(camera_quad, projector_quad, projector_size=(1280, 800))

    def dense_edges(quad: np.ndarray) -> np.ndarray:
        return np.vstack(
            [
                np.linspace(quad[index], quad[(index + 1) % 4], 20, endpoint=False)
                for index in range(4)
            ]
        )

    projection.table_polygon_cam = dense_edges(camera_quad)
    projection.table_polygon_proj = dense_edges(projector_quad)
    service = CalibrationService(CameraCalibration(metadata={}), projection, _table())

    mapped = service.camera_px_to_table_mm(camera_quad.astype(np.float32))

    np.testing.assert_allclose(
        mapped,
        np.asarray([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]], dtype=np.float32),
        atol=0.1,
    )
