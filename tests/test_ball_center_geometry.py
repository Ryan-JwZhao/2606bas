from __future__ import annotations

import numpy as np

from bas.ball_center_geometry import BallCenterGeometry
from bas.calibration.ball_compensation import BallCompensationModel
from bas.calibration.camera import CameraCalibration
from bas.calibration.projector import ProjectionCalibration
from bas.calibration.service import CalibrationService
from bas.schemas import TableModel


def _calibration() -> CalibrationService:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.asarray([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64),
        np.asarray([[0, 0], [1000, 0], [900, 500], [100, 500]], dtype=np.float64),
        projector_size=(1000, 500),
    )
    projection.table_polygon_cam = np.asarray([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64)
    projection.table_polygon_proj = np.asarray([[0, 0], [1000, 0], [900, 500], [100, 500]], dtype=np.float64)
    controls = np.asarray(
        [[100, 100], [500, 100], [900, 100], [100, 250], [500, 250], [900, 250], [100, 400], [500, 400], [900, 400]],
        dtype=np.float64,
    )
    return CalibrationService(
        camera=CameraCalibration(metadata={}),
        projection=projection,
        ball_compensation_model=BallCompensationModel(
            mode="engineered_ball_comp_v3",
            control_points_camera_px=controls,
            delta_table_mm=np.zeros_like(controls),
            target_table_mm=controls,
            sample_weights=np.ones((len(controls),), dtype=np.float64),
            quality_report={"quality_gate_passed": True},
        ),
        table=TableModel(1000.0, 500.0, 57.15, [(0, 0), (1000, 0), (1000, 500), (0, 500)], []),
    )


def test_ball_center_geometry_returns_one_shared_physical_result() -> None:
    module = BallCenterGeometry(_calibration())

    result = module.locate((500.0, 250.0), radius_px=14.0, geometry_quality=0.9, geometry_method="appearance_ellipse")

    np.testing.assert_allclose(result.table_center_mm, (500.0, 250.0), atol=0.1)
    assert result.radius_mm == 28.575
    assert result.projector_ellipse.radius_x_px > 0.0
    assert result.uncertainty_mm > 0.0
    assert result.support_weight > 0.95


def test_bbox_center_has_materially_higher_uncertainty() -> None:
    module = BallCenterGeometry(_calibration())
    refined = module.locate((500.0, 250.0), geometry_quality=0.9, geometry_method="appearance_ellipse")
    bbox = module.locate((500.0, 250.0), geometry_quality=0.45, geometry_method="bbox")

    assert bbox.uncertainty_mm >= refined.uncertainty_mm * 1.8
    assert bbox.reliability < refined.reliability * 0.4


def test_base_geometry_without_ball_model_keeps_full_support() -> None:
    calibration = _calibration()
    calibration.ball_compensation_model = BallCompensationModel()
    calibration._rebuild_geometry()
    module = BallCenterGeometry(calibration)

    result = module.locate((500.0, 250.0), geometry_quality=0.9, geometry_method="appearance_ellipse")

    assert result.support_weight == 1.0
    assert result.reliability > 0.70
