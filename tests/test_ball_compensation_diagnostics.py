from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bas.calibration import (
    BallCompensationSample,
    CameraCalibration,
    CalibrationService,
    ProjectionCalibration,
    build_engineered_ball_sampling_grid,
    fit_with_reused_holdout_diagnostic,
    load_reusable_holdout_source,
)
from bas.schemas import TableModel
from bas.ui.ball_compensation_residual_view import (
    render_table_ball_residual_view,
    render_training_residual_bubble_chart,
    residual_color_bgr,
)


def _sample(index: int, point: np.ndarray) -> BallCompensationSample:
    xy = (float(point[0]), float(point[1]))
    return BallCompensationSample(
        sample_index=index,
        target_table_mm=xy,
        detected_camera_px=xy,
        projected_target_px=xy,
        expected_camera_px=xy,
        observed_table_mm=xy,
        delta_table_mm=(0.0, 0.0),
        detected_radius_px=20.0,
        detection_confidence=0.95,
        stability_spread_px=0.1,
        geometry_quality=0.9,
        geometry_method="appearance_ellipse",
        detector_version="test",
    )


def _service() -> CalibrationService:
    corners = np.asarray([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]])
    projection = ProjectionCalibration.fit_from_correspondences(corners, corners, projector_size=(1000, 500))
    projection.table_polygon_cam = corners.copy()
    projection.table_polygon_proj = corners.copy()
    return CalibrationService(
        CameraCalibration(metadata={}),
        projection,
        TableModel(1000.0, 500.0, 57.15, corners.tolist(), []),
    )


def test_reusable_holdout_source_loads_10_locations_representing_30_observations(tmp_path: Path) -> None:
    grid = build_engineered_ball_sampling_grid(1000.0, 500.0, 57.15, cols=8, rows=7)
    training = [_sample(index, point) for index, point in enumerate(grid)]
    holdout = [_sample(100 + index, point) for index, point in enumerate(grid[::6])]
    path = tmp_path / "ball.json"
    path.write_text(
        json.dumps(
            {
                "sampling_grid_table_mm": grid.tolist(),
                "calibration_context": {"camera_coordinate_domain": "raw"},
                "samples": [sample.to_dict() for sample in training],
                "holdout_samples": [sample.to_dict() for sample in holdout],
                "diagnostic_holdout_repeatability": {
                    "location_count": 10,
                    "repeat_count": 3,
                    "observation_count": 30,
                },
            }
        ),
        encoding="utf-8",
    )

    source = load_reusable_holdout_source(
        path,
        expected_sampling_grid_table_mm=grid,
        expected_calibration_context={"camera_coordinate_domain": "raw"},
    )

    assert source is not None
    assert len(source.samples) == 10
    assert source.observation_count == 30


def test_reusable_holdout_source_rejects_another_sampling_grid(tmp_path: Path) -> None:
    path = tmp_path / "ball.json"
    path.write_text(
        json.dumps(
            {
                "sampling_grid_table_mm": [[0.0, 0.0]],
                "holdout_samples": [_sample(0, np.asarray([0.0, 0.0])).to_dict()] * 10,
            }
        ),
        encoding="utf-8",
    )

    source = load_reusable_holdout_source(
        path,
        expected_sampling_grid_table_mm=[[20.0, 20.0]],
    )

    assert source is None


def test_reused_holdout_diagnostic_restores_service_and_never_claims_formal_holdout(tmp_path: Path) -> None:
    service = _service()
    previous_model = service.ball_compensation_model
    grid = build_engineered_ball_sampling_grid(1000.0, 500.0, 57.15, cols=8, rows=7)
    training = [_sample(index, point) for index, point in enumerate(grid)]
    holdout = [_sample(100 + index, point) for index, point in enumerate(grid[::6])]
    artifact = tmp_path / "ball.json"
    artifact.write_text(
        json.dumps(
            {
                "sampling_grid_table_mm": grid.tolist(),
                "holdout_samples": [sample.to_dict() for sample in holdout],
                "diagnostic_holdout_repeatability": {"observation_count": 30},
            }
        ),
        encoding="utf-8",
    )
    source = load_reusable_holdout_source(artifact, expected_sampling_grid_table_mm=grid)

    diagnostic = fit_with_reused_holdout_diagnostic(training, service, source)

    assert service.ball_compensation_model is previous_model
    assert diagnostic.training_report["sample_count"] == 56
    assert diagnostic.holdout_report["observation_count"] == 30
    assert diagnostic.holdout_report["diagnostic_only"] is True
    assert diagnostic.holdout_report["formal_holdout"] is False


def test_residual_view_draws_all_threshold_colors() -> None:
    samples = []
    for index, error in enumerate((0.5, 1.5, 3.0, 5.5)):
        samples.append(
            {
                "sample_index": index,
                "target_table_mm": [100.0 + index * 200.0, 250.0],
                "predicted_table_mm": [100.0 + index * 200.0 + error, 250.0],
                "error_vector_mm": [error, 0.0],
                "error_mm": error,
            }
        )
    image = render_table_ball_residual_view(
        1000.0,
        500.0,
        {"sample_count": 4, "error_mm": {"mean": 2.625, "median": 2.25, "p95": 5.1, "max": 5.5}, "samples": samples},
        output_size=(900, 500),
    )

    assert image.shape == (500, 900, 3)
    for error in (0.5, 1.5, 3.0, 5.5):
        color = np.asarray(residual_color_bgr(error), dtype=np.uint8)
        assert np.any(np.all(image == color, axis=2))


def test_training_residual_bubble_chart_matches_three_operator_thresholds() -> None:
    samples = []
    for index, error in enumerate((1.5, 3.0, 6.0)):
        samples.append(
            {
                "sample_index": index,
                "target_table_mm": [200.0 + index * 300.0, 250.0],
                "error_vector_mm": [error, 0.0],
                "error_mm": error,
            }
        )
    image = render_training_residual_bubble_chart(
        1000.0,
        500.0,
        {"sample_count": 3, "samples": samples},
        output_size=(1000, 560),
    )

    assert image.shape == (560, 1000, 3)
    for rgb in ((0, 108, 54), (49, 151, 244), (163, 65, 14)):
        bgr = np.asarray(rgb[::-1], dtype=np.uint8)
        assert np.any(np.all(image == bgr, axis=2))
