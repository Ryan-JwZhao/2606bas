from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from bas.calibration import BallCompensationModel
from bas.calibration.ball_compensation_sampling import (
    BallCompensationSample,
    build_ball_compensation_model,
    build_engineered_ball_sampling_grid,
    update_calibration_table_boundaries_from_geometry_frame,
)
from bas.calibration.camera import CameraCalibration
from bas.calibration.projector import ProjectionCalibration
from bas.calibration.service import CalibrationService
from bas.geometry import TableGeometry
from bas.schemas import TableModel
from bas.table_boundaries import EdgeInsets
from bas.ui.engineered_ball_compensation_wizard import (
    ball_compensation_path_or_default,
    ball_compensation_path_from_input,
    timestamped_ball_compensation_output_path,
)


def test_build_engineered_ball_sampling_grid_covers_table_interior() -> None:
    grid = build_engineered_ball_sampling_grid(2540.0, 1270.0, 57.15, cols=5, rows=4)

    assert grid.shape == (20, 2)
    assert float(np.min(grid[:, 0])) > 0.0
    assert float(np.min(grid[:, 1])) > 0.0
    assert float(np.max(grid[:, 0])) < 2540.0
    assert float(np.max(grid[:, 1])) < 1270.0
    assert len(np.unique(grid[:, 0])) == 5
    assert len(np.unique(grid[:, 1])) == 4


def test_build_engineered_ball_sampling_grid_stays_inside_center_playable_polygon() -> None:
    polygon = np.asarray(
        [
            [80.0, 60.0],
            [920.0, 60.0],
            [920.0, 440.0],
            [560.0, 440.0],
            [500.0, 400.0],
            [440.0, 440.0],
            [80.0, 440.0],
        ],
        dtype=np.float32,
    )
    grid = build_engineered_ball_sampling_grid(
        1000.0,
        500.0,
        57.15,
        cols=5,
        rows=4,
        preferred_polygon_mm=polygon,
        extra_safe_inset_mm=28.0,
    )

    assert grid.shape[0] >= 12
    for point in grid:
        inside = cv2.pointPolygonTest(polygon.reshape((-1, 1, 2)), (float(point[0]), float(point[1])), False)
        assert inside >= 0.0
    assert float(np.min(grid[:, 0])) > 100.0
    assert float(np.max(grid[:, 1])) < 430.0


def test_update_calibration_table_boundaries_from_geometry_frame_refreshes_center_polygon() -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64),
        np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64),
        projector_size=(1000, 500),
    )
    projection.table_polygon_proj = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64)
    service = CalibrationService(
        camera=CameraCalibration(metadata={}),
        projection=projection,
        table=TableModel(
            width_mm=1000.0,
            height_mm=500.0,
            ball_diameter_mm=57.15,
            inner_polygon_mm=[(0.0, 0.0), (1000.0, 0.0), (1000.0, 500.0), (0.0, 500.0)],
            pockets_mm=[],
        ),
    )
    geometry = TableGeometry(
        inner_norm=np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        pockets_norm=[],
    )

    changed = update_calibration_table_boundaries_from_geometry_frame(
        service,
        geometry,
        (500, 1000, 3),
        projection_visible_insets=EdgeInsets(),
        physical_rail_insets=EdgeInsets.uniform(10.0),
        physical_middle_pocket_relief_top_mm=0.0,
        physical_middle_pocket_relief_bottom_mm=0.0,
        center_reachable_extra_margin_mm=2.0,
    )
    center_poly = np.asarray(service.table.center_playable_polygon_mm, dtype=np.float32).reshape((-1, 2))

    assert changed is True
    assert center_poly.shape[0] >= 4
    assert float(np.min(center_poly[:, 0])) > 35.0
    assert float(np.min(center_poly[:, 1])) > 35.0
    assert float(np.max(center_poly[:, 0])) < 965.0
    assert float(np.max(center_poly[:, 1])) < 465.0


def test_build_ball_compensation_model_reports_quality_and_roundtrips(tmp_path) -> None:
    samples = [
        BallCompensationSample(
            sample_index=0,
            target_table_mm=(300.0, 220.0),
            detected_camera_px=(200.0, 180.0),
            projected_target_px=(120.0, 90.0),
            expected_camera_px=(210.0, 185.0),
            observed_table_mm=(296.0, 222.0),
            delta_table_mm=(4.0, -2.0),
            detected_radius_px=18.0,
            detection_confidence=0.92,
            stability_spread_px=1.2,
        ),
        BallCompensationSample(
            sample_index=1,
            target_table_mm=(900.0, 220.0),
            detected_camera_px=(620.0, 184.0),
            projected_target_px=(420.0, 92.0),
            expected_camera_px=(615.0, 187.0),
            observed_table_mm=(894.0, 225.0),
            delta_table_mm=(6.0, -5.0),
            detected_radius_px=19.0,
            detection_confidence=0.95,
            stability_spread_px=1.8,
        ),
        BallCompensationSample(
            sample_index=2,
            target_table_mm=(300.0, 860.0),
            detected_camera_px=(205.0, 520.0),
            projected_target_px=(125.0, 310.0),
            expected_camera_px=(209.0, 518.0),
            observed_table_mm=(298.0, 854.0),
            delta_table_mm=(2.0, 6.0),
            detected_radius_px=17.5,
            detection_confidence=0.91,
            stability_spread_px=1.4,
        ),
        BallCompensationSample(
            sample_index=3,
            target_table_mm=(900.0, 860.0),
            detected_camera_px=(624.0, 523.0),
            projected_target_px=(424.0, 312.0),
            expected_camera_px=(618.0, 520.0),
            observed_table_mm=(895.0, 852.0),
            delta_table_mm=(5.0, 8.0),
            detected_radius_px=18.4,
            detection_confidence=0.97,
            stability_spread_px=1.1,
        ),
    ]

    model = build_ball_compensation_model(samples, ball_diameter_mm=57.15, max_neighbors=8)
    path = tmp_path / "engineered_ball_compensation.json"
    model.save_json(
        path,
        extra_data={
            "ball_diameter_mm": 57.15,
            "samples": [sample.to_dict() for sample in samples],
        },
    )
    reloaded = BallCompensationModel.load_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert model.max_neighbors == 4
    assert model.quality_report["sample_count"] == 4
    assert model.quality_report["delta_norm_mm"]["max"] > 0.0
    assert reloaded.is_valid is True
    assert reloaded.max_neighbors == 4
    assert np.allclose(reloaded.control_points_camera_px, np.asarray([sample.detected_camera_px for sample in samples], dtype=np.float64))
    assert payload["ball_diameter_mm"] == 57.15
    assert len(payload["samples"]) == 4


def test_ball_compensation_path_from_input_resolves_relative_to_current_directory() -> None:
    current = r"C:\calib\engineered_ball_compensation.json"
    out = ball_compensation_path_from_input("next_ball.json", current)
    assert out == Path(r"C:\calib\next_ball.json")


def test_ball_compensation_path_from_input_empty_uses_default_file_template() -> None:
    out = ball_compensation_path_from_input("", None)
    assert out == Path(r"C:\CodeProject\2606BAS\local_settings\calibrations\engineered_ball_compensation.json")


def test_ball_compensation_path_or_default_turns_directory_into_default_file(tmp_path) -> None:
    out = ball_compensation_path_or_default(str(tmp_path))
    assert out == tmp_path / "engineered_ball_compensation.json"


def test_timestamped_ball_compensation_output_path_replaces_existing_timestamp(monkeypatch) -> None:
    from bas.ui import engineered_ball_compensation_wizard as wizard

    monkeypatch.setattr(wizard.time, "strftime", lambda _fmt: "20260626_154233")
    out = timestamped_ball_compensation_output_path(
        r"C:\calib\engineered_ball_compensation_20260625_173306.json"
    )
    assert out == Path(
        r"C:\CodeProject\2606BAS\local_settings\calibrations\engineered_ball_compensation_20260626_154233.json"
    )


def test_timestamped_ball_compensation_output_path_uses_default_stem_for_directory(monkeypatch) -> None:
    from bas.ui import engineered_ball_compensation_wizard as wizard

    monkeypatch.setattr(wizard.time, "strftime", lambda _fmt: "20260626_154233")
    out = timestamped_ball_compensation_output_path(r"C:\CodeProject\2606BAS\local_settings\calibrations")
    assert out == Path(
        r"C:\CodeProject\2606BAS\local_settings\calibrations\engineered_ball_compensation_20260626_154233.json"
    )
