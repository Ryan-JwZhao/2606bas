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
    evaluate_ball_compensation_holdout,
    split_ball_compensation_samples,
    update_calibration_table_boundaries_from_geometry_frame,
)


def test_dense_samples_reserve_spread_holdout_without_overlap() -> None:
    samples = [
        BallCompensationSample(
            sample_index=index,
            target_table_mm=(float(index % 8) * 300.0, float(index // 8) * 180.0),
            detected_camera_px=(float(index % 8) * 100.0, float(index // 8) * 80.0),
            projected_target_px=(0.0, 0.0),
            expected_camera_px=(0.0, 0.0),
            observed_table_mm=(0.0, 0.0),
            delta_table_mm=(0.0, 0.0),
        )
        for index in range(56)
    ]

    training, holdout = split_ball_compensation_samples(samples)

    assert len(training) == 46
    assert len(holdout) == 10
    assert {item.sample_index for item in training}.isdisjoint({item.sample_index for item in holdout})
    holdout_points = np.asarray([item.target_table_mm for item in holdout], dtype=np.float64)
    assert np.ptp(holdout_points[:, 0]) >= 1800.0
    assert np.ptp(holdout_points[:, 1]) >= 900.0


def test_dense_sampling_grid_is_spread_without_near_duplicates() -> None:
    polygon = np.asarray(
        [[40.0, 40.0], [2500.0, 40.0], [2500.0, 1230.0], [40.0, 1230.0]],
        dtype=np.float32,
    )

    grid = build_engineered_ball_sampling_grid(
        2540.0,
        1270.0,
        57.15,
        cols=8,
        rows=7,
        preferred_polygon_mm=polygon,
        extra_safe_inset_mm=6.0,
    )

    distances = np.linalg.norm(grid[:, None, :] - grid[None, :, :], axis=2)
    distances[distances < 1e-9] = np.inf
    assert grid.shape == (56, 2)
    assert float(np.min(distances)) >= 100.0
    # Sampling order should be locally efficient instead of jumping back to
    # the opposite end of the table after each row.
    travel = np.linalg.norm(np.diff(grid, axis=0), axis=1)
    assert float(np.percentile(travel, 95)) < 700.0
from bas.calibration.camera import CameraCalibration
from bas.calibration.projector import ProjectionCalibration
from bas.calibration.service import CalibrationService
from bas.geometry import TableGeometry
from bas.schemas import Detection, TableModel
from bas.table_boundaries import EdgeInsets
from bas.ui.engineered_ball_compensation_wizard import (
    _pick_ball_candidate,
    _render_target_image,
    _target_guide_radii,
    ball_compensation_path_or_default,
    ball_compensation_path_from_input,
    resolve_ball_sampling_region,
    timestamped_ball_compensation_output_path,
)


def test_ball_sampling_rejects_detector_bbox_fallback_near_target() -> None:
    detection = Detection(
        bbox=(470.0, 370.0, 530.0, 430.0),
        conf=0.92,
        cls_id=2,
        cls_name="sob",
        geometry_quality=0.45,
        geometry_method="bbox",
    )

    candidate, candidate_count, distance_px = _pick_ball_candidate(
        [detection],
        np.asarray([500.0, 400.0], dtype=np.float32),
        expected_radius_px=30.0,
    )

    assert candidate is None
    assert candidate_count == 0
    assert np.isinf(distance_px)


def test_ball_sampling_target_keeps_ball_interior_dark() -> None:
    target = type(
        "TargetEllipse",
        (),
        {
            "center_px": (320.0, 240.0),
            "radius_x_px": 28.0,
            "radius_y_px": 24.0,
            "rotation_deg": 0.0,
        },
    )()

    image = _render_target_image((640, 480), target, 1, 30)
    interior = image[228:253, 306:335]

    assert int(np.max(interior)) <= 20


def test_ball_sampling_target_guide_stays_close_to_physical_ball_edge() -> None:
    target = type("TargetEllipse", (), {"radius_x_px": 13.5, "radius_y_px": 15.3})()

    assert _target_guide_radii(target) == (16, 17)


def test_ball_compensation_holdout_rejects_systematic_bias() -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.asarray([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]]),
        np.asarray([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]]),
        projector_size=(1000, 500),
    )
    service = CalibrationService(
        CameraCalibration(metadata={}),
        projection,
        TableModel(
            width_mm=1000.0,
            height_mm=500.0,
            ball_diameter_mm=57.15,
            inner_polygon_mm=[(0.0, 0.0), (1000.0, 0.0), (1000.0, 500.0), (0.0, 500.0)],
            pockets_mm=[],
        ),
    )
    samples = [
        BallCompensationSample(
            sample_index=index,
            target_table_mm=(float(100 + index * 50), 200.0),
            detected_camera_px=(float(110 + index * 50), 200.0),
            projected_target_px=(0.0, 0.0),
            expected_camera_px=(0.0, 0.0),
            observed_table_mm=(0.0, 0.0),
            delta_table_mm=(0.0, 0.0),
        )
        for index in range(10)
    ]

    report = evaluate_ball_compensation_holdout(samples, service)

    assert report["quality_gate_passed"] is False
    assert any("mean bias" in error for error in report["quality_gate_errors"])


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


def test_build_engineered_ball_sampling_grid_adds_corner_and_edge_coverage() -> None:
    polygon = np.asarray(
        [
            [0.0, 0.0],
            [1000.0, 0.0],
            [1000.0, 500.0],
            [0.0, 500.0],
        ],
        dtype=np.float32,
    )
    grid = build_engineered_ball_sampling_grid(
        1000.0,
        500.0,
        57.15,
        cols=6,
        rows=5,
        preferred_polygon_mm=polygon,
        extra_safe_inset_mm=6.0,
    )

    assert grid.shape == (30, 2)
    assert np.sum(grid[:, 1] <= 24.0) >= 3
    assert np.sum(grid[:, 1] >= 476.0) >= 3
    assert np.sum(grid[:, 0] <= 24.0) >= 3
    assert np.sum(grid[:, 0] >= 976.0) >= 3
    assert np.any((grid[:, 0] <= 24.0) & (grid[:, 1] <= 24.0))
    assert np.any((grid[:, 0] >= 976.0) & (grid[:, 1] <= 24.0))
    assert np.any((grid[:, 0] <= 24.0) & (grid[:, 1] >= 476.0))
    assert np.any((grid[:, 0] >= 976.0) & (grid[:, 1] >= 476.0))


def test_build_engineered_ball_sampling_grid_spreads_edges_on_dense_polygon() -> None:
    top = np.column_stack([np.linspace(0.0, 1000.0, 101), np.zeros(101)])
    right = np.column_stack([np.full(51, 1000.0), np.linspace(10.0, 500.0, 51)])
    bottom = np.column_stack([np.linspace(990.0, 0.0, 100), np.full(100, 500.0)])
    left = np.column_stack([np.zeros(49), np.linspace(490.0, 10.0, 49)])
    polygon = np.vstack([top, right, bottom, left]).astype(np.float32)

    grid = build_engineered_ball_sampling_grid(
        1000.0,
        500.0,
        57.15,
        cols=6,
        rows=5,
        preferred_polygon_mm=polygon,
        extra_safe_inset_mm=0.0,
    )

    assert np.sum(grid[:, 1] <= 5.0) >= 3
    assert np.sum(grid[:, 0] >= 995.0) >= 3
    assert np.sum(grid[:, 1] >= 495.0) >= 3
    assert np.sum(grid[:, 0] <= 5.0) >= 3
    spatial_bins = np.zeros((2, 3), dtype=np.int32)
    for x, y in grid:
        col = min(2, max(0, int(float(x) / 1000.0 * 3.0)))
        row = min(1, max(0, int(float(y) / 500.0 * 2.0)))
        spatial_bins[row, col] += 1
    assert int(np.min(spatial_bins)) >= 3


def test_build_engineered_ball_sampling_grid_keeps_pocket_edge_anchors() -> None:
    polygon = np.asarray(
        [[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]],
        dtype=np.float32,
    )
    pocket_anchors = np.asarray(
        [
            [0.0, 0.0],
            [500.0, 0.0],
            [1000.0, 0.0],
            [1000.0, 500.0],
            [500.0, 500.0],
            [0.0, 500.0],
        ],
        dtype=np.float32,
    )

    grid = build_engineered_ball_sampling_grid(
        1000.0,
        500.0,
        57.15,
        cols=6,
        rows=5,
        preferred_polygon_mm=polygon,
        extra_safe_inset_mm=0.0,
        priority_points_mm=pocket_anchors,
    )
    nearest = np.min(
        np.linalg.norm(pocket_anchors[:, None, :] - grid[None, :, :], axis=2),
        axis=1,
    )

    assert float(np.max(nearest)) <= 5.0


def test_missing_geometry_uses_ball_radius_fallback_instead_of_default_center_polygon() -> None:
    table = TableModel(
        width_mm=2540.0,
        height_mm=1270.0,
        ball_diameter_mm=57.15,
        inner_polygon_mm=[(0.0, 0.0), (2540.0, 0.0), (2540.0, 1270.0), (0.0, 1270.0)],
        pockets_mm=[],
        center_playable_polygon_mm=[
            (0.0, 0.0),
            (2540.0, 0.0),
            (2540.0, 1270.0),
            (0.0, 1270.0),
        ],
    )

    polygon, safe_inset_mm, source = resolve_ball_sampling_region(
        table,
        boundaries_ready=False,
    )
    grid = build_engineered_ball_sampling_grid(
        table.width_mm,
        table.height_mm,
        table.ball_diameter_mm,
        cols=6,
        rows=5,
        preferred_polygon_mm=polygon,
        extra_safe_inset_mm=safe_inset_mm,
    )
    edge_clearance = np.min(
        np.column_stack(
            [
                grid[:, 0],
                table.width_mm - grid[:, 0],
                grid[:, 1],
                table.height_mm - grid[:, 1],
            ]
        ),
        axis=1,
    )

    assert source == "inner_polygon_fallback"
    assert safe_inset_mm >= 0.5 * table.ball_diameter_mm
    assert float(np.min(edge_clearance)) >= 0.5 * table.ball_diameter_mm - 1.0


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

    model = build_ball_compensation_model(
        samples,
        ball_diameter_mm=57.15,
        max_neighbors=8,
        minimum_samples=4,
    )
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
    assert model.quality_report["quality_gate_passed"] is True
    assert model.quality_report["mapping_cross_validation"]["p95_mm"] < 10.29
    assert reloaded.is_valid is True
    assert reloaded.max_neighbors == 4
    assert np.allclose(reloaded.control_points_camera_px, np.asarray([sample.detected_camera_px for sample in samples], dtype=np.float64))
    assert payload["ball_diameter_mm"] == 57.15
    assert len(payload["samples"]) == 4


def test_ball_compensation_context_mismatch_disables_loaded_artifact(tmp_path) -> None:
    path = tmp_path / "ball_context.json"
    controls = np.asarray([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float64)
    BallCompensationModel(
        mode="direct",
        control_points_camera_px=controls,
        target_table_mm=controls,
    ).save_json(
        path,
        extra_data={"calibration_context": {"frame_rotation_degrees": 0, "camera_coordinate_domain": "raw"}},
    )

    loaded = BallCompensationModel.load_json(
        path,
        expected_context={"frame_rotation_degrees": 180, "camera_coordinate_domain": "raw"},
    )

    assert loaded.is_valid is False
    assert "frame_rotation_degrees" in loaded.compatibility_errors


def test_ball_compensation_requires_twenty_samples_by_default() -> None:
    sample = BallCompensationSample(
        sample_index=0,
        target_table_mm=(100.0, 100.0),
        detected_camera_px=(50.0, 50.0),
        projected_target_px=(50.0, 50.0),
        expected_camera_px=(50.0, 50.0),
        observed_table_mm=(100.0, 100.0),
        delta_table_mm=(0.0, 0.0),
        geometry_quality=0.9,
        geometry_method="segmentation_ellipse",
    )

    with np.testing.assert_raises_regex(ValueError, "20"):
        build_ball_compensation_model([sample] * 19, ball_diameter_mm=57.15)


def test_ball_compensation_rejects_unpredictable_cross_validated_mapping() -> None:
    rng = np.random.default_rng(20260801)
    camera_points = np.column_stack(
        [
            np.tile(np.linspace(160.0, 1760.0, 6), 5),
            np.repeat(np.linspace(140.0, 940.0, 5), 6),
        ]
    )
    random_targets = rng.uniform([0.0, 0.0], [2540.0, 1270.0], size=(30, 2))
    samples = [
        BallCompensationSample(
            sample_index=index,
            target_table_mm=tuple(random_targets[index]),
            detected_camera_px=tuple(camera_points[index]),
            projected_target_px=(0.0, 0.0),
            expected_camera_px=tuple(camera_points[index]),
            observed_table_mm=tuple(random_targets[index]),
            delta_table_mm=(0.0, 0.0),
            detected_radius_px=22.0,
            detection_confidence=0.92,
            stability_spread_px=0.3,
            geometry_quality=0.9,
            geometry_method="segmentation_ellipse",
        )
        for index in range(30)
    ]

    with np.testing.assert_raises_regex(ValueError, "cross-validation"):
        build_ball_compensation_model(samples, ball_diameter_mm=57.15)


def test_ball_compensation_rejects_samples_that_cover_too_little_of_table() -> None:
    camera_points = np.column_stack(
        [
            np.tile(np.linspace(100.0, 500.0, 5), 4),
            np.repeat(np.linspace(100.0, 400.0, 4), 5),
        ]
    )
    target_points = camera_points * np.asarray([1.25, 1.1]) + np.asarray([120.0, 80.0])
    samples = [
        BallCompensationSample(
            sample_index=index,
            target_table_mm=tuple(target_points[index]),
            detected_camera_px=tuple(camera_points[index]),
            projected_target_px=(0.0, 0.0),
            expected_camera_px=tuple(camera_points[index]),
            observed_table_mm=tuple(target_points[index]),
            delta_table_mm=(0.0, 0.0),
            detected_radius_px=22.0,
            detection_confidence=0.95,
            stability_spread_px=0.2,
            geometry_quality=0.95,
            geometry_method="segmentation_ellipse",
        )
        for index in range(20)
    ]

    with np.testing.assert_raises_regex(ValueError, "target width coverage"):
        build_ball_compensation_model(
            samples,
            ball_diameter_mm=57.15,
            table_width_mm=2540.0,
            table_height_mm=1270.0,
        )


def test_ball_compensation_model_rejects_explicit_failed_quality_gate() -> None:
    controls = np.asarray([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float64)
    model = BallCompensationModel(
        mode="direct",
        control_points_camera_px=controls,
        target_table_mm=controls.copy(),
        quality_report={"quality_gate_passed": False},
    )

    assert model.is_valid is False


def test_ball_compensation_downweights_bbox_geometry() -> None:
    samples = []
    for index in range(20):
        samples.append(
            BallCompensationSample(
                sample_index=index,
                target_table_mm=(float(index * 10), float((index % 5) * 20)),
                detected_camera_px=(float(index * 4), float((index % 5) * 8)),
                projected_target_px=(0.0, 0.0),
                expected_camera_px=(0.0, 0.0),
                observed_table_mm=(0.0, 0.0),
                delta_table_mm=(1.0, -1.0),
                detection_confidence=0.95,
                stability_spread_px=0.5,
                geometry_quality=0.9,
                geometry_method="bbox" if index == 0 else "segmentation_ellipse",
            )
        )

    model = build_ball_compensation_model(samples, ball_diameter_mm=57.15)

    assert model.sample_weights[0] < model.sample_weights[1] * 0.4


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
