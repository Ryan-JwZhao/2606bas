from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from bas.calibration import BallCompensationModel
from bas.calibration.ball_compensation_sampling import (
    BallCompensationValidationError,
    BallCompensationSample,
    aggregate_ball_compensation_holdout_repeats,
    ball_holdout_geometry_is_formal,
    build_ball_compensation_model,
    build_engineered_ball_sampling_grid,
    evaluate_ball_compensation_holdout,
    fit_and_validate_ball_compensation,
    select_ball_compensation_holdout_targets,
    split_ball_compensation_samples,
    update_calibration_table_boundaries_from_geometry_frame,
)
from bas.calibration.ball_sampling_detection import ball_sampling_delta_is_plausible
from bas.calibration.quality_standards import ball_holdout_quality_errors


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


def test_sample_split_rejects_total_below_training_plus_minimum_holdout() -> None:
    samples = [
        _sample_from_identical_point(index, np.asarray([float(index % 7), float(index // 7)]))
        for index in range(27)
    ]

    with np.testing.assert_raises_regex(ValueError, "(?i)at least 28.*20 training.*8 holdout"):
        split_ball_compensation_samples(samples)


def test_sample_split_keeps_declared_minimum_holdout_at_exact_boundary() -> None:
    samples = [
        _sample_from_identical_point(index, np.asarray([float(index % 7), float(index // 7)]))
        for index in range(28)
    ]

    training, holdout = split_ball_compensation_samples(samples)

    assert len(training) == 20
    assert len(holdout) == 8


def test_ball_holdout_rejects_bad_median_even_when_p95_and_directional_bias_pass() -> None:
    errors = ball_holdout_quality_errors(
        sample_count=8,
        median_mm=1.5,
        p95_mm=1.5,
        mean_bias_mm=0.0,
    )

    assert errors == ["holdout median 1.50 mm is not below 1.00 mm"]


def test_wizard_fit_seam_builds_and_activates_holdout_validated_model() -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.asarray([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]]),
        np.asarray([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]]),
        projector_size=(1000, 500),
    )
    projection.table_polygon_cam = np.asarray([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]])
    projection.table_polygon_proj = projection.table_polygon_cam.copy()
    service = CalibrationService(
        CameraCalibration(metadata={}),
        projection,
        TableModel(1000.0, 500.0, 57.15, [(0, 0), (1000, 0), (1000, 500), (0, 500)], []),
    )
    points = build_engineered_ball_sampling_grid(1000.0, 500.0, 57.15, cols=8, rows=7)
    samples = [_sample_from_identical_point(index, point) for index, point in enumerate(points)]

    holdout = [_sample_from_identical_point(1000 + index, point) for index, point in enumerate(points[::6])]

    validated = fit_and_validate_ball_compensation(
        samples,
        service,
        holdout_samples=holdout,
        calibration_context={"camera_coordinate_domain": "raw"},
    )

    assert service.ball_compensation_model is validated.model
    assert len(validated.training_samples) == 56
    assert len(validated.holdout_samples) == 10
    assert validated.holdout_report["quality_gate_passed"] is True
    assert validated.model.calibration_context["camera_coordinate_domain"] == "raw"
    assert validated.model.quality_report["quality_gate_passed"] is True
    assert validated.model.quality_report["holdout_validation"] == validated.holdout_report


def test_wizard_fit_seam_restores_previous_model_after_holdout_failure() -> None:
    projection = ProjectionCalibration.fit_from_correspondences(
        np.asarray([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]]),
        np.asarray([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]]),
        projector_size=(1000, 500),
    )
    projection.table_polygon_cam = np.asarray([[0.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]])
    projection.table_polygon_proj = projection.table_polygon_cam.copy()
    service = CalibrationService(
        CameraCalibration(metadata={}),
        projection,
        TableModel(1000.0, 500.0, 57.15, [(0, 0), (1000, 0), (1000, 500), (0, 500)], []),
    )
    previous = service.ball_compensation_model
    points = build_engineered_ball_sampling_grid(1000.0, 500.0, 57.15, cols=8, rows=7)
    samples = [_sample_from_identical_point(index, point) for index, point in enumerate(points)]
    holdout = [_sample_from_identical_point(1000 + index, point) for index, point in enumerate(points[::6])]
    for sample in holdout:
        sample.detected_camera_px = (sample.detected_camera_px[0] + 40.0, sample.detected_camera_px[1])

    with np.testing.assert_raises(BallCompensationValidationError):
        fit_and_validate_ball_compensation(samples, service, holdout_samples=holdout)

    assert service.ball_compensation_model is previous


def test_independent_holdout_targets_keep_every_first_pass_sample_for_training() -> None:
    samples = [
        _sample_from_identical_point(index, np.asarray([float(index % 8) * 300.0, float(index // 8) * 180.0]))
        for index in range(56)
    ]

    targets = select_ball_compensation_holdout_targets(samples, count=10)

    assert len(targets) == 10
    assert len(samples) == 56
    assert {sample.sample_index for sample in targets}.issubset({sample.sample_index for sample in samples})
    points = np.asarray([sample.target_table_mm for sample in targets], dtype=np.float64)
    assert np.ptp(points[:, 0]) >= 1800.0
    assert np.ptp(points[:, 1]) >= 900.0


def test_independent_holdout_targets_can_exclude_known_bbox_only_locations() -> None:
    samples = [
        _sample_from_identical_point(index, np.asarray([float(index % 8) * 300.0, float(index // 8) * 180.0]))
        for index in range(56)
    ]
    for index in (0, 7, 48, 55):
        samples[index].geometry_method = "bbox"
        samples[index].geometry_quality = 0.45

    targets = select_ball_compensation_holdout_targets(
        samples,
        count=10,
        require_formal_geometry=True,
        minimum_geometry_quality=0.70,
    )

    assert len(targets) == 10
    assert all(ball_holdout_geometry_is_formal(sample.geometry_method) for sample in targets)
    assert all(sample.geometry_quality >= 0.70 for sample in targets)
    points = np.asarray([sample.target_table_mm for sample in targets], dtype=np.float64)
    assert np.ptp(points[:, 0]) >= 1500.0
    assert np.ptp(points[:, 1]) >= 700.0


def test_formal_holdout_rejects_bbox_geometry() -> None:
    assert ball_holdout_geometry_is_formal("segmentation_ellipse") is True
    assert ball_holdout_geometry_is_formal("appearance_ellipse") is True
    assert ball_holdout_geometry_is_formal("bbox") is False
    assert ball_holdout_geometry_is_formal("bbox:size_outlier") is False


def test_holdout_repeat_aggregation_uses_median_and_reports_repeatability() -> None:
    group = []
    for index, center_x in enumerate((99.5, 100.5, 140.0)):
        sample = _sample_from_identical_point(index, np.asarray([100.0, 200.0]))
        sample.detected_camera_px = (center_x, 200.0)
        sample.observed_table_mm = (center_x, 200.0)
        sample.delta_table_mm = (100.0 - center_x, 0.0)
        group.append(sample)

    aggregated, repeatability = aggregate_ball_compensation_holdout_repeats([group])

    assert len(aggregated) == 1
    assert aggregated[0].detected_camera_px == (100.5, 200.0)
    assert aggregated[0].geometry_method == "segmentation_ellipse"
    assert repeatability["repeat_count"] == 3
    assert repeatability["location_count"] == 1
    assert repeatability["error_mm"]["max"] == 39.5


def _sample_from_identical_point(index: int, point: np.ndarray) -> BallCompensationSample:
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
        stability_spread_px=0.2,
        geometry_quality=0.95,
        geometry_method="segmentation_ellipse",
    )


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
    EngineeredBallCompensationWizardDialog,
    _ball_compensation_completion_summary,
    _pick_ball_candidate,
    _render_target_image,
    _stable_measurement,
    _target_guide_radii,
    ball_compensation_path_or_default,
    ball_compensation_path_from_input,
    resolve_ball_sampling_region,
    timestamped_ball_compensation_output_path,
)


def test_stable_sample_tolerates_isolated_detector_dropouts(monkeypatch) -> None:
    from bas.ui import engineered_ball_compensation_wizard as wizard

    detection = Detection(
        bbox=(80.0, 80.0, 120.0, 120.0),
        conf=0.93,
        cls_id=0,
        cls_name="cue",
        refined_center_px=(100.0, 100.0),
        refined_radius_px=20.0,
        geometry_quality=0.9,
        geometry_method="appearance_ellipse",
    )

    class AlternatingDetector:
        version = "test"

        def __init__(self) -> None:
            self.calls = 0

        def detect(self, _frame, *, mask_polygon=None):
            del mask_polygon
            self.calls += 1
            return [] if self.calls % 6 == 0 else [detection]

    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def perf_counter(self) -> float:
            self.now += 0.02
            return self.now

    clock = FakeClock()
    monkeypatch.setattr(wizard.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(wizard.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(wizard, "ENGINEERED_SAMPLE_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(wizard, "ENGINEERED_SAMPLE_SETTLE_DELAY_SECONDS", 0.3)

    calibration = SimpleNamespace(
        table=SimpleNamespace(ball_diameter_mm=40.0),
        projection=SimpleNamespace(table_polygon_cam=None),
        table_mm_to_camera_px=lambda points: np.asarray(points, dtype=np.float32),
        camera_px_to_table_mm=lambda points: np.asarray(points, dtype=np.float32),
    )
    dialog = SimpleNamespace(
        _abort_requested=False,
        summary=SimpleNamespace(setText=lambda _text: None),
        _set_preview_image=lambda _image: None,
        _pump_ui=lambda: None,
        _append_log=lambda _text: None,
    )
    capture = SimpleNamespace(
        read=lambda: SimpleNamespace(image=np.zeros((240, 320, 3), dtype=np.uint8))
    )

    sample = EngineeredBallCompensationWizardDialog._wait_for_stable_sample(
        dialog,
        capture,
        AlternatingDetector(),
        calibration,
        None,
        1,
        56,
        np.asarray([100.0, 100.0], dtype=np.float32),
        np.asarray([100.0, 100.0], dtype=np.float32),
        np.asarray([100.0, 100.0], dtype=np.float32),
    )

    assert sample is not None


def test_stable_sample_completes_at_unchanged_target_with_bbox_recovery(monkeypatch) -> None:
    from bas.ui import engineered_ball_compensation_wizard as wizard

    detection = Detection(
        bbox=(70.0, 70.0, 130.0, 130.0),
        conf=0.94,
        cls_id=0,
        cls_name="cue",
        geometry_quality=0.45,
        geometry_method="bbox",
    )

    class Detector:
        version = "bbox-recovery-test"

        def detect(self, _frame, *, mask_polygon=None):
            del mask_polygon
            return [detection]

    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def perf_counter(self) -> float:
            self.now += 0.02
            return self.now

    clock = FakeClock()
    monkeypatch.setattr(wizard.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(wizard.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(wizard, "ENGINEERED_SAMPLE_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(wizard, "ENGINEERED_SAMPLE_SETTLE_DELAY_SECONDS", 0.12)

    calibration = SimpleNamespace(
        table=SimpleNamespace(ball_diameter_mm=60.0),
        projection=SimpleNamespace(table_polygon_cam=None),
        table_mm_to_camera_px=lambda points: np.asarray(points, dtype=np.float32),
        camera_px_to_table_mm=lambda points: np.asarray(points, dtype=np.float32),
    )
    dialog = SimpleNamespace(
        _abort_requested=False,
        summary=SimpleNamespace(setText=lambda _text: None),
        _set_preview_image=lambda _image: None,
        _pump_ui=lambda: None,
        _append_log=lambda _text: None,
    )
    capture = SimpleNamespace(read=lambda: SimpleNamespace(image=np.zeros((240, 320, 3), dtype=np.uint8)))
    fixed_target = np.asarray([100.0, 100.0], dtype=np.float32)
    projected_target = np.asarray([240.0, 180.0], dtype=np.float32)

    sample = EngineeredBallCompensationWizardDialog._wait_for_stable_sample(
        dialog,
        capture,
        Detector(),
        calibration,
        None,
        0,
        56,
        fixed_target,
        projected_target,
        fixed_target,
    )

    assert sample is not None
    assert sample.target_table_mm == (100.0, 100.0)
    assert sample.projected_target_px == (240.0, 180.0)
    assert sample.detected_camera_px == (100.0, 100.0)
    assert sample.geometry_method == "bbox"

    formal_sample = EngineeredBallCompensationWizardDialog._wait_for_stable_sample(
        dialog,
        capture,
        Detector(),
        calibration,
        None,
        0,
        30,
        fixed_target,
        projected_target,
        fixed_target,
        formal_geometry_required=True,
    )

    assert formal_sample is None


def test_stable_sample_resets_after_sustained_detector_dropout() -> None:
    from bas.ui import engineered_ball_compensation_wizard as wizard

    grace = wizard.ENGINEERED_SAMPLE_DROPOUT_GRACE_SECONDS

    assert wizard._candidate_dropout_expired(10.0, 10.0 + grace - 0.01) is False
    assert wizard._candidate_dropout_expired(10.0, 10.0 + grace + 0.01) is True


def test_bbox_stability_ignores_two_isolated_box_center_outliers() -> None:
    centers = (
        (100.0, 100.0),
        (101.0, 100.0),
        (99.0, 100.0),
        (100.0, 101.0),
        (100.0, 99.0),
        (107.0, 100.0),
        (93.0, 100.0),
    )
    history = [
        (np.asarray(center, dtype=np.float32), 20.0, 0.92, 0.45, "bbox")
        for center in centers
    ]

    stable = _stable_measurement(history)

    assert stable is not None
    assert np.linalg.norm(stable[0] - np.asarray([100.0, 100.0], dtype=np.float32)) < 0.1
    assert stable[3] <= 1.0


def test_ball_sampling_accepts_stable_size_matched_bbox_at_fixed_target() -> None:
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

    assert candidate is detection
    assert candidate_count == 1
    assert distance_px == 0.0


def test_ball_sampling_still_rejects_unsafe_bbox_fallbacks() -> None:
    expected = np.asarray([500.0, 400.0], dtype=np.float32)
    unsafe = [
        Detection((470.0, 370.0, 530.0, 430.0), 0.20, 2, "sob", geometry_method="bbox"),
        Detection((450.0, 385.0, 550.0, 415.0), 0.95, 2, "sob", geometry_method="bbox"),
        Detection((430.0, 330.0, 570.0, 470.0), 0.95, 2, "sob", geometry_method="bbox"),
        Detection((540.0, 410.0, 600.0, 470.0), 0.95, 2, "sob", geometry_method="bbox"),
    ]

    for detection in unsafe:
        candidate, candidate_count, distance_px = _pick_ball_candidate(
            [detection],
            expected,
            expected_radius_px=30.0,
        )
        assert candidate is None
        assert candidate_count == 0
        assert np.isinf(distance_px)


def test_ball_sampling_rejects_previous_point_ellipse_after_target_switch() -> None:
    previous_point_ball = Detection(
        bbox=(470.0, 370.0, 530.0, 430.0),
        conf=0.94,
        cls_id=2,
        cls_name="sob",
        refined_center_px=(500.0, 400.0),
        refined_radius_px=30.0,
        geometry_quality=0.90,
        geometry_method="appearance_ellipse",
    )

    candidate, candidate_count, distance_px = _pick_ball_candidate(
        [previous_point_ball],
        np.asarray([620.0, 400.0], dtype=np.float32),
        expected_radius_px=30.0,
    )

    assert candidate is None
    assert candidate_count == 1
    assert distance_px == 120.0


def test_ball_sampling_rejects_physically_impossible_cross_point_delta() -> None:
    assert ball_sampling_delta_is_plausible(np.asarray([24.0, -12.0]), 57.15) is True
    assert ball_sampling_delta_is_plausible(np.asarray([178.3, 23.1]), 57.15) is False


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


def test_ball_compensation_completion_summary_exposes_validation_metrics() -> None:
    summary = _ball_compensation_completion_summary(
        accepted_count=56,
        total_count=56,
        training_count=46,
        holdout_count=10,
        training_report={
            "mapping_cross_validation": {
                "model_kind": "homography_rbf",
                "p95_mm": 3.2,
                "maximum_p95_mm": 10.29,
            },
            "delta_norm_mm": {"p95": 7.5},
        },
        holdout_report={
            "error_mm": {"p95": 4.1},
            "mean_vector_mm": [1.2, -0.9],
            "maximum_p95_mm": 6.86,
            "maximum_mean_bias_mm": 2.86,
            "quality_gate_passed": True,
        },
        save_path="calibration.json",
    )

    assert "训练=46 独立Holdout=10" in summary
    assert "homography_rbf" in summary
    assert "Holdout P95=4.10/6.86 mm" in summary
    assert "平均偏差=1.50/2.86 mm" in summary
    assert "验收=通过" in summary


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


def test_ball_compensation_holdout_rejects_visible_error_even_when_bias_cancels() -> None:
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
    samples = []
    for index in range(10):
        target_x = float(100 + index * 70)
        error_x = 4.0 if index % 2 == 0 else -4.0
        samples.append(
            BallCompensationSample(
                sample_index=index,
                target_table_mm=(target_x, 200.0),
                detected_camera_px=(target_x + error_x, 200.0),
                projected_target_px=(0.0, 0.0),
                expected_camera_px=(0.0, 0.0),
                observed_table_mm=(0.0, 0.0),
                delta_table_mm=(0.0, 0.0),
            )
        )

    report = evaluate_ball_compensation_holdout(samples, service)

    assert report["maximum_p95_mm"] == 2.0
    assert report["quality_gate_passed"] is False
    assert any("holdout P95" in error for error in report["quality_gate_errors"])


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


def test_loaded_ball_compensation_rechecks_embedded_holdout_with_current_standard(tmp_path: Path) -> None:
    controls = np.asarray([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float64)
    model = BallCompensationModel(
        mode="direct",
        control_points_camera_px=controls,
        target_table_mm=controls.copy(),
        quality_report={"quality_gate_passed": True},
    )
    path = tmp_path / "loose_holdout.json"
    model.save_json(
        path,
        extra_data={
            "holdout_quality_report": {
                "sample_count": 8,
                "error_mm": {"p95": 4.0},
                "mean_vector_mm": [0.0, 0.0],
                "quality_gate_passed": True,
            }
        },
    )

    loaded = BallCompensationModel.load_json(path)

    assert loaded.is_valid is False
    assert loaded.quality_report["quality_gate_passed"] is False
    assert any("holdout P95" in error for error in loaded.quality_report["quality_gate_errors"])


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
