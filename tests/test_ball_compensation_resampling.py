from __future__ import annotations

import numpy as np

from bas.calibration.ball_compensation_resampling import (
    aggregate_training_repeats,
    assess_holdout_repeatability,
    plan_failed_holdout_recovery,
)
from bas.calibration.ball_compensation_sampling import BallCompensationSample


def _sample(index: int, target: tuple[float, float], observed: tuple[float, float]) -> BallCompensationSample:
    return BallCompensationSample(
        sample_index=index,
        target_table_mm=target,
        detected_camera_px=observed,
        projected_target_px=target,
        expected_camera_px=target,
        observed_table_mm=observed,
        delta_table_mm=(target[0] - observed[0], target[1] - observed[1]),
        detected_radius_px=20.0,
        detection_confidence=0.9,
        stability_spread_px=0.3,
        geometry_quality=0.9,
        geometry_method="appearance_ellipse",
        detector_version="test",
    )


def test_training_repeats_are_median_aggregated_without_changing_target_index() -> None:
    repeats = [
        _sample(7, (100.0, 200.0), (98.0, 201.0)),
        _sample(7, (100.0, 200.0), (99.0, 200.0)),
        _sample(7, (100.0, 200.0), (140.0, 180.0)),
    ]

    aggregated = aggregate_training_repeats(repeats, minimum_repeats=3)

    assert aggregated.sample_index == 7
    assert aggregated.target_table_mm == (100.0, 200.0)
    assert aggregated.observed_table_mm == (99.0, 200.0)
    assert aggregated.detected_camera_px == (99.0, 200.0)
    assert aggregated.delta_table_mm == (1.0, 0.0)


def test_wizard_training_collector_requests_three_stable_batches() -> None:
    from bas.ui.engineered_ball_compensation_wizard import EngineeredBallCompensationWizardDialog

    samples = [
        _sample(3, (100.0, 200.0), (99.0, 200.0)),
        _sample(3, (100.0, 200.0), (100.0, 200.0)),
        _sample(3, (100.0, 200.0), (101.0, 200.0)),
    ]
    calls: list[int] = []

    class FakeWizard:
        def _append_log(self, _message: str) -> None:
            pass

        def _collect_sample_for_target(self, *_args, **_kwargs):
            calls.append(len(calls))
            return samples[len(calls) - 1]

    aggregated = EngineeredBallCompensationWizardDialog._collect_training_repeats(
        FakeWizard(),
        None,
        None,
        None,
        None,
        3,
        56,
        np.asarray([100.0, 200.0], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
    )

    assert len(calls) == 3
    assert aggregated is not None
    assert aggregated.observed_table_mm == (100.0, 200.0)


def test_holdout_repeatability_flags_only_locations_over_limit_and_ignores_target_offset() -> None:
    stable_but_offset = [
        _sample(i, (100.0, 200.0), point)
        for i, point in enumerate(((110.0, 210.0), (110.4, 210.2), (109.8, 209.9)))
    ]
    unstable = [
        _sample(10 + i, (300.0, 400.0), point)
        for i, point in enumerate(((300.0, 400.0), (304.5, 400.0), (299.8, 400.2)))
    ]

    assessment = assess_holdout_repeatability(
        [stable_but_offset, unstable],
        maximum_deviation_mm=2.0,
        minimum_repeats=3,
    )

    assert assessment.unstable_location_indices == (1,)
    assert assessment.locations[0].maximum_deviation_mm < 1.0
    assert assessment.locations[0].passed is True
    assert assessment.locations[1].maximum_deviation_mm > 4.0
    assert assessment.locations[1].passed is False


def test_failed_holdout_recovery_marks_failed_locations_and_expands_nearest_training_ring() -> None:
    training = [
        _sample(row * 4 + col, (float(col * 100), float(row * 100)), (float(col * 100), float(row * 100)))
        for row in range(4)
        for col in range(4)
    ]
    holdout_report = {
        "samples": [
            {"target_table_mm": [100.0, 100.0], "error_mm": 3.2},
            {"target_table_mm": [300.0, 300.0], "error_mm": 0.8},
        ]
    }

    recovery = plan_failed_holdout_recovery(
        training,
        holdout_report,
        failure_error_mm=2.0,
        surrounding_neighbor_count=8,
    )

    assert recovery.failed_holdout_location_indices == (0,)
    assert recovery.failed_targets_table_mm == ((100.0, 100.0),)
    assert len(recovery.training_sample_indices) == 9
    assert 5 in recovery.training_sample_indices
    selected_points = np.asarray(
        [training[index].target_table_mm for index in recovery.training_sample_indices],
        dtype=np.float64,
    )
    assert np.min(np.linalg.norm(selected_points - np.asarray([100.0, 100.0]), axis=1)) == 0.0


def test_formal_failure_region_uses_one_mm_scale_not_repeatability_limit() -> None:
    training = [_sample(index, (float(index * 100), 0.0), (float(index * 100), 0.0)) for index in range(12)]
    report = {"samples": [{"target_table_mm": [500.0, 0.0], "error_mm": 1.5}]}

    recovery = plan_failed_holdout_recovery(training, report, failure_error_mm=1.0)

    assert recovery.failed_holdout_location_indices == (0,)
    assert recovery.training_sample_indices


def test_failed_holdout_recovery_never_promotes_holdout_samples_into_training() -> None:
    training = [_sample(index, (float(index * 100), 0.0), (float(index * 100), 0.0)) for index in range(12)]
    holdout_report = {
        "samples": [
            {"sample_index": 999, "target_table_mm": [450.0, 0.0], "error_mm": 4.0},
        ]
    }

    recovery = plan_failed_holdout_recovery(training, holdout_report)

    assert 999 not in recovery.training_sample_indices
    assert set(recovery.training_sample_indices).issubset({sample.sample_index for sample in training})
