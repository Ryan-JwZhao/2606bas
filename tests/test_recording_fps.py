from __future__ import annotations

from bas.recording_fps import RecordingFpsEstimator


def test_recording_fps_estimator_waits_for_ready_window() -> None:
    estimator = RecordingFpsEstimator(min_samples=6, min_span_ns=250_000_000)
    for idx in range(5):
        estimator.observe(1_000_000_000 + idx * 33_333_333)

    assert estimator.estimate(require_ready=True) is None
    assert estimator.estimate(require_ready=False) is not None


def test_recording_fps_estimator_estimates_actual_frame_rate_from_timestamps() -> None:
    estimator = RecordingFpsEstimator(min_samples=6, min_span_ns=250_000_000)
    base = 10_000_000_000
    for idx in range(12):
        estimator.observe(base + idx * 33_333_333)

    fps = estimator.estimate(require_ready=True)

    assert fps is not None
    assert 29.5 <= fps <= 30.5


def test_recording_fps_estimator_resets_on_timestamp_regression() -> None:
    estimator = RecordingFpsEstimator(min_samples=3, min_span_ns=1)
    estimator.observe(1_000)
    estimator.observe(2_000)
    estimator.observe(500)
    estimator.observe(1_500)

    fps = estimator.estimate(require_ready=False)

    assert fps is not None
    assert fps == 1_000_000_000.0 / 1_000.0
