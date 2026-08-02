from __future__ import annotations

from .ball_compensation import BallCompensationModel
from .audit import CalibrationAudit, load_calibration_audit_summaries, start_calibration_audit
from .ball_compensation_sampling import (
    BallCompensationSample,
    BallCompensationValidationError,
    MIN_BALL_COMPENSATION_ACCEPTED_SAMPLES,
    MIN_BALL_COMPENSATION_HOLDOUT_SAMPLES,
    MIN_BALL_COMPENSATION_TRAINING_SAMPLES,
    ValidatedBallCompensation,
    build_ball_compensation_model,
    build_engineered_ball_sampling_grid,
    evaluate_ball_compensation_holdout,
    fit_and_validate_ball_compensation,
    split_ball_compensation_samples,
    update_calibration_table_boundaries_from_geometry_frame,
)
from .camera import CameraCalibration
from .geometry import CalibrationUnavailableError, IndependentGeometry, ProjectedEllipse
from .linked import (
    LinkedCalibrationObservation,
    LinkedCalibrationPattern,
    LinkedCalibrationResult,
    LinkedPatternCaptureResult,
    build_linked_patterns,
    collect_linked_pattern_observation,
    linked_calibration_runtime_summary,
    linked_pattern_requires_retry,
    match_linked_pattern_observation,
    projection_output_summary,
    solve_linked_projection_calibration,
)
from .projector import ProjectionCalibration
from .service import (
    CalibrationService,
    create_calibration_service,
    create_setting_aware_calibration_service,
)
from .verification import format_holdout_report, verify_holdout_file, verify_holdout_samples

__all__ = [
    "BallCompensationModel",
    "CalibrationAudit",
    "load_calibration_audit_summaries",
    "start_calibration_audit",
    "BallCompensationSample",
    "BallCompensationValidationError",
    "MIN_BALL_COMPENSATION_ACCEPTED_SAMPLES",
    "MIN_BALL_COMPENSATION_HOLDOUT_SAMPLES",
    "MIN_BALL_COMPENSATION_TRAINING_SAMPLES",
    "ValidatedBallCompensation",
    "CameraCalibration",
    "IndependentGeometry",
    "ProjectedEllipse",
    "CalibrationUnavailableError",
    "ProjectionCalibration",
    "CalibrationService",
    "create_calibration_service",
    "create_setting_aware_calibration_service",
    "build_ball_compensation_model",
    "build_engineered_ball_sampling_grid",
    "evaluate_ball_compensation_holdout",
    "fit_and_validate_ball_compensation",
    "split_ball_compensation_samples",
    "update_calibration_table_boundaries_from_geometry_frame",
    "LinkedCalibrationPattern",
    "LinkedCalibrationObservation",
    "LinkedCalibrationResult",
    "LinkedPatternCaptureResult",
    "build_linked_patterns",
    "collect_linked_pattern_observation",
    "linked_calibration_runtime_summary",
    "linked_pattern_requires_retry",
    "match_linked_pattern_observation",
    "solve_linked_projection_calibration",
    "projection_output_summary",
    "format_holdout_report",
    "verify_holdout_file",
    "verify_holdout_samples",
]
