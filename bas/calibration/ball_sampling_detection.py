from __future__ import annotations

import numpy as np

from ..schemas import Detection


MIN_BBOX_SAMPLING_CONFIDENCE = 0.35
MIN_BBOX_SAMPLING_ASPECT_RATIO = 0.70
MIN_BBOX_SAMPLING_RADIUS_RATIO = 0.65
MAX_BBOX_SAMPLING_RADIUS_RATIO = 1.40
MAX_BBOX_TARGET_DISTANCE_RADII = 1.50
MAX_REFINED_TARGET_DISTANCE_RADII = 1.75
MIN_TARGET_DISTANCE_LIMIT_PX = 18.0
MAX_SAMPLE_DELTA_BALL_DIAMETERS = 1.0
BALL_SAMPLING_DETECTION_VERSION = "target_locked_bbox_recovery_v3"


def ball_sampling_geometry_is_usable(
    detection: Detection,
    expected_cam: np.ndarray,
    expected_radius_px: float,
) -> bool:
    """Accept precise contours, or a tightly constrained calibration-only bbox.

    A YOLO box is not generally a trustworthy ball center.  During engineered
    sampling, however, the expected center and physical radius are known.  A
    high-confidence, nearly square box that agrees with both priors is safe to
    feed into the existing multi-frame median and is retained as low-weight
    ``bbox`` geometry by the compensation fitter.
    """

    method = str(detection.geometry_method or "").strip().lower()
    if not method.startswith("bbox"):
        return float(detection.geometry_quality) >= 0.40

    if float(detection.conf) < MIN_BBOX_SAMPLING_CONFIDENCE:
        return False
    x1, y1, x2, y2 = [float(value) for value in detection.bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if min(width, height) < 4.0:
        return False
    aspect = min(width, height) / max(width, height)
    if aspect < MIN_BBOX_SAMPLING_ASPECT_RATIO:
        return False

    expected_radius = max(2.0, float(expected_radius_px))
    radius_ratio = float(detection.radius_px) / expected_radius
    if not (MIN_BBOX_SAMPLING_RADIUS_RATIO <= radius_ratio <= MAX_BBOX_SAMPLING_RADIUS_RATIO):
        return False

    expected = np.asarray(expected_cam, dtype=np.float32).reshape((2,))
    center = np.asarray(detection.center, dtype=np.float32).reshape((2,))
    distance = float(np.linalg.norm(center - expected))
    return distance <= MAX_BBOX_TARGET_DISTANCE_RADII * expected_radius


def ball_sampling_target_distance_limit_px(expected_radius_px: float) -> float:
    return max(MIN_TARGET_DISTANCE_LIMIT_PX, MAX_REFINED_TARGET_DISTANCE_RADII * max(2.0, float(expected_radius_px)))


def ball_sampling_delta_is_plausible(delta_table_mm: np.ndarray, ball_diameter_mm: float) -> bool:
    delta = np.asarray(delta_table_mm, dtype=np.float64).reshape((-1,))
    if delta.size < 2 or not np.isfinite(delta[:2]).all():
        return False
    maximum = MAX_SAMPLE_DELTA_BALL_DIAMETERS * max(1.0, float(ball_diameter_mm))
    return float(np.linalg.norm(delta[:2])) <= maximum
