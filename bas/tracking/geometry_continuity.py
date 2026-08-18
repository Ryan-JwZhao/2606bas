from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..schemas import Detection


FORMAL_GEOMETRY_METHODS = ("segmentation_ellipse", "appearance_ellipse")
MAX_FALLBACK_CENTER_OFFSET_RADII = 0.5
MIN_FALLBACK_CENTER_OFFSET_PX = 2.0
MIN_FALLBACK_RADIUS_RATIO = 0.5
MAX_FALLBACK_RADIUS_RATIO = 1.5
FORMAL_TO_BBOX_FALLBACK_ALPHA = 0.10
FINE_CENTER_DISPLACEMENT_RADIUS_RATIO = 0.05
FINE_CENTER_MEASUREMENT_ALPHA = 0.30


@dataclass(frozen=True)
class ContinuousGeometryMeasurement:
    center_px: np.ndarray
    radius_px: float
    fallback_from_formal: bool = False


@dataclass
class BallGeometryContinuity:
    """Keep a ball's physical center definition across geometry fallbacks.

    A refined ellipse center is commonly offset from the detector box midpoint.
    When refinement is briefly unavailable, the box still follows real motion;
    applying the last reliable center-to-box offset avoids changing the meaning
    of "ball center" merely because the estimator method changed.
    """

    center_offset_px: np.ndarray = field(
        default_factory=lambda: np.zeros((2,), dtype=np.float32)
    )
    radius_ratio: float = 1.0
    has_formal_geometry: bool = False

    def measure(self, detection: Detection) -> ContinuousGeometryMeasurement:
        bbox_center = _bbox_center(detection)
        bbox_radius = _bbox_radius(detection)
        measured_center = np.asarray(detection.center, dtype=np.float32).reshape((2,))
        measured_radius = max(2.0, float(detection.radius_px))
        method = str(detection.geometry_method or "").strip().lower()

        if method.startswith(FORMAL_GEOMETRY_METHODS):
            offset = measured_center - bbox_center
            limit = max(
                MIN_FALLBACK_CENTER_OFFSET_PX,
                MAX_FALLBACK_CENTER_OFFSET_RADII * bbox_radius,
            )
            offset_length = float(np.linalg.norm(offset))
            if offset_length > limit and offset_length > 1e-6:
                offset = offset * (limit / offset_length)
            self.center_offset_px = offset.astype(np.float32, copy=True)
            if "size_outlier" not in method:
                self.radius_ratio = float(
                    np.clip(
                        measured_radius / max(2.0, bbox_radius),
                        MIN_FALLBACK_RADIUS_RATIO,
                        MAX_FALLBACK_RADIUS_RATIO,
                    )
                )
            self.has_formal_geometry = True
            return ContinuousGeometryMeasurement(measured_center.copy(), measured_radius)

        if not self.has_formal_geometry:
            return ContinuousGeometryMeasurement(measured_center.copy(), measured_radius)

        return ContinuousGeometryMeasurement(
            center_px=(bbox_center + self.center_offset_px).astype(np.float32, copy=False),
            radius_px=max(2.0, bbox_radius * self.radius_ratio),
            fallback_from_formal=True,
        )


def blend_tracked_center(
    previous_center,
    measurement: ContinuousGeometryMeasurement,
    *,
    previous_radius_px: float,
    geometry_quality: float,
) -> np.ndarray:
    previous = np.asarray(previous_center, dtype=np.float32).reshape((2,))
    measured = np.asarray(measurement.center_px, dtype=np.float32).reshape((2,))
    displacement = float(np.linalg.norm(measured - previous))
    radius = max(2.0, float(previous_radius_px), float(measurement.radius_px))
    if displacement > 0.35 * radius:
        return measured.copy()

    measurement_alpha = 0.35 + 0.40 * float(np.clip(geometry_quality, 0.0, 1.0))
    fine_displacement_limit = max(1.0, FINE_CENTER_DISPLACEMENT_RADIUS_RATIO * radius)
    if displacement <= fine_displacement_limit:
        measurement_alpha = min(measurement_alpha, FINE_CENTER_MEASUREMENT_ALPHA)
    if measurement.fallback_from_formal:
        measurement_alpha = min(measurement_alpha, FORMAL_TO_BBOX_FALLBACK_ALPHA)
    return previous + float(measurement_alpha) * (measured - previous)


def _bbox_center(detection: Detection) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in detection.bbox]
    return np.asarray([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float32)


def _bbox_radius(detection: Detection) -> float:
    x1, y1, x2, y2 = [float(value) for value in detection.bbox]
    return max(2.0, 0.25 * ((x2 - x1) + (y2 - y1)))


__all__ = [
    "BallGeometryContinuity",
    "ContinuousGeometryMeasurement",
    "blend_tracked_center",
]
