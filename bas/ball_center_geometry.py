from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .calibration.geometry import ProjectedEllipse
from .schemas import Point

if TYPE_CHECKING:
    from .calibration.service import CalibrationService


@dataclass(frozen=True)
class BallCenterResult:
    table_center_mm: Point
    radius_mm: float
    projector_ellipse: ProjectedEllipse
    uncertainty_mm: float
    support_weight: float
    geometry_method: str


class BallCenterGeometry:
    """Single seam for camera ball observations consumed by planning and projection."""

    def __init__(self, calibration: "CalibrationService") -> None:
        self._calibration = calibration

    def locate(
        self,
        center_px: Point,
        *,
        radius_px: float = 0.0,
        geometry_quality: float = 1.0,
        geometry_method: str = "unknown",
    ) -> BallCenterResult:
        point = np.asarray([center_px], dtype=np.float32)
        table = self._calibration.ball_camera_px_to_table_mm(point)[0]
        radius_mm = float(self._calibration.ball_pixel_radius_to_mm(center_px, float(radius_px)))
        if not np.isfinite(radius_mm) or radius_mm <= 1.0 or radius_mm > 80.0:
            radius_mm = 0.5 * float(self._calibration.table.ball_diameter_mm)
        ellipse = self._calibration.table_circle_to_projector_ellipse(table, radius_mm)
        camera_point = self._calibration._camera_geometry_points(point)
        support = float(self._calibration._geometry.ball_support_weights(camera_point)[0])
        report = self._calibration.geometry_quality_report
        model_p95 = float(report.get("ball_map_cv_p95_mm", report.get("ball_residual_cv_p95_mm", 8.0)))
        method = str(geometry_method or "unknown").strip().lower()
        method_floor = 7.0 if method.startswith("bbox") else 2.0 if "ellipse" in method else 4.0
        quality = float(np.clip(geometry_quality, 0.05, 1.0))
        uncertainty = max(method_floor, 0.45 * model_p95) / max(0.45, quality)
        uncertainty *= 1.0 + 0.75 * (1.0 - support)
        return BallCenterResult(
            table_center_mm=(float(table[0]), float(table[1])),
            radius_mm=radius_mm,
            projector_ellipse=ellipse,
            uncertainty_mm=float(uncertainty),
            support_weight=support,
            geometry_method=method,
        )
