from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..schemas import Point
from ..geometry_contract import context_compatibility_errors
from ..utils import ensure_numpy_points, percentile


MIN_PROJECTION_INLIER_RATIO = 0.80
MAX_PROJECTION_INLIER_P95_PX = 3.0


@dataclass
class ResidualField:
    control_points_cam: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    offsets_proj: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    max_neighbors: int = 8

    def offsets_for(self, points_cam: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_cam, dtype=np.float64).reshape((-1, 2))
        if pts.size == 0 or self.control_points_cam.shape[0] < 4:
            return np.zeros_like(pts)
        out = np.zeros_like(pts)
        k = min(int(self.max_neighbors), self.control_points_cam.shape[0])
        for idx, pt in enumerate(pts):
            d2 = np.sum((self.control_points_cam - pt.reshape((1, 2))) ** 2, axis=1)
            nearest = int(np.argmin(d2))
            if d2[nearest] <= 1e-9:
                out[idx] = self.offsets_proj[nearest]
                continue
            nn = np.argsort(d2)[:k]
            weights = 1.0 / np.maximum(d2[nn], 1.0)
            out[idx] = np.sum(self.offsets_proj[nn] * weights[:, None], axis=0) / max(1e-9, float(np.sum(weights)))
        return out


@dataclass
class ProjectionCalibration:
    mode: str = "none"
    homography: Optional[np.ndarray] = None
    cam_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    proj_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    residual_field: ResidualField = field(default_factory=ResidualField)
    table_control_points_norm: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    table_control_points_proj: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    table_polygon_cam: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    table_polygon_proj: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    projector_size: Tuple[int, int] = (0, 0)
    source_path: Optional[str] = None
    quality_report: Dict[str, Any] = field(default_factory=dict)
    calibration_context: Dict[str, Any] = field(default_factory=dict)
    compatibility_errors: tuple[str, ...] = ()

    @classmethod
    def load_json(
        cls,
        path: str | Path | None,
        *,
        expected_context: Optional[Dict[str, Any]] = None,
    ) -> "ProjectionCalibration":
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls(source_path=str(p), quality_report={"missing": True})
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        residual = ResidualField(
            control_points_cam=np.asarray(data.get("residual_cam_points", []), dtype=np.float64).reshape((-1, 2)),
            offsets_proj=np.asarray(data.get("residual_proj_offsets", []), dtype=np.float64).reshape((-1, 2)),
        )
        proj_size = data.get("projector_size") or [0, 0]
        stored_context = dict(data.get("calibration_context", {}))
        return cls(
            mode=str(data.get("mode", "none")),
            homography=_validated_homography(data.get("homography")),
            cam_points=np.asarray(data.get("cam_points", []), dtype=np.float64).reshape((-1, 2)),
            proj_points=np.asarray(data.get("proj_points", []), dtype=np.float64).reshape((-1, 2)),
            residual_field=residual,
            table_control_points_norm=np.asarray(
                data.get("table_control_points_norm", []),
                dtype=np.float64,
            ).reshape((-1, 2)),
            table_control_points_proj=np.asarray(
                data.get("table_control_points_proj", []),
                dtype=np.float64,
            ).reshape((-1, 2)),
            table_polygon_cam=np.asarray(data.get("table_polygon_cam", []), dtype=np.float64).reshape((-1, 2)),
            table_polygon_proj=np.asarray(data.get("table_polygon_proj", []), dtype=np.float64).reshape((-1, 2)),
            projector_size=(int(proj_size[0]), int(proj_size[1])) if len(proj_size) >= 2 else (0, 0),
            source_path=str(p),
            quality_report=dict(data.get("quality_report", {})),
            calibration_context=stored_context,
            compatibility_errors=context_compatibility_errors(stored_context, expected_context),
        )

    @classmethod
    def fit_from_correspondences(
        cls,
        cam_points: Sequence[Point] | np.ndarray,
        proj_points: Sequence[Point] | np.ndarray,
        *,
        mode: str = "manual",
        ransac_threshold: float = 3.0,
        projector_size: Tuple[int, int] = (0, 0),
    ) -> "ProjectionCalibration":
        src = np.asarray(cam_points, dtype=np.float64).reshape((-1, 2))
        dst = np.asarray(proj_points, dtype=np.float64).reshape((-1, 2))
        if src.shape[0] < 4 or src.shape != dst.shape:
            raise ValueError("At least four paired camera/projector points are required.")
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, float(ransac_threshold))
        if H is None:
            raise RuntimeError("cv2.findHomography failed.")
        inlier_mask = _normalize_homography_mask(mask, src.shape[0])
        residual = ResidualField()
        quality_report = {
            "ransac_threshold_px": float(ransac_threshold),
            "ransac_inliers": int(np.count_nonzero(inlier_mask)),
            "ransac_outliers": int(src.shape[0] - np.count_nonzero(inlier_mask)),
            "ransac_inlier_ratio": float(np.count_nonzero(inlier_mask) / max(1, src.shape[0])),
        }
        base_errors = np.linalg.norm(_perspective_transform(src, H) - dst, axis=1)
        inlier_errors = base_errors[inlier_mask]
        if inlier_errors.size:
            quality_report.update(
                {
                    "ransac_inlier_mean_px": float(np.mean(inlier_errors)),
                    "ransac_inlier_p95_px": float(np.percentile(inlier_errors, 95)),
                    "ransac_inlier_max_px": float(np.max(inlier_errors)),
                }
            )
        if src.shape[0] >= 12 and int(np.count_nonzero(inlier_mask)) >= 4:
            base = _perspective_transform(src, H)
            residual = ResidualField(control_points_cam=src[inlier_mask].copy(), offsets_proj=(dst - base)[inlier_mask])
        return cls(
            mode=mode,
            homography=H,
            cam_points=src,
            proj_points=dst,
            residual_field=residual,
            projector_size=projector_size,
            quality_report=quality_report,
        )

    @property
    def is_valid(self) -> bool:
        if self.compatibility_errors:
            return False
        if _validated_homography(self.homography) is None:
            return False
        ratio = self.quality_report.get("ransac_inlier_ratio")
        if ratio is not None:
            try:
                ratio_value = float(ratio)
            except (TypeError, ValueError):
                return False
            if not np.isfinite(ratio_value) or ratio_value < MIN_PROJECTION_INLIER_RATIO:
                return False
        inlier_p95 = self.quality_report.get("ransac_inlier_p95_px")
        if inlier_p95 is not None:
            try:
                inlier_p95_value = float(inlier_p95)
            except (TypeError, ValueError):
                return False
            if not np.isfinite(inlier_p95_value) or inlier_p95_value > MAX_PROJECTION_INLIER_P95_PX:
                return False
        if self.quality_report.get("quality_gate_passed") is False:
            return False
        pattern_cv_p95 = self.quality_report.get("pattern_cv_p95_px")
        pattern_cv_limit = self.quality_report.get("maximum_pattern_cv_p95_px")
        if (
            pattern_cv_p95 is not None
            and pattern_cv_limit is not None
        ):
            try:
                pattern_value = float(pattern_cv_p95)
                pattern_limit = float(pattern_cv_limit)
            except (TypeError, ValueError):
                return False
            if not np.isfinite(pattern_value) or pattern_value > pattern_limit:
                return False
        return True

    @property
    def version(self) -> str:
        if self.source_path:
            return Path(self.source_path).stem
        return "no_projection_calibration"

    def camera_to_projector_points(self, points_cam: np.ndarray, *, refined: bool = True) -> np.ndarray:
        pts = np.asarray(points_cam, dtype=np.float64).reshape((-1, 2))
        if pts.size == 0:
            return pts.astype(np.float32)
        if not self.is_valid:
            reason = ", ".join(self.compatibility_errors) if self.compatibility_errors else "missing or low-quality homography"
            raise RuntimeError(f"Projection calibration is unavailable: {reason}.")
        base = _perspective_transform(pts, self.homography)
        if refined:
            base = base + self.residual_field.offsets_for(pts)
        return base.astype(np.float32)

    def camera_to_projector_point(self, point: Point) -> Point:
        out = self.camera_to_projector_points(np.asarray([point], dtype=np.float64))
        return (float(out[0, 0]), float(out[0, 1]))

    def calibration_error_stats(self) -> Dict[str, float]:
        homography = _validated_homography(self.homography)
        if homography is None or self.cam_points.shape[0] == 0 or self.proj_points.shape != self.cam_points.shape:
            return {}
        pred = _perspective_transform(self.cam_points, homography)
        if self.residual_field.control_points_cam.shape[0] >= 4:
            pred = pred + self.residual_field.offsets_for(self.cam_points)
        err = np.linalg.norm(pred - self.proj_points, axis=1)
        return {
            "mean_px": float(np.mean(err)),
            "median_px": float(np.median(err)),
            "p95_px": percentile(err, 95),
            "max_px": float(np.max(err)),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        quality_report = dict(self.quality_report)
        quality_report.update(self.calibration_error_stats())
        data = {
            "mode": self.mode,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "homography": self.homography.tolist() if self.homography is not None else None,
            "cam_points": self.cam_points.tolist(),
            "proj_points": self.proj_points.tolist(),
            "residual_cam_points": self.residual_field.control_points_cam.tolist(),
            "residual_proj_offsets": self.residual_field.offsets_proj.tolist(),
            "table_control_points_norm": self.table_control_points_norm.tolist(),
            "table_control_points_proj": self.table_control_points_proj.tolist(),
            "table_polygon_cam": self.table_polygon_cam.tolist(),
            "table_polygon_proj": self.table_polygon_proj.tolist(),
            "projector_size": list(self.projector_size),
            "quality_report": quality_report,
            "calibration_context": dict(self.calibration_context),
        }
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _perspective_transform(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 1, 2))
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape((-1, 2)).astype(np.float64)


def _validated_homography(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None
    if int(np.linalg.matrix_rank(matrix)) < 3 or abs(float(np.linalg.det(matrix))) < 1e-12:
        return None
    return matrix


def _normalize_homography_mask(mask: Optional[np.ndarray], count: int) -> np.ndarray:
    if mask is None:
        return np.ones((count,), dtype=bool)
    arr = np.asarray(mask).reshape((-1,))
    if arr.shape[0] != count:
        return np.ones((count,), dtype=bool)
    return arr.astype(bool)


def polygon_quad(points: np.ndarray | Sequence[Point]) -> np.ndarray:
    """Reduce a dense table outline to an image-ordered TL/TR/BR/BL quad."""

    pts = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    if pts.shape[0] < 4:
        return np.zeros((0, 2), dtype=np.float32)
    hull = cv2.convexHull(pts.reshape((-1, 1, 2))).reshape((-1, 2))
    candidate = hull
    if hull.shape[0] != 4:
        perimeter = float(cv2.arcLength(hull.reshape((-1, 1, 2)), True))
        for ratio in np.linspace(0.005, 0.15, 30):
            approx = cv2.approxPolyDP(hull.reshape((-1, 1, 2)), perimeter * float(ratio), True).reshape((-1, 2))
            if approx.shape[0] == 4:
                candidate = approx
                break
        else:
            candidate = cv2.boxPoints(cv2.minAreaRect(hull.reshape((-1, 1, 2)))).reshape((-1, 2))
    sums = candidate[:, 0] + candidate[:, 1]
    differences = candidate[:, 0] - candidate[:, 1]
    ordered = np.asarray(
        [
            candidate[int(np.argmin(sums))],
            candidate[int(np.argmax(differences))],
            candidate[int(np.argmax(sums))],
            candidate[int(np.argmin(differences))],
        ],
        dtype=np.float32,
    )
    if np.unique(np.round(ordered, decimals=4), axis=0).shape[0] != 4:
        return np.zeros((0, 2), dtype=np.float32)
    return ordered


def table_bbox_from_polygon(poly: np.ndarray, projector_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    pts = ensure_numpy_points(poly)
    if pts.shape[0] >= 3:
        return (float(np.min(pts[:, 0])), float(np.min(pts[:, 1])), float(np.max(pts[:, 0])), float(np.max(pts[:, 1])))
    w, h = projector_size
    if w > 0 and h > 0:
        margin_x = float(w) * 0.05
        margin_y = float(h) * 0.08
        return (margin_x, margin_y, float(w) - margin_x, float(h) - margin_y)
    return (0.0, 0.0, 1280.0, 800.0)
