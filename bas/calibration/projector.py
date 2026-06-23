from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..schemas import Point
from ..utils import ensure_numpy_points, percentile


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
    table_polygon_cam: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    table_polygon_proj: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    projector_size: Tuple[int, int] = (0, 0)
    source_path: Optional[str] = None
    quality_report: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load_json(cls, path: str | Path | None) -> "ProjectionCalibration":
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
        return cls(
            mode=str(data.get("mode", "none")),
            homography=np.asarray(data["homography"], dtype=np.float64) if "homography" in data else None,
            cam_points=np.asarray(data.get("cam_points", []), dtype=np.float64).reshape((-1, 2)),
            proj_points=np.asarray(data.get("proj_points", []), dtype=np.float64).reshape((-1, 2)),
            residual_field=residual,
            table_polygon_cam=np.asarray(data.get("table_polygon_cam", []), dtype=np.float64).reshape((-1, 2)),
            table_polygon_proj=np.asarray(data.get("table_polygon_proj", []), dtype=np.float64).reshape((-1, 2)),
            projector_size=(int(proj_size[0]), int(proj_size[1])) if len(proj_size) >= 2 else (0, 0),
            source_path=str(p),
            quality_report=dict(data.get("quality_report", {})),
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
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, float(ransac_threshold))
        if H is None:
            raise RuntimeError("cv2.findHomography failed.")
        residual = ResidualField()
        if src.shape[0] >= 12:
            base = _perspective_transform(src, H)
            residual = ResidualField(control_points_cam=src.copy(), offsets_proj=dst - base)
        return cls(
            mode=mode,
            homography=H,
            cam_points=src,
            proj_points=dst,
            residual_field=residual,
            projector_size=projector_size,
        )

    @property
    def is_valid(self) -> bool:
        return self.homography is not None

    @property
    def version(self) -> str:
        if self.source_path:
            return Path(self.source_path).stem
        return "no_projection_calibration"

    def camera_to_projector_points(self, points_cam: np.ndarray, *, refined: bool = True) -> np.ndarray:
        pts = np.asarray(points_cam, dtype=np.float64).reshape((-1, 2))
        if pts.size == 0:
            return pts.astype(np.float32)
        if self.homography is None:
            return pts.astype(np.float32)
        base = _perspective_transform(pts, self.homography)
        if refined:
            base = base + self.residual_field.offsets_for(pts)
        return base.astype(np.float32)

    def camera_to_projector_point(self, point: Point) -> Point:
        out = self.camera_to_projector_points(np.asarray([point], dtype=np.float64))
        return (float(out[0, 0]), float(out[0, 1]))

    def calibration_error_stats(self) -> Dict[str, float]:
        if self.cam_points.shape[0] == 0 or self.proj_points.shape != self.cam_points.shape:
            return {}
        pred = self.camera_to_projector_points(self.cam_points).astype(np.float64)
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
        data = {
            "mode": self.mode,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "homography": self.homography.tolist() if self.homography is not None else None,
            "cam_points": self.cam_points.tolist(),
            "proj_points": self.proj_points.tolist(),
            "residual_cam_points": self.residual_field.control_points_cam.tolist(),
            "residual_proj_offsets": self.residual_field.offsets_proj.tolist(),
            "table_polygon_cam": self.table_polygon_cam.tolist(),
            "table_polygon_proj": self.table_polygon_proj.tolist(),
            "projector_size": list(self.projector_size),
            "quality_report": self.calibration_error_stats(),
        }
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _perspective_transform(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 1, 2))
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape((-1, 2)).astype(np.float64)


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

