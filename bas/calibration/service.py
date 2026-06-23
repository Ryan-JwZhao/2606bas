from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import CalibrationConfig
from ..schemas import Point, TableModel
from ..utils import ensure_numpy_points
from .camera import CameraCalibration
from .projector import ProjectionCalibration, table_bbox_from_polygon


def default_pockets(width_mm: float, height_mm: float) -> List[Point]:
    return [
        (0.0, 0.0),
        (width_mm * 0.5, 0.0),
        (width_mm, 0.0),
        (width_mm, height_mm),
        (width_mm * 0.5, height_mm),
        (0.0, height_mm),
    ]


def default_inner_polygon(width_mm: float, height_mm: float) -> List[Point]:
    return [(0.0, 0.0), (width_mm, 0.0), (width_mm, height_mm), (0.0, height_mm)]


@dataclass
class CalibrationService:
    camera: CameraCalibration
    projection: ProjectionCalibration
    table: TableModel

    @property
    def calib_version(self) -> str:
        return f"{self.camera.version}+{self.projection.version}"

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        return self.camera.undistort(frame)

    def camera_px_to_projector_px(self, points: np.ndarray) -> np.ndarray:
        undistorted = self.camera.undistort_points(points) if self.camera.is_valid else ensure_numpy_points(points)
        return self.projection.camera_to_projector_points(undistorted)

    def camera_px_to_table_mm(self, points: np.ndarray) -> np.ndarray:
        proj = self.camera_px_to_projector_px(points).astype(np.float32)
        return self.projector_px_to_table_mm(proj)

    def projector_px_to_table_mm(self, points: np.ndarray) -> np.ndarray:
        pts = ensure_numpy_points(points)
        if pts.size == 0:
            return pts
        x1, y1, x2, y2 = table_bbox_from_polygon(self.projection.table_polygon_proj, self.projection.projector_size)
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        out = pts.copy().astype(np.float32)
        out[:, 0] = (out[:, 0] - x1) / w * float(self.table.width_mm)
        out[:, 1] = (out[:, 1] - y1) / h * float(self.table.height_mm)
        return out

    def table_mm_to_projector_px(self, points: np.ndarray) -> np.ndarray:
        pts = ensure_numpy_points(points)
        if pts.size == 0:
            return pts
        x1, y1, x2, y2 = table_bbox_from_polygon(self.projection.table_polygon_proj, self.projection.projector_size)
        out = pts.copy().astype(np.float32)
        out[:, 0] = x1 + out[:, 0] / max(1e-6, float(self.table.width_mm)) * (x2 - x1)
        out[:, 1] = y1 + out[:, 1] / max(1e-6, float(self.table.height_mm)) * (y2 - y1)
        return out

    def table_inner_polygon_projector(self) -> np.ndarray:
        return self.table_mm_to_projector_px(np.asarray(self.table.inner_polygon_mm, dtype=np.float32))

    def pocket_points_projector(self) -> np.ndarray:
        return self.table_mm_to_projector_px(np.asarray(self.table.pockets_mm, dtype=np.float32))

    def pixel_radius_to_mm(self, center_px: Point, radius_px: float) -> float:
        center = np.asarray([center_px], dtype=np.float32)
        edge = np.asarray([[center_px[0] + radius_px, center_px[1]]], dtype=np.float32)
        mm = self.camera_px_to_table_mm(np.vstack([center, edge]))
        return float(np.linalg.norm(mm[1] - mm[0]))


def create_calibration_service(config: CalibrationConfig) -> CalibrationService:
    camera = CameraCalibration.load_opencv_yaml(config.camera_file)
    projection = ProjectionCalibration.load_json(config.projection_file)
    table = TableModel(
        width_mm=float(config.table_width_mm),
        height_mm=float(config.table_height_mm),
        ball_diameter_mm=float(config.ball_diameter_mm),
        inner_polygon_mm=default_inner_polygon(float(config.table_width_mm), float(config.table_height_mm)),
        pockets_mm=default_pockets(float(config.table_width_mm), float(config.table_height_mm)),
    )
    return CalibrationService(camera=camera, projection=projection, table=table)

