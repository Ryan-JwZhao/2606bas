from __future__ import annotations

from dataclasses import dataclass, field
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
class BallCenterCompensation:
    enabled: bool = False
    ref_x_px: float = -1.0
    ref_y_px: float = -1.0
    scale_x_pct: float = 0.0
    scale_y_pct: float = 0.0


@dataclass
class CalibrationService:
    camera: CameraCalibration
    projection: ProjectionCalibration
    table: TableModel
    frame_undistorted: bool = False
    ball_center_compensation: BallCenterCompensation = field(default_factory=BallCenterCompensation)

    @property
    def calib_version(self) -> str:
        return f"{self.camera.version}+{self.projection.version}"

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        return self.camera.undistort(frame)

    def camera_px_to_projector_px(self, points: np.ndarray) -> np.ndarray:
        undistorted = self.camera.undistort_points(points) if self.camera.is_valid and not self.frame_undistorted else ensure_numpy_points(points)
        return self.projection.camera_to_projector_points(undistorted)

    def compensate_ball_image_points(self, points: np.ndarray) -> np.ndarray:
        pts = ensure_numpy_points(points).astype(np.float32)
        cfg = self.ball_center_compensation
        if pts.size == 0 or not cfg.enabled:
            return pts
        sx = float(cfg.scale_x_pct) * 0.01
        sy = float(cfg.scale_y_pct) * 0.01
        if abs(sx) < 1e-6 and abs(sy) < 1e-6:
            return pts
        ref_x, ref_y = self._ball_center_reference_px()
        out = pts.copy()
        out[:, 0] += (float(ref_x) - out[:, 0]) * sx
        out[:, 1] += (float(ref_y) - out[:, 1]) * sy
        return out

    def ball_camera_px_to_projector_px(self, points: np.ndarray) -> np.ndarray:
        return self.camera_px_to_projector_px(self.compensate_ball_image_points(points))

    def camera_px_to_table_mm(self, points: np.ndarray) -> np.ndarray:
        proj = self.camera_px_to_projector_px(points).astype(np.float32)
        return self.projector_px_to_table_mm(proj)

    def ball_camera_px_to_table_mm(self, points: np.ndarray) -> np.ndarray:
        proj = self.ball_camera_px_to_projector_px(points).astype(np.float32)
        return self.projector_px_to_table_mm(proj)

    def projector_px_to_table_mm(self, points: np.ndarray) -> np.ndarray:
        pts = ensure_numpy_points(points)
        if pts.size == 0:
            return pts
        _, projector_to_table = self._table_projector_homographies()
        if projector_to_table is not None:
            return cv2.perspectiveTransform(pts.reshape((-1, 1, 2)).astype(np.float32), projector_to_table).reshape((-1, 2))
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
        table_to_projector, _ = self._table_projector_homographies()
        if table_to_projector is not None:
            return cv2.perspectiveTransform(pts.reshape((-1, 1, 2)).astype(np.float32), table_to_projector).reshape((-1, 2))
        x1, y1, x2, y2 = table_bbox_from_polygon(self.projection.table_polygon_proj, self.projection.projector_size)
        out = pts.copy().astype(np.float32)
        out[:, 0] = x1 + out[:, 0] / max(1e-6, float(self.table.width_mm)) * (x2 - x1)
        out[:, 1] = y1 + out[:, 1] / max(1e-6, float(self.table.height_mm)) * (y2 - y1)
        return out

    def table_mm_to_camera_px(self, points: np.ndarray) -> np.ndarray:
        proj = self.table_mm_to_projector_px(points).astype(np.float32)
        if proj.size == 0:
            return proj
        homography = self.projection.homography
        if homography is None:
            return proj
        inv_h = np.linalg.inv(np.asarray(homography, dtype=np.float64))
        cam = cv2.perspectiveTransform(proj.reshape((-1, 1, 2)), inv_h).reshape((-1, 2))
        return cam.astype(np.float32)

    def table_inner_polygon_projector(self) -> np.ndarray:
        return self.table_mm_to_projector_px(np.asarray(self.table.inner_polygon_mm, dtype=np.float32))

    def pocket_points_projector(self) -> np.ndarray:
        return self.table_mm_to_projector_px(np.asarray(self.table.pockets_mm, dtype=np.float32))

    def pixel_radius_to_mm(self, center_px: Point, radius_px: float) -> float:
        center = np.asarray([center_px], dtype=np.float32)
        edge = np.asarray([[center_px[0] + radius_px, center_px[1]]], dtype=np.float32)
        mm = self.camera_px_to_table_mm(np.vstack([center, edge]))
        return float(np.linalg.norm(mm[1] - mm[0]))

    def ball_pixel_radius_to_mm(self, center_px: Point, radius_px: float) -> float:
        center = np.asarray([center_px], dtype=np.float32)
        edge = np.asarray([[center_px[0] + radius_px, center_px[1]]], dtype=np.float32)
        mm = self.ball_camera_px_to_table_mm(np.vstack([center, edge]))
        return float(np.linalg.norm(mm[1] - mm[0]))

    def _table_projector_homographies(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        poly = ensure_numpy_points(self.projection.table_polygon_proj).astype(np.float32)
        if poly.shape[0] != 4:
            return None, None
        width = max(1e-3, float(self.table.width_mm))
        height = max(1e-3, float(self.table.height_mm))
        rect = np.asarray(
            [
                [0.0, 0.0],
                [width, 0.0],
                [width, height],
                [0.0, height],
            ],
            dtype=np.float32,
        )
        table_to_projector = cv2.getPerspectiveTransform(rect, poly)
        projector_to_table = cv2.getPerspectiveTransform(poly, rect)
        return table_to_projector, projector_to_table

    def _ball_center_reference_px(self) -> tuple[float, float]:
        cfg = self.ball_center_compensation
        ref_x = float(cfg.ref_x_px)
        ref_y = float(cfg.ref_y_px)
        if ref_x >= 0.0 and ref_y >= 0.0:
            return ref_x, ref_y
        if self.camera.camera_matrix is not None:
            k = np.asarray(self.camera.camera_matrix, dtype=np.float64)
            if k.shape[0] >= 3 and k.shape[1] >= 3:
                return float(k[0, 2]), float(k[1, 2])
        if self.camera.image_size[0] > 0 and self.camera.image_size[1] > 0:
            return 0.5 * float(self.camera.image_size[0]), 0.5 * float(self.camera.image_size[1])
        poly = ensure_numpy_points(self.projection.table_polygon_cam)
        if poly.shape[0] >= 3:
            return float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))
        return 960.0, 540.0


def create_calibration_service(config: CalibrationConfig, frame_undistorted: bool = False) -> CalibrationService:
    camera = CameraCalibration.load_opencv_yaml(config.camera_file)
    projection = ProjectionCalibration.load_json(config.projection_file)
    table = TableModel(
        width_mm=float(config.table_width_mm),
        height_mm=float(config.table_height_mm),
        ball_diameter_mm=float(config.ball_diameter_mm),
        inner_polygon_mm=default_inner_polygon(float(config.table_width_mm), float(config.table_height_mm)),
        pockets_mm=default_pockets(float(config.table_width_mm), float(config.table_height_mm)),
        projection_visible_polygon_mm=default_inner_polygon(float(config.table_width_mm), float(config.table_height_mm)),
        center_playable_polygon_mm=default_inner_polygon(float(config.table_width_mm), float(config.table_height_mm)),
        projection_visible_pockets_mm=default_pockets(float(config.table_width_mm), float(config.table_height_mm)),
    )
    return CalibrationService(
        camera=camera,
        projection=projection,
        table=table,
        frame_undistorted=bool(frame_undistorted),
        ball_center_compensation=BallCenterCompensation(
            enabled=bool(config.ball_center_compensation_enabled),
            ref_x_px=float(config.ball_center_compensation_ref_x_px),
            ref_y_px=float(config.ball_center_compensation_ref_y_px),
            scale_x_pct=float(config.ball_center_compensation_scale_x_pct),
            scale_y_pct=float(config.ball_center_compensation_scale_y_pct),
        ),
    )
