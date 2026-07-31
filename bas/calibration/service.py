from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..config import CalibrationConfig, CameraConfig
from ..schemas import Point, TableModel
from ..utils import ensure_numpy_points
from .ball_compensation import BallCompensationModel
from .camera import CameraCalibration
from .geometry import IndependentGeometry
from .projector import ProjectionCalibration


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
    auto_reference: bool = True
    ref_x_px: float = 0.0
    ref_y_px: float = 0.0
    scale_x_pct: float = 0.0
    scale_y_pct: float = 0.0

    @classmethod
    def from_config(cls, config: CalibrationConfig) -> "BallCenterCompensation":
        return cls(
            enabled=bool(config.ball_center_compensation_enabled),
            auto_reference=bool(config.ball_center_compensation_auto_reference),
            ref_x_px=float(config.ball_center_compensation_ref_x_px),
            ref_y_px=float(config.ball_center_compensation_ref_y_px),
            scale_x_pct=float(config.ball_center_compensation_scale_x_pct),
            scale_y_pct=float(config.ball_center_compensation_scale_y_pct),
        )


@dataclass
class CalibrationService:
    camera: CameraCalibration
    projection: ProjectionCalibration
    table: TableModel
    frame_undistorted: bool = False
    distortion_correction_enabled: bool = True
    projection_mode: str = "legacy"
    ball_compensation_model: BallCompensationModel = field(default_factory=BallCompensationModel)
    ball_center_compensation: BallCenterCompensation = field(default_factory=BallCenterCompensation)
    _geometry: IndependentGeometry = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_geometry()

    def _rebuild_geometry(self) -> None:
        self._geometry = IndependentGeometry(
            projection=self.projection,
            table=self.table,
            ball_compensation=self.ball_compensation_model,
        )

    def sync_ball_center_compensation(self, config: CalibrationConfig) -> None:
        self.ball_center_compensation = BallCenterCompensation.from_config(config)

    @property
    def calib_version(self) -> str:
        return f"{self.camera.version}+{self.projection.version}+{self.ball_compensation_model.version}"

    @property
    def is_engineered_projection(self) -> bool:
        return str(self.projection_mode).strip().lower() == "engineered"

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.distortion_correction_enabled:
            return frame
        return self.camera.undistort(frame)

    def camera_px_to_projector_px(self, points: np.ndarray) -> np.ndarray:
        camera_points = self._camera_geometry_points(points)
        return self._geometry.camera_to_projector(camera_points)

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
        return self.table_mm_to_projector_px(self.ball_camera_px_to_table_mm(points))

    def camera_px_to_table_mm(self, points: np.ndarray) -> np.ndarray:
        return self._geometry.camera_to_table(self._camera_geometry_points(points))

    def ball_camera_px_to_table_mm(self, points: np.ndarray) -> np.ndarray:
        pts = ensure_numpy_points(points).astype(np.float32)
        if not self.is_engineered_projection or not self.ball_compensation_model.is_valid:
            pts = self.compensate_ball_image_points(pts)
        camera_points = self._camera_geometry_points(pts)
        if self.is_engineered_projection and self.ball_compensation_model.is_valid:
            return self._geometry.ball_to_table(camera_points)
        return self._geometry.camera_to_table(camera_points)

    def projector_px_to_table_mm(self, points: np.ndarray) -> np.ndarray:
        return self._geometry.projector_to_table(points)

    def table_mm_to_projector_px(self, points: np.ndarray) -> np.ndarray:
        return self._geometry.table_to_projector(points)

    def table_mm_to_camera_px(self, points: np.ndarray) -> np.ndarray:
        camera_points = self._geometry.table_to_camera(points)
        if self._should_undistort_camera_points():
            return self.camera.distort_points(camera_points)
        return camera_points

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
        if self.is_engineered_projection:
            return 0.5 * float(self.table.ball_diameter_mm)
        center = np.asarray([center_px], dtype=np.float32)
        edge = np.asarray([[center_px[0] + radius_px, center_px[1]]], dtype=np.float32)
        mm = self.ball_camera_px_to_table_mm(np.vstack([center, edge]))
        return float(np.linalg.norm(mm[1] - mm[0]))

    def ball_projector_radius_px(self, center_px: Point) -> float:
        center = np.asarray([center_px], dtype=np.float32)
        center_mm = self.ball_camera_px_to_table_mm(center)[0]
        return self.table_radius_to_projector_px(center_mm, 0.5 * float(self.table.ball_diameter_mm))

    def table_radius_to_projector_px(self, center_mm: np.ndarray | Point, radius_mm: float) -> float:
        radius = float(max(0.0, radius_mm))
        if radius <= 0.0:
            return 0.0
        point = np.asarray(center_mm, dtype=np.float32).reshape((2,))
        refs = np.asarray(
            [
                point,
                point + np.asarray([radius, 0.0], dtype=np.float32),
                point + np.asarray([0.0, radius], dtype=np.float32),
            ],
            dtype=np.float32,
        )
        proj = self.table_mm_to_projector_px(refs).astype(np.float32)
        rx = float(np.linalg.norm(proj[1] - proj[0]))
        ry = float(np.linalg.norm(proj[2] - proj[0]))
        return max(1.0, 0.5 * (rx + ry))

    def _ball_center_reference_px(self) -> tuple[float, float]:
        cfg = self.ball_center_compensation
        if not bool(cfg.auto_reference):
            return float(cfg.ref_x_px), float(cfg.ref_y_px)
        if self.camera.image_size[0] > 0 and self.camera.image_size[1] > 0:
            return 0.5 * float(self.camera.image_size[0]), 0.5 * float(self.camera.image_size[1])
        poly = ensure_numpy_points(self.projection.table_polygon_cam)
        if poly.shape[0] >= 3:
            return float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))
        return 960.0, 540.0

    @property
    def geometry_quality_report(self) -> dict[str, float | int | str]:
        return self._geometry.quality_report

    def _should_undistort_camera_points(self) -> bool:
        return (
            self.distortion_correction_enabled
            and self.camera.is_valid
            and not self.frame_undistorted
        )

    def _camera_geometry_points(self, points: np.ndarray) -> np.ndarray:
        pts = ensure_numpy_points(points)
        if self._should_undistort_camera_points():
            return self.camera.undistort_points(pts)
        return pts.astype(np.float32)


def create_calibration_service(
    config: CalibrationConfig,
    frame_undistorted: bool = False,
    *,
    distortion_correction_enabled: bool,
) -> CalibrationService:
    config.sync_projection_file_alias()
    correction_enabled = bool(distortion_correction_enabled)
    camera = (
        CameraCalibration.load_opencv_yaml(config.camera_file)
        if correction_enabled
        else CameraCalibration(metadata={"disabled": True})
    )
    projection = ProjectionCalibration.load_json(config.active_projection_file())
    projection_mode = config.normalized_projection_mode()
    ball_compensation_model = BallCompensationModel.load_json(
        config.engineered_ball_compensation_file if projection_mode == "engineered" else None
    )
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
        distortion_correction_enabled=correction_enabled,
        projection_mode=projection_mode,
        ball_compensation_model=ball_compensation_model,
        ball_center_compensation=BallCenterCompensation.from_config(config),
    )


def create_setting_aware_calibration_service(
    calibration_config: CalibrationConfig,
    camera_config: CameraConfig,
    frame_undistorted: bool = False,
) -> CalibrationService:
    """Create calibration state without overriding the active camera correction setting."""

    return create_calibration_service(
        calibration_config,
        frame_undistorted=bool(frame_undistorted),
        distortion_correction_enabled=bool(camera_config.distortion_correction_enabled),
    )
