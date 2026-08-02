from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..config import CalibrationConfig, CameraConfig, DetectorConfig, ProjectionConfig
from ..geometry_contract import calibration_context, projection_calibration_context
from ..schemas import Point, TableModel
from ..utils import ensure_numpy_points
from .ball_compensation import BallCompensationModel
from .ball_compensation_sampling import evaluate_ball_compensation_quality
from .quality_standards import BALL_COMPENSATION_MIN_TRAINING_SAMPLES
from .camera import CameraCalibration
from .geometry import IndependentGeometry, ProjectedEllipse
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
    ball_compensation_model: BallCompensationModel = field(default_factory=BallCompensationModel)
    ball_center_compensation: BallCenterCompensation = field(default_factory=BallCenterCompensation)
    _geometry: IndependentGeometry = field(init=False, repr=False)
    ball_geometry: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_geometry()
        from ..ball_center_geometry import BallCenterGeometry

        self.ball_geometry = BallCenterGeometry(self)

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
        if not self.ball_compensation_model.is_valid:
            pts = self.compensate_ball_image_points(pts)
        camera_points = self._camera_geometry_points(pts)
        if self.ball_compensation_model.is_valid:
            return self._geometry.ball_to_table(camera_points)
        return self._geometry.camera_to_table(camera_points)

    def projector_px_to_table_mm(self, points: np.ndarray) -> np.ndarray:
        return self._geometry.projector_to_table(points)

    def table_mm_to_projector_px(self, points: np.ndarray) -> np.ndarray:
        return self._geometry.table_to_projector(points)

    def table_circle_to_projector_ellipse(
        self,
        center_mm: np.ndarray | Point,
        radius_mm: float,
    ) -> ProjectedEllipse:
        return self._geometry.table_circle_to_projector_ellipse(center_mm, radius_mm)

    def ball_projector_ellipse(self, center_px: Point) -> ProjectedEllipse:
        center_mm = self.ball_camera_px_to_table_mm(np.asarray([center_px], dtype=np.float32))[0]
        return self.table_circle_to_projector_ellipse(center_mm, 0.5 * float(self.table.ball_diameter_mm))

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
        if self.ball_compensation_model.is_valid:
            return 0.5 * float(self.table.ball_diameter_mm)
        center = np.asarray([center_px], dtype=np.float32)
        edge = np.asarray([[center_px[0] + radius_px, center_px[1]]], dtype=np.float32)
        mm = self.ball_camera_px_to_table_mm(np.vstack([center, edge]))
        return float(np.linalg.norm(mm[1] - mm[0]))

    def ball_projector_radius_px(self, center_px: Point) -> float:
        ellipse = self.ball_projector_ellipse(center_px)
        return 0.5 * (float(ellipse.radius_x_px) + float(ellipse.radius_y_px))

    def table_radius_to_projector_px(self, center_mm: np.ndarray | Point, radius_mm: float) -> float:
        radius = float(max(0.0, radius_mm))
        if radius <= 0.0:
            return 0.0
        ellipse = self.table_circle_to_projector_ellipse(center_mm, radius)
        return max(1.0, 0.5 * (float(ellipse.radius_x_px) + float(ellipse.radius_y_px)))

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
    ball_compensation_expected_context: dict[str, object] | None = None,
    projection_expected_context: dict[str, object] | None = None,
) -> CalibrationService:
    correction_enabled = bool(distortion_correction_enabled)
    camera = (
        CameraCalibration.load_opencv_yaml(config.camera_file)
        if correction_enabled
        else CameraCalibration(metadata={"disabled": True})
    )
    projection = ProjectionCalibration.load_json(
        config.projection_file,
        expected_context=projection_expected_context,
    )
    ball_compensation_model = BallCompensationModel.load_json(
        config.engineered_ball_compensation_file,
        expected_context=ball_compensation_expected_context,
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
    _audit_legacy_ball_compensation_model(ball_compensation_model, table)
    return CalibrationService(
        camera=camera,
        projection=projection,
        table=table,
        frame_undistorted=bool(frame_undistorted),
        distortion_correction_enabled=correction_enabled,
        ball_compensation_model=ball_compensation_model,
        ball_center_compensation=BallCenterCompensation.from_config(config),
    )


def _audit_legacy_ball_compensation_model(
    model: BallCompensationModel,
    table: TableModel,
) -> None:
    """Backfill the quality gate for old full-grid artifacts at load time."""

    if "quality_gate_passed" in model.quality_report:
        return
    controls = np.asarray(model.control_points_camera_px, dtype=np.float64).reshape((-1, 2))
    targets = np.asarray(model.target_table_mm, dtype=np.float64).reshape((-1, 2))
    if (
        targets.shape[0] > 0
        and controls.shape == targets.shape
        and controls.shape[0] < BALL_COMPENSATION_MIN_TRAINING_SAMPLES
    ):
        model.quality_report = {
            **model.quality_report,
            "legacy_model_audited": True,
            "quality_gate_passed": False,
            "quality_gate_errors": [
                f"legacy global ball-center model has {controls.shape[0]} samples; "
                f"at least {BALL_COMPENSATION_MIN_TRAINING_SAMPLES} are required"
            ],
        }
        return
    if controls.shape != targets.shape or controls.shape[0] < BALL_COMPENSATION_MIN_TRAINING_SAMPLES:
        return
    weights = model.sample_weights if model.sample_weights.shape[0] == controls.shape[0] else None
    audit = evaluate_ball_compensation_quality(
        controls,
        targets,
        sample_weights=weights,
        ball_diameter_mm=float(table.ball_diameter_mm),
        table_width_mm=float(table.width_mm),
        table_height_mm=float(table.height_mm),
    )
    model.quality_report = {
        **model.quality_report,
        **audit,
        "legacy_model_audited": True,
    }


def create_setting_aware_calibration_service(
    calibration_config: CalibrationConfig,
    camera_config: CameraConfig,
    frame_undistorted: bool = False,
    detector_config: DetectorConfig | None = None,
    projection_config: ProjectionConfig | None = None,
    actual_frame_size: tuple[int, int] | None = None,
) -> CalibrationService:
    """Create calibration state; actual_frame_size is the post-rotation capture size."""

    coordinate_domain = (
        "undistorted"
        if bool(camera_config.distortion_correction_enabled)
        and CameraCalibration.load_opencv_yaml(camera_config.distortion_correction_file).is_valid
        else "raw"
    )
    rotation_degrees = int(camera_config.frame_rotation_degrees) % 360
    if actual_frame_size is not None:
        frame_width = int(actual_frame_size[0])
        frame_height = int(actual_frame_size[1])
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("actual_frame_size must contain positive oriented-frame dimensions")
    else:
        frame_width = int(camera_config.height) if rotation_degrees in {90, 270} else int(camera_config.width)
        frame_height = int(camera_config.width) if rotation_degrees in {90, 270} else int(camera_config.height)
    projection_context = projection_calibration_context(
        frame_width=frame_width,
        frame_height=frame_height,
        frame_rotation_degrees=rotation_degrees,
        camera_coordinate_domain=coordinate_domain,
        distortion_file=camera_config.distortion_correction_file if coordinate_domain == "undistorted" else None,
        projector_width=int(projection_config.projector_width) if projection_config is not None else 0,
        projector_height=int(projection_config.projector_height) if projection_config is not None else 0,
    )
    if projection_config is None:
        projection_context.pop("projector_width", None)
        projection_context.pop("projector_height", None)
    expected_context = calibration_context(
        frame_width=frame_width,
        frame_height=frame_height,
        frame_rotation_degrees=rotation_degrees,
        camera_coordinate_domain=coordinate_domain,
        distortion_file=camera_config.distortion_correction_file if coordinate_domain == "undistorted" else None,
        projection_file=calibration_config.projection_file,
        detector_model_file=detector_config.model_path if detector_config is not None else None,
        ball_diameter_mm=float(calibration_config.ball_diameter_mm),
    )
    return create_calibration_service(
        calibration_config,
        frame_undistorted=bool(frame_undistorted),
        distortion_correction_enabled=bool(camera_config.distortion_correction_enabled),
        ball_compensation_expected_context=expected_context,
        projection_expected_context=projection_context,
    )
