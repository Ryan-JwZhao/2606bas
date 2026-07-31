from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..schemas import TableModel
from ..utils import ensure_numpy_points
from .ball_compensation import BallCompensationModel
from .projector import ProjectionCalibration, table_bbox_from_polygon


def _perspective(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = ensure_numpy_points(points).astype(np.float64)
    if pts.size == 0:
        return pts.astype(np.float32)
    return cv2.perspectiveTransform(
        pts.reshape((-1, 1, 2)),
        np.asarray(homography, dtype=np.float64),
    ).reshape((-1, 2)).astype(np.float32)


def _features(points: np.ndarray, degree: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    x = pts[:, 0]
    y = pts[:, 1]
    columns = [np.ones_like(x), x, y]
    if int(degree) >= 2:
        columns.extend([x * x, x * y, y * y])
    return np.column_stack(columns)


@dataclass
class _RegularizedPointMap:
    """Small, robust and smooth 2-D point mapping used inside rig geometry."""

    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    degree: int
    source_min: np.ndarray
    source_max: np.ndarray
    cv_p95: float
    rbf_centers: Optional[np.ndarray] = None
    rbf_weights: Optional[np.ndarray] = None
    rbf_sigma: float = 0.0
    rbf_lambda: float = 0.0

    @classmethod
    def fit(
        cls,
        source: np.ndarray,
        target: np.ndarray,
        *,
        sample_weights: Optional[np.ndarray] = None,
        allow_quadratic: bool = True,
    ) -> Optional["_RegularizedPointMap"]:
        src = ensure_numpy_points(source).astype(np.float64)
        dst = ensure_numpy_points(target).astype(np.float64)
        if src.shape != dst.shape or src.shape[0] < 3:
            return None
        center = np.median(src, axis=0)
        scale = np.ptp(src, axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        normalized = (src - center) / scale
        weights = np.ones((src.shape[0],), dtype=np.float64)
        if sample_weights is not None:
            supplied = np.asarray(sample_weights, dtype=np.float64).reshape((-1,))
            if supplied.shape[0] == src.shape[0]:
                weights = np.clip(supplied, 0.05, None)
                weights /= max(1e-9, float(np.median(weights)))

        degrees = [1]
        if allow_quadratic and src.shape[0] >= 12:
            degrees.append(2)
        candidates: list[tuple[int, np.ndarray, float]] = []
        for degree in degrees:
            cv_p95 = _cross_validated_p95(normalized, dst, weights, degree)
            coefficients = _robust_fit(normalized, dst, weights, degree)
            candidates.append((degree, coefficients, cv_p95))

        selected = candidates[0]
        if len(candidates) > 1:
            quadratic = candidates[1]
            # A more complex surface must earn a material holdout improvement.
            if quadratic[2] < selected[2] * 0.92:
                selected = quadratic
        rbf_candidate = _select_rbf_candidate(normalized, dst, weights) if src.shape[0] >= 12 else None
        rbf_centers = None
        rbf_weights = None
        rbf_sigma = 0.0
        rbf_lambda = 0.0
        if rbf_candidate is not None and rbf_candidate[2] < selected[2] * 0.92:
            rbf_sigma, rbf_lambda, rbf_cv_p95 = rbf_candidate
            affine_coefficients = _robust_fit(normalized, dst, weights, degree=1)
            affine_residual = dst - _features(normalized, 1) @ affine_coefficients
            rbf_weights = _fit_rbf(normalized, affine_residual, weights, rbf_sigma, rbf_lambda)
            rbf_centers = normalized.copy()
            selected = (1, affine_coefficients, rbf_cv_p95)
        return cls(
            center=center,
            scale=scale,
            coefficients=selected[1],
            degree=selected[0],
            source_min=np.min(src, axis=0),
            source_max=np.max(src, axis=0),
            cv_p95=float(selected[2]),
            rbf_centers=rbf_centers,
            rbf_weights=rbf_weights,
            rbf_sigma=float(rbf_sigma),
            rbf_lambda=float(rbf_lambda),
        )

    def map(self, points: np.ndarray) -> np.ndarray:
        pts = ensure_numpy_points(points).astype(np.float64)
        if pts.size == 0:
            return pts.astype(np.float32)
        normalized = (pts - self.center) / self.scale
        mapped = _features(normalized, self.degree) @ self.coefficients
        if (
            self.rbf_centers is not None
            and self.rbf_weights is not None
            and self.rbf_sigma > 0.0
        ):
            mapped += _rbf_kernel(normalized, self.rbf_centers, self.rbf_sigma) @ self.rbf_weights
        return mapped.astype(np.float32)

    @property
    def model_kind(self) -> str:
        return "affine_rbf" if self.rbf_centers is not None else f"polynomial_{self.degree}"

    def supported(self, points: np.ndarray, margin_ratio: float = 0.08) -> np.ndarray:
        pts = ensure_numpy_points(points).astype(np.float64)
        margin = np.maximum(1.0, (self.source_max - self.source_min) * float(max(0.0, margin_ratio)))
        return np.all((pts >= self.source_min - margin) & (pts <= self.source_max + margin), axis=1)


def _robust_fit(
    normalized_source: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
    degree: int,
) -> np.ndarray:
    design = _features(normalized_source, degree)
    robust_weights = np.asarray(sample_weights, dtype=np.float64).copy()
    coefficients = np.zeros((design.shape[1], 2), dtype=np.float64)
    for _ in range(5):
        root_w = np.sqrt(np.clip(robust_weights, 1e-6, None))[:, None]
        weighted_design = design * root_w
        weighted_target = target * root_w
        ridge = np.eye(design.shape[1], dtype=np.float64) * 1e-5
        ridge[0, 0] = 0.0
        lhs = weighted_design.T @ weighted_design + ridge
        rhs = weighted_design.T @ weighted_target
        coefficients = np.linalg.solve(lhs, rhs)
        residual = np.linalg.norm(design @ coefficients - target, axis=1)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        huber = max(1e-6, median + 1.5 * 1.4826 * mad)
        robust = np.minimum(1.0, huber / np.maximum(residual, 1e-9))
        robust_weights = np.asarray(sample_weights, dtype=np.float64) * np.clip(robust, 0.08, 1.0)
    return coefficients


def _cross_validated_p95(
    normalized_source: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
    degree: int,
) -> float:
    count = normalized_source.shape[0]
    if count < 6:
        coefficients = _robust_fit(normalized_source, target, sample_weights, degree)
        errors = np.linalg.norm(_features(normalized_source, degree) @ coefficients - target, axis=1)
        return float(np.percentile(errors, 95))
    order = np.lexsort((normalized_source[:, 0], normalized_source[:, 1]))
    fold_count = min(5, count)
    errors: list[float] = []
    for fold in range(fold_count):
        validation = order[fold::fold_count]
        training = np.setdiff1d(np.arange(count), validation)
        minimum = 6 if degree >= 2 else 3
        if training.shape[0] < minimum or validation.shape[0] == 0:
            continue
        coefficients = _robust_fit(
            normalized_source[training],
            target[training],
            sample_weights[training],
            degree,
        )
        predicted = _features(normalized_source[validation], degree) @ coefficients
        errors.extend(np.linalg.norm(predicted - target[validation], axis=1).tolist())
    return float(np.percentile(errors, 95)) if errors else float("inf")


def _rbf_kernel(first: np.ndarray, second: np.ndarray, sigma: float) -> np.ndarray:
    delta = np.asarray(first, dtype=np.float64)[:, None, :] - np.asarray(second, dtype=np.float64)[None, :, :]
    squared_distance = np.sum(delta * delta, axis=2)
    return np.exp(-squared_distance / max(1e-9, 2.0 * float(sigma) * float(sigma)))


def _fit_rbf(
    source: np.ndarray,
    residual: np.ndarray,
    sample_weights: np.ndarray,
    sigma: float,
    regularization: float,
) -> np.ndarray:
    kernel = _rbf_kernel(source, source, sigma)
    weights = np.clip(np.asarray(sample_weights, dtype=np.float64), 0.05, None)
    penalty = float(regularization) * np.diag(1.0 / weights)
    return np.linalg.solve(kernel + penalty, np.asarray(residual, dtype=np.float64))


def _select_rbf_candidate(
    normalized_source: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
) -> Optional[tuple[float, float, float]]:
    count = normalized_source.shape[0]
    if count < 12:
        return None
    order = np.lexsort((normalized_source[:, 0], normalized_source[:, 1]))
    fold_count = min(5, count)
    best: Optional[tuple[float, float, float]] = None
    for sigma in (0.08, 0.12, 0.18, 0.25, 0.35):
        for regularization in (0.001, 0.01, 0.05, 0.1, 0.3, 1.0):
            errors: list[float] = []
            for fold in range(fold_count):
                validation = order[fold::fold_count]
                training = np.setdiff1d(np.arange(count), validation)
                if training.shape[0] < 6 or validation.shape[0] == 0:
                    continue
                affine = _robust_fit(
                    normalized_source[training],
                    target[training],
                    sample_weights[training],
                    degree=1,
                )
                training_residual = target[training] - _features(normalized_source[training], 1) @ affine
                rbf_weights = _fit_rbf(
                    normalized_source[training],
                    training_residual,
                    sample_weights[training],
                    sigma,
                    regularization,
                )
                predicted = _features(normalized_source[validation], 1) @ affine
                predicted += (
                    _rbf_kernel(normalized_source[validation], normalized_source[training], sigma)
                    @ rbf_weights
                )
                errors.extend(np.linalg.norm(predicted - target[validation], axis=1).tolist())
            if not errors:
                continue
            p95 = float(np.percentile(errors, 95))
            if best is None or p95 < best[2]:
                best = (float(sigma), float(regularization), p95)
    return best


@dataclass
class _PlanarMap:
    homography: np.ndarray
    residual: Optional[_RegularizedPointMap] = None

    def forward(self, points: np.ndarray) -> np.ndarray:
        pts = ensure_numpy_points(points).astype(np.float32)
        base = _perspective(pts, self.homography)
        if self.residual is None or pts.size == 0:
            return base
        correction = self.residual.map(pts)
        supported = self.residual.supported(pts, margin_ratio=0.15)
        correction[~supported] = 0.0
        return (base + correction).astype(np.float32)

    def inverse(self, points: np.ndarray) -> np.ndarray:
        target = ensure_numpy_points(points).astype(np.float32)
        if target.size == 0:
            return target
        inverse_h = np.linalg.inv(np.asarray(self.homography, dtype=np.float64))
        estimate = _perspective(target, inverse_h)
        if self.residual is None:
            return estimate
        # Invert the smooth residual field with a short numerical Newton solve.
        epsilon = 0.5
        for _ in range(6):
            predicted = self.forward(estimate)
            error = target - predicted
            if float(np.max(np.linalg.norm(error, axis=1))) < 1e-3:
                break
            dx = np.zeros_like(estimate)
            dy = np.zeros_like(estimate)
            dx[:, 0] = epsilon
            dy[:, 1] = epsilon
            jx = (self.forward(estimate + dx) - predicted) / epsilon
            jy = (self.forward(estimate + dy) - predicted) / epsilon
            for index in range(estimate.shape[0]):
                jacobian = np.column_stack([jx[index], jy[index]])
                try:
                    step = np.linalg.solve(jacobian, error[index])
                except np.linalg.LinAlgError:
                    step = np.linalg.lstsq(jacobian, error[index], rcond=None)[0]
                estimate[index] += np.clip(step, -25.0, 25.0)
        return estimate.astype(np.float32)


class IndependentGeometry:
    """Independent camera-plane, ball-center and projector-plane geometry.

    The module intentionally uses only 2-D calibrated correspondences. Camera
    extrinsics are neither required nor consulted.
    """

    def __init__(
        self,
        projection: ProjectionCalibration,
        table: TableModel,
        ball_compensation: Optional[BallCompensationModel] = None,
    ):
        self.projection = projection
        self.table = table
        self.ball_compensation = ball_compensation or BallCompensationModel()
        self._camera_table = self._build_camera_table_map()
        self._table_projector = self._build_table_projector_map()
        self._ball_direct, self._ball_residual = self._build_ball_maps()

    def camera_to_table(self, points_camera: np.ndarray) -> np.ndarray:
        return self._camera_table.forward(points_camera)

    def table_to_camera(self, points_table: np.ndarray) -> np.ndarray:
        return self._camera_table.inverse(points_table)

    def table_to_projector(self, points_table: np.ndarray) -> np.ndarray:
        return self._table_projector.forward(points_table)

    def projector_to_table(self, points_projector: np.ndarray) -> np.ndarray:
        return self._table_projector.inverse(points_projector)

    def camera_to_projector(self, points_camera: np.ndarray) -> np.ndarray:
        if not self.projection.is_valid:
            return ensure_numpy_points(points_camera).astype(np.float32)
        return self.table_to_projector(self.camera_to_table(points_camera))

    def ball_to_table(self, points_camera: np.ndarray) -> np.ndarray:
        points = ensure_numpy_points(points_camera).astype(np.float32)
        base = self.camera_to_table(points)
        if points.size == 0:
            return base
        if self._ball_direct is not None:
            direct = self._ball_direct.map(points)
            supported = self._ball_direct.supported(points)
            base[supported] = direct[supported]
        if self._ball_residual is not None:
            correction = self._ball_residual.map(points)
            supported = self._ball_residual.supported(points, margin_ratio=0.12)
            base[supported] += correction[supported]
        elif self._ball_direct is None and self.ball_compensation.is_valid:
            # Compatibility for old one/two-point files that cannot support a
            # stable smooth fit. New calibration sessions always use >= 3.
            base += self.ball_compensation.offsets_for_camera_points(points).astype(np.float32)
        return base.astype(np.float32)

    @property
    def quality_report(self) -> dict[str, float | int | str]:
        report: dict[str, float | int | str] = {
            "geometry_mode": "independent_2d",
            "camera_extrinsics_used": 0,
        }
        if self._table_projector.residual is not None:
            report["projector_residual_model"] = self._table_projector.residual.model_kind
            report["projector_residual_degree"] = int(self._table_projector.residual.degree)
            report["projector_residual_cv_p95_px"] = float(self._table_projector.residual.cv_p95)
        if self._ball_direct is not None:
            report["ball_map_model"] = self._ball_direct.model_kind
            report["ball_map_degree"] = int(self._ball_direct.degree)
            report["ball_map_cv_p95_mm"] = float(self._ball_direct.cv_p95)
        elif self._ball_residual is not None:
            report["ball_residual_model"] = self._ball_residual.model_kind
            report["ball_residual_degree"] = int(self._ball_residual.degree)
            report["ball_residual_cv_p95_mm"] = float(self._ball_residual.cv_p95)
        return report

    def _build_camera_table_map(self) -> _PlanarMap:
        table_rect = _table_rectangle(self.table)
        camera_polygon = ensure_numpy_points(self.projection.table_polygon_cam).astype(np.float32)
        if camera_polygon.shape[0] == 4:
            homography = cv2.getPerspectiveTransform(camera_polygon, table_rect)
            return _PlanarMap(homography=homography)

        projector_to_table = self._base_projector_to_table_homography()
        if self.projection.homography is not None:
            homography = projector_to_table @ np.asarray(self.projection.homography, dtype=np.float64)
            return _PlanarMap(homography=homography)
        return _PlanarMap(homography=np.eye(3, dtype=np.float64))

    def _build_table_projector_map(self) -> _PlanarMap:
        table_rect = _table_rectangle(self.table)
        projector_polygon = _projector_polygon(self.projection)
        homography = cv2.getPerspectiveTransform(table_rect, projector_polygon)
        residual = None
        normalized_controls = ensure_numpy_points(self.projection.table_control_points_norm).astype(np.float32)
        projector_controls = ensure_numpy_points(self.projection.table_control_points_proj).astype(np.float32)
        if normalized_controls.shape[0] >= 6 and normalized_controls.shape == projector_controls.shape:
            controls_table = normalized_controls * np.asarray(
                [float(self.table.width_mm), float(self.table.height_mm)],
                dtype=np.float32,
            )
            base_projector = _perspective(controls_table, homography)
            residual = _RegularizedPointMap.fit(
                controls_table,
                projector_controls - base_projector,
                allow_quadratic=True,
            )
        field = self.projection.residual_field
        if (
            residual is None
            and field.control_points_cam.shape[0] >= 6
            and field.control_points_cam.shape == field.offsets_proj.shape
            and self.projection.homography is not None
        ):
            controls_table = self._camera_table.forward(field.control_points_cam)
            observed_projector = (
                _perspective(field.control_points_cam, self.projection.homography)
                + field.offsets_proj.astype(np.float32)
            )
            base_projector = _perspective(controls_table, homography)
            residual = _RegularizedPointMap.fit(
                controls_table,
                observed_projector - base_projector,
                allow_quadratic=True,
            )
        return _PlanarMap(homography=homography, residual=residual)

    def _build_ball_maps(self) -> tuple[Optional[_RegularizedPointMap], Optional[_RegularizedPointMap]]:
        model = self.ball_compensation
        if not model.is_valid:
            return None, None
        weights = model.sample_weights if model.sample_weights.shape[0] == model.control_points_camera_px.shape[0] else None
        if model.target_table_mm.shape == model.control_points_camera_px.shape:
            direct = _RegularizedPointMap.fit(
                model.control_points_camera_px,
                model.target_table_mm,
                sample_weights=weights,
                allow_quadratic=True,
            )
            return direct, None
        residual = _RegularizedPointMap.fit(
            model.control_points_camera_px,
            model.delta_table_mm,
            sample_weights=weights,
            allow_quadratic=True,
        )
        return None, residual

    def _base_projector_to_table_homography(self) -> np.ndarray:
        table_rect = _table_rectangle(self.table)
        projector_polygon = _projector_polygon(self.projection)
        return cv2.getPerspectiveTransform(projector_polygon, table_rect)


def _table_rectangle(table: TableModel) -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [max(1e-3, float(table.width_mm)), 0.0],
            [max(1e-3, float(table.width_mm)), max(1e-3, float(table.height_mm))],
            [0.0, max(1e-3, float(table.height_mm))],
        ],
        dtype=np.float32,
    )


def _projector_polygon(projection: ProjectionCalibration) -> np.ndarray:
    polygon = ensure_numpy_points(projection.table_polygon_proj).astype(np.float32)
    if polygon.shape[0] == 4:
        return polygon
    x1, y1, x2, y2 = table_bbox_from_polygon(polygon, projection.projector_size)
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
