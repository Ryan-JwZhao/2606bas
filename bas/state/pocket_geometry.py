from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from ..config import StateConfig

LOGGER = logging.getLogger(__name__)

Point = tuple[float, float]


@dataclass(frozen=True)
class PocketGeometryContext:
    table_edge_polygon_mm: Sequence[Point] = ()
    ball_center_reachable_polygon_mm: Sequence[Point] = ()
    pockets_mm: Sequence[Point] = ()
    pocket_curves_mm: Sequence[Sequence[Point]] = ()
    ball_diameter_mm: float = 57.15

    def fingerprint(self) -> tuple[object, ...]:
        return (
            _points_fingerprint(self.table_edge_polygon_mm),
            _points_fingerprint(self.ball_center_reachable_polygon_mm),
            _points_fingerprint(self.pockets_mm),
            tuple(_points_fingerprint(curve) for curve in self.pocket_curves_mm),
            round(float(self.ball_diameter_mm), 6),
        )


@dataclass(frozen=True)
class PocketGeometry:
    index: int
    center_mm: Point
    tangent_unit: Point
    outward_normal: Point
    inward_normal: Point
    mouth_width_mm: float
    throat_width_mm: float
    interior_width_mm: float
    throat_depth_mm: float
    interior_depth_mm: float
    outward_probe_distance_mm: Optional[float] = None
    inward_probe_distance_mm: Optional[float] = None
    valid: bool = True
    validation_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PocketSample:
    zone: Optional[str]
    pocket_index: Optional[int]
    distance_mm: Optional[float]
    depth_mm: float
    lateral_mm: float
    pocketward_speed_mm_s: float
    inside_playable: bool
    geometry_valid: bool


@dataclass
class _GeometryValidation:
    valid: bool = True
    reasons: list[str] = field(default_factory=list)
    table_center_mm: Optional[Point] = None
    table_center_zone: Optional[str] = None
    sampled_points: int = 0
    sampled_interior_points: int = 0
    sampled_interior_ratio: float = 0.0


class PocketGeometryModel:
    """Pocket-local coordinates, zone sampling, and fail-closed validation."""

    def __init__(
        self,
        *,
        context: PocketGeometryContext,
        geometries: Sequence[PocketGeometry],
        validation: _GeometryValidation,
    ) -> None:
        self.context = context
        self.geometries = tuple(geometry for geometry in geometries if geometry.valid)
        self._all_geometries = tuple(geometries)
        self.validation = validation
        self._table_polygon = _polygon(context.table_edge_polygon_mm)
        reachable = _polygon(context.ball_center_reachable_polygon_mm)
        self._reachable_polygon = reachable if reachable is not None else self._table_polygon

    @classmethod
    def build(
        cls,
        context: PocketGeometryContext,
        config: StateConfig,
        *,
        log_diagnostics: bool = True,
    ) -> "PocketGeometryModel":
        table_polygon = _polygon(context.table_edge_polygon_mm)
        table_center = _polygon_centroid(table_polygon)
        geometries = _build_geometries(context, config, table_polygon, table_center)
        validation = _validate_model(context, geometries)
        model = cls(context=context, geometries=geometries, validation=validation)
        if log_diagnostics:
            model._log_diagnostics()
        return model

    @property
    def valid(self) -> bool:
        return bool(self.validation.valid and self.geometries)

    def sample(self, center_mm: Point, velocity_mm_s: Point = (0.0, 0.0)) -> PocketSample:
        inside_playable = _inside_polygon(self._reachable_polygon, center_mm)
        if not self.valid:
            return PocketSample(None, None, None, 0.0, 0.0, 0.0, inside_playable, False)
        sample = _sample_geometries(
            self.geometries,
            center_mm,
            velocity_mm_s,
            ball_diameter_mm=float(self.context.ball_diameter_mm),
        )
        return PocketSample(
            zone=sample.zone,
            pocket_index=sample.pocket_index,
            distance_mm=sample.distance_mm,
            depth_mm=sample.depth_mm,
            lateral_mm=sample.lateral_mm,
            pocketward_speed_mm_s=sample.pocketward_speed_mm_s,
            inside_playable=inside_playable,
            geometry_valid=True,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "valid": bool(self.valid),
            "reasons": list(self.validation.reasons),
            "table_center_mm": list(self.validation.table_center_mm) if self.validation.table_center_mm is not None else None,
            "table_center_zone": self.validation.table_center_zone,
            "sampled_points": int(self.validation.sampled_points),
            "sampled_interior_points": int(self.validation.sampled_interior_points),
            "sampled_interior_ratio": float(self.validation.sampled_interior_ratio),
            "pockets": [
                {
                    "index": int(geometry.index),
                    "center_mm": list(geometry.center_mm),
                    "tangent_unit": list(geometry.tangent_unit),
                    "outward_normal": list(geometry.outward_normal),
                    "inward_normal": list(geometry.inward_normal),
                    "mouth_width_mm": float(geometry.mouth_width_mm),
                    "throat_width_mm": float(geometry.throat_width_mm),
                    "interior_width_mm": float(geometry.interior_width_mm),
                    "throat_depth_mm": float(geometry.throat_depth_mm),
                    "interior_depth_mm": float(geometry.interior_depth_mm),
                    "outward_probe_distance_mm": geometry.outward_probe_distance_mm,
                    "inward_probe_distance_mm": geometry.inward_probe_distance_mm,
                    "valid": bool(geometry.valid),
                    "validation_reasons": list(geometry.validation_reasons),
                }
                for geometry in self._all_geometries
            ],
        }

    def _log_diagnostics(self) -> None:
        for geometry in self._all_geometries:
            LOGGER.debug(
                "Pocket geometry p%s center=%s tangent=%s outward=%s inward=%s widths=(%.2f, %.2f, %.2f) "
                "depths=(%.2f, %.2f) probes=(%s, %s) valid=%s reasons=%s",
                geometry.index,
                geometry.center_mm,
                geometry.tangent_unit,
                geometry.outward_normal,
                geometry.inward_normal,
                geometry.mouth_width_mm,
                geometry.throat_width_mm,
                geometry.interior_width_mm,
                geometry.throat_depth_mm,
                geometry.interior_depth_mm,
                geometry.outward_probe_distance_mm,
                geometry.inward_probe_distance_mm,
                geometry.valid,
                list(geometry.validation_reasons),
            )
        if not self.valid:
            LOGGER.error("Pocket geometry disabled (fail-closed): %s", self.validation.reasons)
        else:
            LOGGER.debug(
                "Pocket geometry validation center_zone=%s interior_ratio=%.4f samples=%s",
                self.validation.table_center_zone,
                self.validation.sampled_interior_ratio,
                self.validation.sampled_points,
            )


def _build_geometries(
    context: PocketGeometryContext,
    config: StateConfig,
    table_polygon: Optional[np.ndarray],
    table_center: np.ndarray,
) -> list[PocketGeometry]:
    geometries: list[PocketGeometry] = []
    curves = list(context.pocket_curves_mm or ())
    centers = list(context.pockets_mm or ())
    if curves:
        for index, curve in enumerate(curves):
            geometry = _build_curve_geometry(
                index,
                curve,
                context=context,
                config=config,
                table_polygon=table_polygon,
                table_center=table_center,
            )
            geometries.append(geometry)
        for index in range(len(curves), len(centers)):
            geometries.append(
                _build_center_geometry(index, centers[index], context=context, config=config, table_center=table_center)
            )
    else:
        for index, center in enumerate(centers):
            geometries.append(
                _build_center_geometry(index, center, context=context, config=config, table_center=table_center)
            )
    return geometries


def _build_curve_geometry(
    index: int,
    curve: Sequence[Point],
    *,
    context: PocketGeometryContext,
    config: StateConfig,
    table_polygon: Optional[np.ndarray],
    table_center: np.ndarray,
) -> PocketGeometry:
    reasons: list[str] = []
    curve_points = _points(curve)
    if curve_points.shape[0] < 2:
        return _invalid_geometry(index, reasons=["curve_has_fewer_than_two_points"], config=config, context=context)

    center = np.mean(curve_points, axis=0)
    tangent = _unit(curve_points[-1] - curve_points[0])
    if tangent is None:
        return _invalid_geometry(index, center=center, reasons=["curve_endpoints_are_degenerate"], config=config, context=context)

    normal1 = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    normal2 = -normal1
    outward, outward_probe, inward_probe, orientation_reasons = _choose_outward_normal(
        center,
        normal1,
        normal2,
        table_polygon=table_polygon,
        table_center=table_center,
    )
    reasons.extend(orientation_reasons)
    if outward is None:
        return _invalid_geometry(
            index,
            center=center,
            tangent=tangent,
            reasons=reasons or ["outward_normal_unresolved"],
            config=config,
            context=context,
        )

    mouth_half = max(float(np.linalg.norm(curve_points[-1] - curve_points[0])) * 0.5, context.ball_diameter_mm * 0.95)
    throat_half, interior_half, throat_depth, interior_depth = _dimensions(mouth_half, context, config)
    reasons.extend(_validate_axes(tangent, outward))
    if (
        outward_probe is not None
        and inward_probe is not None
        and outward_probe >= inward_probe - 1e-3
    ):
        reasons.append("invalid_outward_probe_not_more_external")
    valid = not any(reason.startswith("invalid_") for reason in reasons)
    return PocketGeometry(
        index=index,
        center_mm=_point(center),
        tangent_unit=_point(tangent),
        outward_normal=_point(outward),
        inward_normal=_point(-outward),
        mouth_width_mm=float(mouth_half * 2.0),
        throat_width_mm=float(throat_half * 2.0),
        interior_width_mm=float(interior_half * 2.0),
        throat_depth_mm=float(throat_depth),
        interior_depth_mm=float(interior_depth),
        outward_probe_distance_mm=outward_probe,
        inward_probe_distance_mm=inward_probe,
        valid=valid,
        validation_reasons=tuple(reasons),
    )


def _build_center_geometry(
    index: int,
    center: Point,
    *,
    context: PocketGeometryContext,
    config: StateConfig,
    table_center: np.ndarray,
) -> PocketGeometry:
    center_vec = np.asarray(center, dtype=np.float32).reshape((2,))
    outward = _unit(center_vec - table_center)
    if outward is None:
        return _invalid_geometry(index, center=center_vec, reasons=["center_equals_table_center"], config=config, context=context)
    tangent = np.asarray([-outward[1], outward[0]], dtype=np.float32)
    mouth_half = max(float(getattr(config, "pocket_mouth_radius_mm", 125.0)) * 0.68, context.ball_diameter_mm * 0.95)
    throat_half, interior_half, throat_depth, interior_depth = _dimensions(mouth_half, context, config)
    reasons = _validate_axes(tangent, outward)
    return PocketGeometry(
        index=index,
        center_mm=_point(center_vec),
        tangent_unit=_point(tangent),
        outward_normal=_point(outward),
        inward_normal=_point(-outward),
        mouth_width_mm=float(mouth_half * 2.0),
        throat_width_mm=float(throat_half * 2.0),
        interior_width_mm=float(interior_half * 2.0),
        throat_depth_mm=float(throat_depth),
        interior_depth_mm=float(interior_depth),
        valid=not any(reason.startswith("invalid_") for reason in reasons),
        validation_reasons=tuple(reasons),
    )


def _invalid_geometry(
    index: int,
    *,
    reasons: Sequence[str],
    config: StateConfig,
    context: PocketGeometryContext,
    center: np.ndarray | Sequence[float] = (0.0, 0.0),
    tangent: np.ndarray | Sequence[float] = (1.0, 0.0),
) -> PocketGeometry:
    mouth_half = max(context.ball_diameter_mm * 0.95, 1.0)
    throat_half, interior_half, throat_depth, interior_depth = _dimensions(mouth_half, context, config)
    return PocketGeometry(
        index=index,
        center_mm=_point(np.asarray(center, dtype=np.float32)),
        tangent_unit=_point(np.asarray(tangent, dtype=np.float32)),
        outward_normal=(0.0, 0.0),
        inward_normal=(0.0, 0.0),
        mouth_width_mm=float(mouth_half * 2.0),
        throat_width_mm=float(throat_half * 2.0),
        interior_width_mm=float(interior_half * 2.0),
        throat_depth_mm=float(throat_depth),
        interior_depth_mm=float(interior_depth),
        valid=False,
        validation_reasons=tuple(reasons),
    )


def _dimensions(
    mouth_half: float,
    context: PocketGeometryContext,
    config: StateConfig,
) -> tuple[float, float, float, float]:
    diameter = max(1.0, float(context.ball_diameter_mm))
    throat_half = min(float(mouth_half), max(diameter * 0.62, float(mouth_half) * 0.56))
    interior_half = min(
        throat_half,
        max(diameter * 0.52, float(getattr(config, "pocket_interior_radius_mm", 44.0)) * 0.75),
    )
    throat_depth = max(diameter * 0.55, float(getattr(config, "pocket_throat_radius_mm", 75.0)) * 0.55)
    interior_depth = max(
        throat_depth + diameter * 0.45,
        float(getattr(config, "pocket_interior_radius_mm", 44.0)) * 1.25,
    )
    return throat_half, interior_half, throat_depth, interior_depth


def _choose_outward_normal(
    center: np.ndarray,
    normal1: np.ndarray,
    normal2: np.ndarray,
    *,
    table_polygon: Optional[np.ndarray],
    table_center: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[float], Optional[float], list[str]]:
    reasons: list[str] = []
    probe_mm = 50.0
    if table_polygon is not None:
        score1 = _signed_distance(table_polygon, center + normal1 * probe_mm)
        score2 = _signed_distance(table_polygon, center + normal2 * probe_mm)
        if score1 is not None and score2 is not None and abs(score1 - score2) > 1e-3:
            if score1 < score2:
                return normal1, score1, score2, reasons
            return normal2, score2, score1, reasons
        reasons.append("probe_tie_radial_fallback")

    radial = center - table_center
    radial_unit = _unit(radial)
    if radial_unit is None:
        reasons.append("invalid_radial_fallback")
        return None, None, None, reasons
    dot1 = float(np.dot(normal1, radial_unit))
    dot2 = float(np.dot(normal2, radial_unit))
    if abs(dot1 - dot2) <= 1e-4:
        reasons.append("invalid_outward_normal_ambiguous")
        return None, None, None, reasons
    outward = normal1 if dot1 > dot2 else normal2
    outward_score = _signed_distance(table_polygon, center + outward * probe_mm) if table_polygon is not None else None
    inward_score = _signed_distance(table_polygon, center - outward * probe_mm) if table_polygon is not None else None
    return outward, outward_score, inward_score, reasons


def _validate_axes(tangent: np.ndarray, outward: np.ndarray) -> list[str]:
    reasons: list[str] = []
    if not np.all(np.isfinite(tangent)) or not np.all(np.isfinite(outward)):
        reasons.append("invalid_nonfinite_axes")
        return reasons
    if abs(float(np.linalg.norm(tangent)) - 1.0) > 1e-3:
        reasons.append("invalid_tangent_not_unit")
    if abs(float(np.linalg.norm(outward)) - 1.0) > 1e-3:
        reasons.append("invalid_outward_not_unit")
    if abs(float(np.dot(tangent, outward))) > 0.05:
        reasons.append("invalid_axes_not_orthogonal")
    return reasons


def _validate_model(
    context: PocketGeometryContext,
    geometries: Sequence[PocketGeometry],
) -> _GeometryValidation:
    validation = _GeometryValidation()
    valid_geometries = [geometry for geometry in geometries if geometry.valid]
    if not valid_geometries:
        validation.valid = False
        validation.reasons.append("no_valid_pocket_geometries")
        return validation

    table_polygon = _polygon(context.table_edge_polygon_mm)
    reachable_polygon = _polygon(context.ball_center_reachable_polygon_mm)
    sample_polygon = reachable_polygon if reachable_polygon is not None else table_polygon
    if table_polygon is None:
        validation.valid = False
        validation.reasons.append("table_edge_polygon_missing")
        return validation
    if sample_polygon is None:
        validation.valid = False
        validation.reasons.append("ball_center_reachable_polygon_missing")
        return validation

    center = _polygon_centroid(sample_polygon)
    validation.table_center_mm = _point(center)
    center_sample = _sample_geometries(valid_geometries, _point(center), (0.0, 0.0), context.ball_diameter_mm)
    validation.table_center_zone = center_sample.zone
    if center_sample.zone is not None:
        validation.valid = False
        validation.reasons.append("table_center_is_inside_pocket_zone")

    sampled = _sample_polygon_points(sample_polygon, count=500, seed=2606)
    interior_count = 0
    for point in sampled:
        sample = _sample_geometries(valid_geometries, _point(point), (0.0, 0.0), context.ball_diameter_mm)
        if sample.zone == "interior":
            interior_count += 1
    validation.sampled_points = len(sampled)
    validation.sampled_interior_points = interior_count
    validation.sampled_interior_ratio = float(interior_count / len(sampled)) if sampled else 1.0
    if len(sampled) < 500:
        validation.valid = False
        validation.reasons.append("insufficient_playable_validation_samples")
    if validation.sampled_interior_ratio >= 0.02:
        validation.valid = False
        validation.reasons.append("playable_interior_ratio_exceeded")
    return validation


def _sample_geometries(
    geometries: Sequence[PocketGeometry],
    center_mm: Point,
    velocity_mm_s: Point,
    ball_diameter_mm: float,
) -> PocketSample:
    pos = np.asarray(center_mm, dtype=np.float32)
    velocity = np.asarray(velocity_mm_s, dtype=np.float32)
    radius = max(0.5, float(ball_diameter_mm) * 0.5)
    matches: list[tuple[int, float, PocketSample]] = []
    nearest_distance: Optional[float] = None
    for geometry in geometries:
        center = np.asarray(geometry.center_mm, dtype=np.float32)
        tangent = np.asarray(geometry.tangent_unit, dtype=np.float32)
        outward = np.asarray(geometry.outward_normal, dtype=np.float32)
        delta = pos - center
        distance = float(np.linalg.norm(delta))
        nearest_distance = distance if nearest_distance is None else min(nearest_distance, distance)
        depth = float(np.dot(delta, outward))
        lateral = float(abs(np.dot(delta, tangent)))
        pocketward_speed = float(np.dot(velocity, outward))

        zone: Optional[str] = None
        if depth >= geometry.interior_depth_mm and lateral <= geometry.interior_width_mm * 0.5:
            zone = "interior"
        elif depth >= geometry.throat_depth_mm and lateral <= geometry.throat_width_mm * 0.5 + radius:
            zone = "throat"
        elif depth >= -radius * 0.6 and lateral <= geometry.mouth_width_mm * 0.5 + radius:
            zone = "mouth"
        if zone is None:
            continue
        rank = {"mouth": 1, "throat": 2, "interior": 3}[zone]
        matches.append(
            (
                rank,
                distance,
                PocketSample(
                    zone,
                    geometry.index,
                    distance,
                    depth,
                    lateral,
                    pocketward_speed,
                    False,
                    True,
                ),
            )
        )
    if not matches:
        return PocketSample(None, None, nearest_distance, 0.0, 0.0, 0.0, False, True)
    matches.sort(key=lambda item: (-item[0], item[1], int(item[2].pocket_index or 0)))
    return matches[0][2]


def _sample_polygon_points(polygon: np.ndarray, *, count: int, seed: int) -> list[np.ndarray]:
    points = polygon.reshape((-1, 2))
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    max_attempts = max(10_000, count * 200)
    for _ in range(max_attempts):
        point = rng.uniform(lo, hi).astype(np.float32)
        if _inside_polygon(polygon, _point(point)):
            sampled.append(point)
            if len(sampled) >= count:
                break
    return sampled


def _polygon(points: Sequence[Point]) -> Optional[np.ndarray]:
    arr = _points(points)
    if arr.shape[0] < 3:
        return None
    return arr.reshape((-1, 1, 2)).astype(np.float32)


def _polygon_centroid(polygon: Optional[np.ndarray]) -> np.ndarray:
    if polygon is None:
        return np.asarray([0.0, 0.0], dtype=np.float32)
    moments = cv2.moments(polygon)
    if abs(float(moments.get("m00", 0.0))) > 1e-6:
        return np.asarray(
            [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])],
            dtype=np.float32,
        )
    return np.mean(polygon.reshape((-1, 2)), axis=0).astype(np.float32)


def _inside_polygon(polygon: Optional[np.ndarray], point: Point) -> bool:
    if polygon is None:
        return False
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def _signed_distance(polygon: Optional[np.ndarray], point: np.ndarray) -> Optional[float]:
    if polygon is None:
        return None
    return float(cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), True))


def _points(points: Sequence[Point]) -> np.ndarray:
    try:
        values = list(points) if points is not None else []
        arr = np.asarray(values, dtype=np.float32).reshape((-1, 2))
    except (TypeError, ValueError):
        return np.zeros((0, 2), dtype=np.float32)
    return arr


def _points_fingerprint(points: Sequence[Point]) -> tuple[tuple[float, float], ...]:
    values = list(points) if points is not None else []
    return tuple((round(float(point[0]), 5), round(float(point[1]), 5)) for point in values if len(point) >= 2)


def _unit(vector: np.ndarray) -> Optional[np.ndarray]:
    value = np.asarray(vector, dtype=np.float32).reshape((2,))
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-6:
        return None
    return value / norm


def _point(value: np.ndarray | Sequence[float]) -> Point:
    arr = np.asarray(value, dtype=np.float32).reshape((2,))
    return (float(arr[0]), float(arr[1]))


__all__ = [
    "PocketGeometry",
    "PocketGeometryContext",
    "PocketGeometryModel",
    "PocketSample",
]
