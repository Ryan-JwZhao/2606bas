from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..calibration.service import CalibrationService
from ..config import PlannerConfig
from ..route_geometry import polygon_contains_with_margin, segment_inside_polygon, segment_inside_polygon_to_pocket
from ..schemas import ShotCandidate
from ..utils import angle_deg, clamp, point_segment_distance, unit
from .pocket_clearance import assess_pocket_entry
from .pocket_targets import planning_pocket_mouth, planning_pocket_points


RAILS = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class _Rect:
    left: float
    right: float
    top: float
    bottom: float


class RailAssistedShotPlanner:
    """Conservative fallback for an object ball constrained by one cushion.

    The interface deliberately returns ordinary ``ShotCandidate`` objects, so
    rule, manual-target and cue-held planning share exactly the same safety
    checks.  It is consulted only when the caller found no ordinary route for
    that target.
    """

    version = "rail_assisted_funnel_v1"

    def __init__(self, config: PlannerConfig, calibration: CalibrationService):
        self.config = config
        self.calibration = calibration
        self.last_status = "idle"

    def candidates(
        self,
        *,
        cue_ball: object,
        target: object,
        balls: Sequence[object],
    ) -> list[ShotCandidate]:
        if not bool(getattr(self.config, "rail_assist_enabled", True)):
            self.last_status = "disabled"
            return []
        target_center = self._center(target)
        rect = self._center_rect()
        rail, rail_distance = self._nearest_rail(target_center, rect)
        maximum_distance = self._maximum_rail_distance()
        if rail_distance > maximum_distance:
            self.last_status = f"not_rail_adjacent:{rail_distance:.1f}>{maximum_distance:.1f}"
            return []

        routes: list[ShotCandidate] = []
        for pocket_index, pocket_value in enumerate(planning_pocket_points(self.calibration.table)):
            pocket = np.asarray(pocket_value, dtype=np.float32).reshape((2,))
            if not self._is_adjacent_corner_pocket(pocket, rail, rect):
                continue
            candidate = self._candidate(
                cue_ball=cue_ball,
                target=target,
                balls=balls,
                rail=rail,
                rail_distance_mm=rail_distance,
                pocket_index=pocket_index,
                pocket=pocket,
            )
            if candidate is not None:
                routes.append(candidate)
        routes.sort(key=lambda item: (float(item.score), -float(item.risk)), reverse=True)
        self.last_status = f"ok:{rail}:{len(routes)}" if routes else f"no_safe_funnel:{rail}"
        return routes

    def _candidate(
        self,
        *,
        cue_ball: object,
        target: object,
        balls: Sequence[object],
        rail: str,
        rail_distance_mm: float,
        pocket_index: int,
        pocket: np.ndarray,
    ) -> ShotCandidate | None:
        mouth_value = planning_pocket_mouth(self.calibration.table, pocket_index)
        if mouth_value is None or len(mouth_value) != 2:
            return None
        jaw_a = np.asarray(mouth_value[0], dtype=np.float32).reshape((2,))
        jaw_b = np.asarray(mouth_value[1], dtype=np.float32).reshape((2,))
        mouth_midpoint = 0.5 * (jaw_a + jaw_b)
        outward = unit(pocket - mouth_midpoint)
        if float(np.linalg.norm(outward)) <= 1.0e-6:
            return None

        diameter = max(1.0, float(self.calibration.table.ball_diameter_mm))
        cue_radius = self._radius(cue_ball)
        target_radius = self._radius(target)
        target_center = self._center(target)
        cue_center = self._center(cue_ball)
        approach_distance = 1.5 * diameter
        funnel_point = mouth_midpoint - outward * approach_distance
        first_direction = unit(funnel_point - target_center)
        final_direction = unit(pocket - funnel_point)
        if float(np.linalg.norm(first_direction)) <= 1.0e-6 or float(np.linalg.norm(final_direction)) <= 1.0e-6:
            return None

        alignment = self._rail_alignment_angle(first_direction, rail)
        max_alignment = max(1.0, float(getattr(self.config, "rail_assist_max_alignment_angle_deg", 18.0)))
        if alignment > max_alignment:
            return None
        deflection = float(angle_deg(first_direction, final_direction))
        max_deflection = max(1.0, float(getattr(self.config, "rail_assist_max_deflection_deg", 60.0)))
        if deflection > max_deflection:
            return None

        safety_margin = max(0.0, float(getattr(self.config, "pocket_entry_safety_margin_mm", 2.0)))
        entry = assess_pocket_entry(
            funnel_point,
            pocket,
            mouth_value,
            ball_radius_mm=0.5 * diameter,
            safety_margin_mm=safety_margin,
        )
        max_entry_angle = max(0.0, float(getattr(self.config, "pocket_entry_max_angle_deg", 50.0)))
        if not entry.feasible or entry.entrance_angle_deg > max_entry_angle:
            return None

        ghost = target_center - first_direction * float(cue_radius + target_radius)
        polygon = self._center_polygon()
        if polygon.shape[0] >= 3:
            margin = max(0.0, float(self.config.collision_padding_mm))
            if not polygon_contains_with_margin(polygon, ghost, margin_mm=margin):
                return None
            if not segment_inside_polygon(polygon, cue_center, ghost, margin_mm=margin):
                return None
            if not segment_inside_polygon(polygon, target_center, funnel_point, margin_mm=0.0):
                return None
            if not segment_inside_polygon_to_pocket(
                polygon,
                funnel_point,
                pocket,
                margin_mm=0.0,
                pocket_relief_mm=3.25 * diameter,
            ):
                return None

        cue_clearance = self._path_clearance(
            cue_center,
            ghost,
            balls,
            ignore={self._track_id(cue_ball), self._track_id(target)},
            moving_radius=cue_radius,
        )
        first_clearance = self._path_clearance(
            target_center,
            funnel_point,
            balls,
            ignore={self._track_id(cue_ball), self._track_id(target)},
            moving_radius=target_radius,
        )
        final_clearance = self._path_clearance(
            funnel_point,
            pocket,
            balls,
            ignore={self._track_id(cue_ball), self._track_id(target)},
            moving_radius=target_radius,
        )
        object_clearance = min(first_clearance, final_clearance)
        required_cue = float(self.config.cue_path_margin_mm) + float(self.config.collision_padding_mm)
        required_object = float(self.config.object_path_margin_mm) + float(self.config.collision_padding_mm)
        if cue_clearance < required_cue or object_clearance < required_object:
            return None

        cue_vector = ghost - cue_center
        cue_distance = float(np.linalg.norm(cue_vector))
        if cue_distance < 1.0:
            return None
        cut = float(angle_deg(unit(cue_vector), first_direction))
        if cut > float(self.config.max_cut_angle_deg):
            return None
        object_distance = float(np.linalg.norm(funnel_point - target_center) + np.linalg.norm(pocket - funnel_point))
        table_diag = max(1.0, math.hypot(float(self.calibration.table.width_mm), float(self.calibration.table.height_mm)))
        cut_penalty = (cut / max(1.0, float(self.config.max_cut_angle_deg))) ** 1.8
        distance_penalty = (cue_distance + 0.65 * object_distance) / table_diag
        deflection_penalty = deflection / max(1.0, max_deflection)
        score = float(1.35 - 1.0 * cut_penalty - 0.65 * distance_penalty - 0.45 * deflection_penalty)
        risk = max(
            0.78,
            clamp(0.40 + 0.25 * cut_penalty + 0.20 * distance_penalty + 0.35 * deflection_penalty, 0.0, 1.0),
        )
        target_id = self._track_id(target)
        group = str(getattr(target, "group"))
        return ShotCandidate(
            candidate_id=f"rail_t{target_id}_p{pocket_index}_{rail}_{int(object_distance)}",
            cue_track_id=self._track_id(cue_ball),
            target_track_id=target_id,
            target_group=group,
            pocket_index=int(pocket_index),
            cue_ball=self._point(cue_center),
            object_ball=self._point(target_center),
            ghost_ball=self._point(ghost),
            pocket_point=self._point(pocket),
            aim_line=[self._point(cue_center), self._point(ghost)],
            object_line=[self._point(target_center), self._point(funnel_point), self._point(pocket)],
            cut_angle_deg=cut,
            cue_distance_mm=cue_distance,
            object_distance_mm=object_distance,
            score=score,
            risk=float(risk),
            explanation={
                "rail_assisted": True,
                "rail_assist_version": self.version,
                "rail_assist_rail": rail,
                "rail_assist_center_distance_mm": float(rail_distance_mm),
                "rail_assist_alignment_deg": float(alignment),
                "rail_assist_deflection_deg": float(deflection),
                "cue_clearance_mm": float(cue_clearance),
                "object_clearance_mm": float(object_clearance),
                "pocket_entry_standard": "full_ball_corridor_v2+rail_funnel_v1",
                "pocket_entry_angle_deg": float(entry.entrance_angle_deg),
                "pocket_jaw_clearance_mm": float(entry.jaw_clearance_mm),
                "pocket_required_clearance_mm": float(entry.required_clearance_mm),
                "pocket_clearance_margin_mm": float(entry.clearance_margin_mm),
            },
        )

    def _maximum_rail_distance(self) -> float:
        configured = getattr(self.config, "rail_assist_max_center_distance_mm", None)
        if configured is not None:
            return max(1.0, float(configured))
        return 1.15 * max(1.0, float(self.calibration.table.ball_diameter_mm))

    def _center_polygon(self) -> np.ndarray:
        value = self.calibration.table.center_playable_polygon_mm or self.calibration.table.inner_polygon_mm
        return np.asarray(value, dtype=np.float32).reshape((-1, 2)) if value else np.empty((0, 2), dtype=np.float32)

    def _center_rect(self) -> _Rect:
        polygon = self._center_polygon()
        if polygon.shape[0] >= 3:
            return _Rect(
                left=float(np.min(polygon[:, 0])),
                right=float(np.max(polygon[:, 0])),
                top=float(np.min(polygon[:, 1])),
                bottom=float(np.max(polygon[:, 1])),
            )
        return _Rect(0.0, float(self.calibration.table.width_mm), 0.0, float(self.calibration.table.height_mm))

    @staticmethod
    def _nearest_rail(point: np.ndarray, rect: _Rect) -> tuple[str, float]:
        distances = {
            "left": abs(float(point[0]) - rect.left),
            "right": abs(rect.right - float(point[0])),
            "top": abs(float(point[1]) - rect.top),
            "bottom": abs(rect.bottom - float(point[1])),
        }
        return min(distances.items(), key=lambda item: item[1])

    @staticmethod
    def _is_adjacent_corner_pocket(pocket: np.ndarray, rail: str, rect: _Rect) -> bool:
        x = float(pocket[0])
        y = float(pocket[1])
        width = max(1.0, rect.right - rect.left)
        height = max(1.0, rect.bottom - rect.top)
        near_left = x <= rect.left + 0.25 * width
        near_right = x >= rect.right - 0.25 * width
        near_top = y <= rect.top + 0.25 * height
        near_bottom = y >= rect.bottom - 0.25 * height
        if rail == "top":
            return near_top and (near_left or near_right)
        if rail == "bottom":
            return near_bottom and (near_left or near_right)
        if rail == "left":
            return near_left and (near_top or near_bottom)
        return near_right and (near_top or near_bottom)

    @staticmethod
    def _rail_alignment_angle(direction: np.ndarray, rail: str) -> float:
        tangent = np.asarray((0.0, 1.0) if rail in {"left", "right"} else (1.0, 0.0), dtype=np.float32)
        return min(float(angle_deg(direction, tangent)), float(angle_deg(direction, -tangent)))

    def _path_clearance(
        self,
        start: np.ndarray,
        end: np.ndarray,
        balls: Sequence[object],
        *,
        ignore: set[int],
        moving_radius: float,
    ) -> float:
        minimum = float("inf")
        for ball in balls:
            if self._track_id(ball) in ignore:
                continue
            center = np.asarray(
                getattr(ball, "clearance_center_mm", getattr(ball, "center_mm")),
                dtype=np.float32,
            ).reshape((2,))
            clearance = point_segment_distance(center, start, end) - self._radius(ball) - moving_radius
            minimum = min(minimum, float(clearance))
        return minimum if np.isfinite(minimum) else 9999.0

    @staticmethod
    def _center(ball: object) -> np.ndarray:
        return np.asarray(getattr(ball, "center_mm"), dtype=np.float32).reshape((2,))

    def _radius(self, ball: object) -> float:
        return max(1.0, float(getattr(ball, "radius_mm", 0.5 * self.calibration.table.ball_diameter_mm)))

    @staticmethod
    def _track_id(ball: object) -> int:
        return int(getattr(ball, "track_id"))

    @staticmethod
    def _point(value: np.ndarray) -> tuple[float, float]:
        point = np.asarray(value, dtype=np.float32).reshape((2,))
        return (float(point[0]), float(point[1]))


__all__ = ["RailAssistedShotPlanner"]
