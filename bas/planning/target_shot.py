from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..calibration.service import CalibrationService
from ..config import PlannerConfig
from ..route_geometry import segment_inside_polygon_to_pocket
from ..schemas import MatchStateFrame, ShotCandidate, TrackObservation
from ..utils import angle_deg, clamp, point_segment_distance, unit
from .aim_context import PlannerAimFrameContext
from .corridor_targeting import rank_object_balls_in_corridor
from .cue_aim import CueStickAimDetector
from .pocket_clearance import assess_pocket_entry
from .pocket_targets import planning_pocket_mouth, planning_pocket_points


OBJECT_GROUPS = {"solid", "stripe", "black"}
RELEASE_PHASES = {"SHOT_ACTIVE", "SETTLING", "TURN_RESOLVE"}
RAILS = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class TargetShotDecision:
    active_target_id: Optional[int]
    active_group: Optional[str]
    status: str
    candidate_target_id: Optional[int] = None
    pending_target_id: Optional[int] = None
    active: bool = False
    switched: bool = False


@dataclass(frozen=True)
class _PointedBall:
    track_id: int
    group: str
    score: float


@dataclass(frozen=True)
class _Route:
    pocket_index: int
    pocket: np.ndarray
    ghost: np.ndarray
    object_points: list[np.ndarray]
    rebounds: int
    cue_distance_mm: float
    object_distance_mm: float
    cut_angle_deg: float
    score: float
    risk: float
    clearance_mm: float
    pocket_entry_angle_deg: float
    pocket_jaw_clearance_mm: float
    pocket_required_clearance_mm: float
    pocket_clearance_margin_mm: float


@dataclass(frozen=True)
class _Rect:
    left: float
    right: float
    top: float
    bottom: float


class TargetShotModeController:
    version = "target_shot_mode_v1"

    def __init__(self, config: PlannerConfig, aim_detector: CueStickAimDetector | None = None):
        self.config = config
        self.aim_detector = aim_detector or CueStickAimDetector()
        self.last_status = "off"
        self.reset()

    def reset(self) -> None:
        self._active_target_id: Optional[int] = None
        self._active_group: Optional[str] = None
        self._pending_target_id: Optional[int] = None
        self._pending_group: Optional[str] = None
        self._pending_hold_ms = 0.0
        self._pending_miss_ms = 0.0
        self._release_phase_ms = 0.0
        self._last_update_ts_ns: Optional[int] = None
        self.last_status = "off"

    def update(
        self,
        *,
        state: MatchStateFrame,
        balls: Sequence[object],
        tracks: Sequence[TrackObservation],
        frame_bgr: Optional[np.ndarray] = None,
        frame_context: PlannerAimFrameContext | None = None,
    ) -> TargetShotDecision:
        if not bool(getattr(self.config, "target_shot_enabled", True)):
            self.reset()
            return self._decision("disabled")

        elapsed_ms = self._elapsed_ms(getattr(state, "ts_cam_ns", 0))
        phase = str(getattr(state, "phase", "") or "").strip().upper()
        if phase in RELEASE_PHASES:
            return self._handle_release_phase(elapsed_ms)

        self._release_phase_ms = 0.0

        pointed = self._pointed_ball(
            balls,
            tracks,
            frame_bgr=frame_bgr,
            frame_context=frame_context,
        )
        if pointed is None:
            return self._handle_non_object(
                elapsed_ms,
                active_status="active_hold_no_aim",
                inactive_status="inactive_no_aim",
            )

        if pointed.group == "cue":
            return self._handle_non_object(
                elapsed_ms,
                active_status="active_hold_cue_aim",
                inactive_status="normal_cue_aim",
            )

        if self._active_target_id is not None and int(pointed.track_id) == int(self._active_target_id):
            self._clear_pending()
            return self._decision("active_hold_same", candidate_target_id=pointed.track_id)

        return self._advance_pending(pointed, elapsed_ms)

    def _advance_pending(self, pointed: _PointedBall, elapsed_ms: float) -> TargetShotDecision:
        if self._pending_target_id == pointed.track_id:
            self._pending_hold_ms += max(0.0, float(elapsed_ms))
            self._pending_miss_ms = 0.0
        else:
            self._pending_target_id = pointed.track_id
            self._pending_group = str(pointed.group)
            self._pending_hold_ms = 0.0
            self._pending_miss_ms = 0.0

        hold_ms = self._switch_hold_ms() if self._active_target_id is not None else self._activate_hold_ms()
        if self._pending_hold_ms < hold_ms:
            return self._decision(
                self._pending_status(),
                candidate_target_id=pointed.track_id,
                pending_target_id=pointed.track_id,
            )

        switched = self._active_target_id is not None and int(self._active_target_id) != int(pointed.track_id)
        self._active_target_id = int(pointed.track_id)
        self._active_group = str(pointed.group)
        self._clear_pending()
        return self._decision(
            "switch_commit" if switched else "active_new",
            candidate_target_id=pointed.track_id,
            switched=switched,
        )

    def _handle_release_phase(self, elapsed_ms: float) -> TargetShotDecision:
        if self._active_target_id is None:
            self._clear_pending()
            self._release_phase_ms = 0.0
            return self._decision("inactive_motion")
        self._clear_pending()
        self._release_phase_ms += max(0.0, float(elapsed_ms))
        confirm_ms = self._release_confirm_ms()
        if self._release_phase_ms < confirm_ms:
            return self._decision(
                f"release_pending {self._status_ms(self._release_phase_ms)}/{confirm_ms}ms",
                candidate_target_id=self._active_target_id,
            )
        self.reset()
        return self._decision("released_by_shot")

    def _handle_non_object(
        self,
        elapsed_ms: float,
        *,
        active_status: str,
        inactive_status: str,
    ) -> TargetShotDecision:
        pending_status = self._advance_pending_grace(elapsed_ms)
        if pending_status is not None:
            return self._decision(
                pending_status,
                pending_target_id=self._pending_target_id,
            )
        return self._decision(active_status if self._active_target_id is not None else inactive_status)

    def _advance_pending_grace(self, elapsed_ms: float) -> Optional[str]:
        if self._pending_target_id is None:
            return None
        self._pending_miss_ms += max(0.0, float(elapsed_ms))
        grace_ms = self._miss_grace_ms()
        if self._pending_miss_ms > grace_ms:
            self._clear_pending()
            return None
        return self._pending_status()

    def _pending_status(self) -> str:
        prefix = "switch" if self._active_target_id is not None else "activate"
        hold_target_ms = self._switch_hold_ms() if self._active_target_id is not None else self._activate_hold_ms()
        status = f"{prefix}_pending {self._status_ms(self._pending_hold_ms)}/{hold_target_ms}ms"
        if self._pending_miss_ms > 0.0:
            grace_ms = self._miss_grace_ms()
            status += f" grace {self._status_ms(self._pending_miss_ms)}/{grace_ms}ms"
        return status

    def _activate_hold_ms(self) -> int:
        return max(1, int(getattr(self.config, "target_shot_activate_hold_ms", 1000) or 1000))

    def _switch_hold_ms(self) -> int:
        return max(1, int(getattr(self.config, "target_shot_switch_hold_ms", 1500) or 1500))

    def _miss_grace_ms(self) -> int:
        return max(0, int(getattr(self.config, "target_shot_miss_grace_ms", 200) or 0))

    def _release_confirm_ms(self) -> int:
        return max(0, int(getattr(self.config, "target_shot_release_confirm_ms", 300) or 0))

    def _elapsed_ms(self, ts_cam_ns: int) -> float:
        current = int(ts_cam_ns or 0)
        previous = self._last_update_ts_ns
        self._last_update_ts_ns = current
        if previous is None or current <= previous:
            return 0.0
        return max(0.0, (current - previous) / 1_000_000.0)

    @staticmethod
    def _status_ms(value: float) -> int:
        return max(0, int(round(float(value))))

    def _pointed_ball(
        self,
        balls: Sequence[object],
        tracks: Sequence[TrackObservation],
        *,
        frame_bgr: Optional[np.ndarray],
        frame_context: PlannerAimFrameContext | None = None,
    ) -> _PointedBall | None:
        min_quality = float(getattr(self.config, "target_shot_min_stick_quality", 0.25))
        cue_ball = next(
            (
                ball
                for ball in balls
                if str(getattr(ball, "group", "")).strip().lower() == "cue"
                and float(getattr(ball, "quality", 0.0)) > 0.25
            ),
            None,
        )
        if cue_ball is None:
            return None
        aim = self._shared_or_local_aim(
            cue_ball=cue_ball,
            tracks=tracks,
            frame_bgr=frame_bgr,
            frame_context=frame_context,
            min_stick_quality=min_quality,
        )
        if aim is None:
            return None
        direction = unit(np.asarray(aim.direction_px, dtype=np.float32).reshape((2,)))
        if float(np.linalg.norm(direction)) < 1.0e-6:
            return None
        half_width = 0.5 * max(1.0, float(getattr(self.config, "cue_sector_corridor_width_px", 140.0)))
        ranked = rank_object_balls_in_corridor(
            cue_ball=cue_ball,
            balls=balls,
            direction_px=direction,
            half_width_px=half_width,
        )
        if ranked:
            return _PointedBall(
                track_id=int(ranked[0].track_id),
                group=str(ranked[0].group),
                score=float(aim.score),
            )
        return _PointedBall(
            track_id=int(getattr(cue_ball, "track_id")),
            group="cue",
            score=float(aim.score),
        )

    def _shared_or_local_aim(
        self,
        *,
        cue_ball: object,
        tracks: Sequence[TrackObservation],
        frame_bgr: Optional[np.ndarray],
        frame_context: PlannerAimFrameContext | None,
        min_stick_quality: float,
    ):
        if frame_context is not None:
            aim = frame_context.shared_aim()
        else:
            aim = self.aim_detector.detect(
                frame_bgr=frame_bgr,
                tracks=tracks,
                cue_center_px=np.asarray(getattr(cue_ball, "center_px"), dtype=np.float32).reshape((2,)),
                cue_radius_px=max(2.0, float(getattr(cue_ball, "radius_px", 0.0))),
                inner_polygon_px=None,
                min_stick_quality=min_stick_quality,
                prefer_tracks=False,
                allow_edge_detection=True,
            )
        if aim is None:
            return None
        if str(getattr(aim, "source", "")).strip().lower() != "track_bbox":
            return aim
        track_quality = getattr(aim, "track_quality", None)
        if track_quality is None or float(track_quality) >= float(min_stick_quality):
            return aim
        return None

    def _clear_pending(self) -> None:
        self._pending_target_id = None
        self._pending_group = None
        self._pending_hold_ms = 0.0
        self._pending_miss_ms = 0.0

    def _decision(
        self,
        status: str,
        *,
        candidate_target_id: Optional[int] = None,
        pending_target_id: Optional[int] = None,
        switched: bool = False,
    ) -> TargetShotDecision:
        self.last_status = str(status)
        return TargetShotDecision(
            active_target_id=self._active_target_id,
            active_group=self._active_group,
            status=self.last_status,
            candidate_target_id=candidate_target_id,
            pending_target_id=pending_target_id,
            active=self._active_target_id is not None,
            switched=bool(switched),
        )


class TargetShotPlanner:
    version = "target_shot_route_v1"

    def __init__(self, config: PlannerConfig, calibration: CalibrationService):
        self.config = config
        self.calibration = calibration
        self.last_status = "idle"

    def plan(
        self,
        *,
        cue_ball: object,
        target: object,
        balls: Sequence[object],
        decision: TargetShotDecision,
    ) -> ShotCandidate | None:
        routes = self._routes(cue_ball=cue_ball, target=target, balls=balls)
        if not routes:
            self.last_status = "no_theoretical_route"
            return None
        best = max(routes, key=lambda route: route.score)
        self.last_status = f"ok:{best.rebounds}_rebounds"
        return self._candidate(cue_ball=cue_ball, target=target, route=best, decision=decision)

    def _routes(self, *, cue_ball: object, target: object, balls: Sequence[object]) -> list[_Route]:
        max_rebounds = max(0, int(getattr(self.config, "target_shot_max_rebounds", 2)))
        pockets = [
            np.asarray(p, dtype=np.float32).reshape((2,))
            for p in planning_pocket_points(self.calibration.table)
        ]
        routes: list[_Route] = []
        for rebound_count in range(max_rebounds + 1):
            for rails in _rail_sequences(rebound_count):
                for pocket_index, pocket in enumerate(pockets):
                    route = self._route_for(cue_ball, target, balls, pocket, pocket_index, rails)
                    if route is not None:
                        routes.append(route)
        return routes

    def _route_for(
        self,
        cue_ball: object,
        target: object,
        balls: Sequence[object],
        pocket: np.ndarray,
        pocket_index: int,
        rails: tuple[str, ...],
    ) -> _Route | None:
        cue = np.asarray(getattr(cue_ball, "center_mm"), dtype=np.float32).reshape((2,))
        target_center = np.asarray(getattr(target, "center_mm"), dtype=np.float32).reshape((2,))
        cue_radius = float(max(1.0, getattr(cue_ball, "radius_mm", 0.5 * self.calibration.table.ball_diameter_mm)))
        target_radius = float(max(1.0, getattr(target, "radius_mm", 0.5 * self.calibration.table.ball_diameter_mm)))
        motion = self._motion_rect(target_radius)

        object_points = self._object_path(target_center, pocket, rails, motion)
        if object_points is None or len(object_points) < 2:
            return None
        entry_assessment = assess_pocket_entry(
            object_points[-2],
            pocket,
            planning_pocket_mouth(self.calibration.table, pocket_index),
            ball_radius_mm=0.5 * float(self.calibration.table.ball_diameter_mm),
            safety_margin_mm=float(self.config.object_path_margin_mm) + float(self.config.collision_padding_mm),
        )
        if not entry_assessment.feasible:
            return None

        first_dir = unit(object_points[1] - target_center)
        if float(np.linalg.norm(first_dir)) < 1e-6:
            return None
        ghost = target_center - first_dir * float(cue_radius + target_radius)
        if not self._point_in_rect(ghost, self._table_rect(), margin=max(2.0, float(self.config.collision_padding_mm))):
            return None

        cue_vec = ghost - cue
        cue_distance = float(np.linalg.norm(cue_vec))
        if cue_distance < 1.0:
            return None
        cut = angle_deg(unit(cue_vec), first_dir)
        if cut > float(self.config.max_cut_angle_deg):
            return None

        cue_clearance = self._path_clearance(cue, ghost, balls, ignore={int(getattr(cue_ball, "track_id")), int(getattr(target, "track_id"))}, moving_radius=cue_radius)
        object_clearance = self._object_path_clearance(object_points, balls, ignore={int(getattr(cue_ball, "track_id")), int(getattr(target, "track_id"))}, moving_radius=target_radius)
        required_cue = float(self.config.cue_path_margin_mm) + float(self.config.collision_padding_mm)
        required_object = float(self.config.object_path_margin_mm) + float(self.config.collision_padding_mm)
        if cue_clearance < required_cue or object_clearance < required_object:
            return None

        object_distance = float(sum(np.linalg.norm(b - a) for a, b in zip(object_points, object_points[1:])))
        table_diag = max(1.0, math.hypot(float(self.calibration.table.width_mm), float(self.calibration.table.height_mm)))
        cut_penalty = (cut / max(1.0, float(self.config.max_cut_angle_deg))) ** 1.8
        distance_penalty = (cue_distance + 0.65 * object_distance) / table_diag
        rebound_penalty = 0.18 * len(rails)
        clearance_norm = clamp(min(cue_clearance, object_clearance) / 80.0, 0.0, 1.0)
        pocket_angle_penalty = float(max(0.0, (entry_assessment.entrance_angle_deg - 20.0) / 70.0) ** 1.5)
        score = float(2.6 - 1.25 * cut_penalty - 0.85 * distance_penalty - rebound_penalty - 0.30 * pocket_angle_penalty + 0.25 * clearance_norm)
        risk = clamp(0.45 * cut_penalty + 0.35 * distance_penalty + rebound_penalty + 0.15 * pocket_angle_penalty + 0.20 * (1.0 - clearance_norm), 0.0, 1.0)
        return _Route(
            pocket_index=int(pocket_index),
            pocket=pocket.astype(np.float32),
            ghost=ghost.astype(np.float32),
            object_points=[point.astype(np.float32) for point in object_points],
            rebounds=len(rails),
            cue_distance_mm=cue_distance,
            object_distance_mm=object_distance,
            cut_angle_deg=float(cut),
            score=score,
            risk=float(risk),
            clearance_mm=float(min(cue_clearance, object_clearance, entry_assessment.clearance_margin_mm)),
            pocket_entry_angle_deg=float(entry_assessment.entrance_angle_deg),
            pocket_jaw_clearance_mm=float(entry_assessment.jaw_clearance_mm),
            pocket_required_clearance_mm=float(entry_assessment.required_clearance_mm),
            pocket_clearance_margin_mm=float(entry_assessment.clearance_margin_mm),
        )

    def _object_path(
        self,
        target: np.ndarray,
        pocket: np.ndarray,
        rails: tuple[str, ...],
        rect: _Rect,
    ) -> list[np.ndarray] | None:
        if not rails:
            if not self._direct_segment_reaches_pocket(target, pocket, rect):
                return None
            return [target.astype(np.float32), pocket.astype(np.float32)]

        virtual = pocket.astype(np.float32)
        for rail in reversed(rails):
            virtual = _reflect_point(virtual, rail, rect)
        direction = unit(virtual - target)
        if float(np.linalg.norm(direction)) < 1e-6:
            return None

        origin = target.astype(np.float32)
        points = [target.astype(np.float32)]
        for rail in rails:
            hit = _first_rect_hit(origin, direction, rect, expected_rail=rail)
            if hit is None:
                return None
            hit_point, hit_rail = hit
            if hit_rail != rail:
                return None
            points.append(hit_point.astype(np.float32))
            direction = _reflect_direction(direction, rail)
            origin = (hit_point + direction * 0.5).astype(np.float32)

        if not self._ray_reaches_pocket(origin, direction, pocket, rect):
            return None
        points.append(pocket.astype(np.float32))
        return points

    def _direct_segment_reaches_pocket(self, start: np.ndarray, pocket: np.ndarray, rect: _Rect) -> bool:
        polygon = self._playable_polygon()
        if polygon.shape[0] < 3:
            return self._segment_reaches_pocket(start, pocket, rect)
        return segment_inside_polygon_to_pocket(
            polygon,
            start,
            pocket,
            margin_mm=max(0.0, float(self.config.collision_padding_mm)),
            pocket_relief_mm=max(18.0, 2.0 * float(self.calibration.table.ball_diameter_mm)),
        )

    def _segment_reaches_pocket(self, start: np.ndarray, pocket: np.ndarray, rect: _Rect) -> bool:
        direction = unit(pocket - start)
        if float(np.linalg.norm(direction)) < 1e-6:
            return False
        return self._ray_reaches_pocket(start, direction, pocket, rect)

    def _ray_reaches_pocket(self, origin: np.ndarray, direction: np.ndarray, pocket: np.ndarray, rect: _Rect) -> bool:
        d = unit(direction)
        rel = pocket - origin
        t = float(np.dot(rel, d))
        if t <= 1.0:
            return False
        closest = origin + d * t
        tolerance = max(1.0, float(getattr(self.config, "target_shot_pocket_tolerance_mm", 42.0)))
        if float(np.linalg.norm(closest - pocket)) > tolerance:
            return False
        boundary = _first_rect_hit(origin, d, rect, expected_rail=None)
        if boundary is None:
            return True
        boundary_point, boundary_rail = boundary
        boundary_t = float(np.linalg.norm(boundary[0] - origin))
        if t <= boundary_t + tolerance:
            return True
        return self._pocket_exit_can_feed_pocket(boundary_point, boundary_rail, pocket, rect)

    def _pocket_exit_can_feed_pocket(self, boundary_point: np.ndarray, boundary_rail: str, pocket: np.ndarray, rect: _Rect) -> bool:
        if not self._pocket_on_rail(pocket, boundary_rail, rect):
            return False
        return float(np.linalg.norm(boundary_point - pocket)) <= self._pocket_exit_tolerance()

    def _pocket_on_rail(self, pocket: np.ndarray, rail: str, rect: _Rect) -> bool:
        tolerance = self._pocket_exit_tolerance()
        x = float(pocket[0])
        y = float(pocket[1])
        if rail == "left":
            return x <= rect.left + tolerance
        if rail == "right":
            return x >= rect.right - tolerance
        if rail == "top":
            return y <= rect.top + tolerance
        if rail == "bottom":
            return y >= rect.bottom - tolerance
        return False

    def _pocket_exit_tolerance(self) -> float:
        configured = max(1.0, float(getattr(self.config, "target_shot_pocket_tolerance_mm", 42.0)))
        ball_diameter = max(1.0, float(getattr(self.calibration.table, "ball_diameter_mm", configured)))
        return max(configured, 1.25 * ball_diameter)

    def _motion_rect(self, ball_radius: float) -> _Rect:
        base = self._table_rect()
        inset = max(0.0, float(ball_radius))
        return _Rect(
            left=base.left + inset,
            right=base.right - inset,
            top=base.top + inset,
            bottom=base.bottom - inset,
        )

    def _table_rect(self) -> _Rect:
        polygon = self.calibration.table.center_playable_polygon_mm or self.calibration.table.inner_polygon_mm
        arr = np.asarray(polygon, dtype=np.float32).reshape((-1, 2)) if polygon else np.empty((0, 2), dtype=np.float32)
        if arr.shape[0] >= 3:
            return _Rect(float(np.min(arr[:, 0])), float(np.max(arr[:, 0])), float(np.min(arr[:, 1])), float(np.max(arr[:, 1])))
        return _Rect(0.0, float(self.calibration.table.width_mm), 0.0, float(self.calibration.table.height_mm))

    def _playable_polygon(self) -> np.ndarray:
        polygon = self.calibration.table.center_playable_polygon_mm or self.calibration.table.inner_polygon_mm
        if not polygon:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(polygon, dtype=np.float32).reshape((-1, 2))

    def _point_in_rect(self, point: np.ndarray, rect: _Rect, *, margin: float = 0.0) -> bool:
        return (
            rect.left + margin <= float(point[0]) <= rect.right - margin
            and rect.top + margin <= float(point[1]) <= rect.bottom - margin
        )

    def _path_clearance(
        self,
        a: np.ndarray,
        b: np.ndarray,
        balls: Sequence[object],
        *,
        ignore: set[int],
        moving_radius: float,
    ) -> float:
        min_clearance = float("inf")
        for ball in balls:
            if int(getattr(ball, "track_id")) in ignore:
                continue
            center = np.asarray(getattr(ball, "center_mm"), dtype=np.float32).reshape((2,))
            d = point_segment_distance(center, a, b)
            radius = float(max(1.0, getattr(ball, "radius_mm", 0.5 * self.calibration.table.ball_diameter_mm)))
            min_clearance = min(min_clearance, float(d - radius - moving_radius))
        return min_clearance if np.isfinite(min_clearance) else 9999.0

    def _object_path_clearance(
        self,
        points: Sequence[np.ndarray],
        balls: Sequence[object],
        *,
        ignore: set[int],
        moving_radius: float,
    ) -> float:
        clearances = [
            self._path_clearance(a, b, balls, ignore=ignore, moving_radius=moving_radius)
            for a, b in zip(points, points[1:])
        ]
        return min(clearances) if clearances else 9999.0

    def _candidate(self, *, cue_ball: object, target: object, route: _Route, decision: TargetShotDecision) -> ShotCandidate:
        target_id = int(getattr(target, "track_id"))
        target_group = str(getattr(target, "group"))
        cid = f"target_t{target_id}_p{route.pocket_index}_r{route.rebounds}_{int(route.object_distance_mm)}"
        explanation = {
            "target_shot": True,
            "target_shot_mode_version": TargetShotModeController.version,
            "target_shot_route_version": self.version,
            "target_shot_status": decision.status,
            "target_shot_rebounds": int(route.rebounds),
            "target_shot_clearance_mm": float(route.clearance_mm),
            "pocket_entry_angle_deg": float(route.pocket_entry_angle_deg),
            "pocket_jaw_clearance_mm": float(route.pocket_jaw_clearance_mm),
            "pocket_required_clearance_mm": float(route.pocket_required_clearance_mm),
            "pocket_clearance_margin_mm": float(route.pocket_clearance_margin_mm),
            "target_shot_independent_of_cue_stick": True,
        }
        return ShotCandidate(
            candidate_id=cid,
            cue_track_id=int(getattr(cue_ball, "track_id")),
            target_track_id=target_id,
            target_group=target_group,
            pocket_index=int(route.pocket_index),
            cue_ball=_pt(np.asarray(getattr(cue_ball, "center_mm"), dtype=np.float32)),
            object_ball=_pt(np.asarray(getattr(target, "center_mm"), dtype=np.float32)),
            ghost_ball=_pt(route.ghost),
            pocket_point=_pt(route.pocket),
            aim_line=[_pt(np.asarray(getattr(cue_ball, "center_mm"), dtype=np.float32)), _pt(route.ghost)],
            object_line=[_pt(point) for point in route.object_points],
            cut_angle_deg=float(route.cut_angle_deg),
            cue_distance_mm=float(route.cue_distance_mm),
            object_distance_mm=float(route.object_distance_mm),
            score=float(route.score),
            risk=float(route.risk),
            explanation=explanation,
        )


def _rail_sequences(count: int) -> list[tuple[str, ...]]:
    if count <= 0:
        return [()]
    sequences: list[tuple[str, ...]] = []
    for rails in itertools.product(RAILS, repeat=count):
        if any(rails[idx] == rails[idx - 1] for idx in range(1, len(rails))):
            continue
        sequences.append(tuple(rails))
    return sequences


def _reflect_point(point: np.ndarray, rail: str, rect: _Rect) -> np.ndarray:
    p = np.asarray(point, dtype=np.float32).copy()
    if rail == "left":
        p[0] = float(2.0 * rect.left - p[0])
    elif rail == "right":
        p[0] = float(2.0 * rect.right - p[0])
    elif rail == "top":
        p[1] = float(2.0 * rect.top - p[1])
    elif rail == "bottom":
        p[1] = float(2.0 * rect.bottom - p[1])
    return p.astype(np.float32)


def _reflect_direction(direction: np.ndarray, rail: str) -> np.ndarray:
    d = unit(direction).astype(np.float32)
    if rail in {"left", "right"}:
        d[0] = -d[0]
    elif rail in {"top", "bottom"}:
        d[1] = -d[1]
    return unit(d).astype(np.float32)


def _first_rect_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    rect: _Rect,
    *,
    expected_rail: str | None,
) -> tuple[np.ndarray, str] | None:
    o = np.asarray(origin, dtype=np.float32)
    d = unit(np.asarray(direction, dtype=np.float32))
    candidates: list[tuple[float, np.ndarray, str]] = []
    eps = 1e-5
    if abs(float(d[0])) > eps:
        for x, rail in ((rect.left, "left"), (rect.right, "right")):
            t = (float(x) - float(o[0])) / float(d[0])
            if t <= 0.5:
                continue
            p = o + d * float(t)
            if rect.top - 1e-3 <= float(p[1]) <= rect.bottom + 1e-3:
                candidates.append((float(t), p.astype(np.float32), rail))
    if abs(float(d[1])) > eps:
        for y, rail in ((rect.top, "top"), (rect.bottom, "bottom")):
            t = (float(y) - float(o[1])) / float(d[1])
            if t <= 0.5:
                continue
            p = o + d * float(t)
            if rect.left - 1e-3 <= float(p[0]) <= rect.right + 1e-3:
                candidates.append((float(t), p.astype(np.float32), rail))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    first_t = candidates[0][0]
    near = [item for item in candidates if abs(item[0] - first_t) <= 1e-3]
    if expected_rail is not None:
        for _t, point, rail in near:
            if rail == expected_rail:
                return point, rail
        return None
    _t, point, rail = near[0]
    return point, rail


def _pt(arr: np.ndarray) -> tuple[float, float]:
    point = np.asarray(arr, dtype=np.float32).reshape((2,))
    return (float(point[0]), float(point[1]))
