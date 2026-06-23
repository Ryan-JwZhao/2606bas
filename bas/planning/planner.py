from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..calibration.service import CalibrationService
from ..config import PlannerConfig
from ..schemas import MatchStateFrame, Point, ShotCandidate, ShotPlan, TrackObservation
from ..utils import angle_deg, clamp, point_segment_distance, unit, wall_time_id
from .learning import DisabledLearningRanker


@dataclass
class _Ball:
    track_id: int
    group: str
    center_mm: np.ndarray
    radius_mm: float
    quality: float


class GeometryPhysicsPlanner:
    version = "geometry_physics_mvp_v1"

    def __init__(self, config: PlannerConfig, calibration: CalibrationService):
        self.config = config
        self.calibration = calibration
        self.learning_ranker = DisabledLearningRanker()

    def plan(self, state: MatchStateFrame) -> ShotPlan:
        if not self.config.enabled:
            return ShotPlan(plan_id=f"plan_{state.frame_id}_{wall_time_id()}", frame_id=state.frame_id, ts_cam_ns=state.ts_cam_ns)
        balls = self._extract_balls(state.layout)
        cue = next((b for b in balls if b.group == "cue"), None)
        if cue is None:
            return ShotPlan(plan_id=f"plan_{state.frame_id}_{wall_time_id()}", frame_id=state.frame_id, ts_cam_ns=state.ts_cam_ns)
        targets = [b for b in balls if b.group in {"solid", "stripe", "black"} and b.quality > 0.25]
        if not targets:
            return ShotPlan(plan_id=f"plan_{state.frame_id}_{wall_time_id()}", frame_id=state.frame_id, ts_cam_ns=state.ts_cam_ns)
        candidates: List[ShotCandidate] = []
        pockets = [np.asarray(p, dtype=np.float32) for p in self.calibration.table.pockets_mm]
        inner = np.asarray(self.calibration.table.inner_polygon_mm, dtype=np.float32)
        for target in targets:
            for pocket_index, pocket in enumerate(pockets):
                candidate = self._candidate(cue, target, pocket, pocket_index, balls, inner)
                if candidate is not None:
                    candidates.append(candidate)
        candidates = self.learning_ranker.rerank(candidates)[: max(1, int(self.config.top_k))]
        best = candidates[0] if candidates else None
        return ShotPlan(
            plan_id=f"plan_{state.frame_id}_{wall_time_id()}",
            frame_id=state.frame_id,
            ts_cam_ns=state.ts_cam_ns,
            candidates=candidates,
            best=best,
            planner_version=self.version,
        )

    def _extract_balls(self, tracks: Sequence[TrackObservation]) -> List[_Ball]:
        balls: List[_Ball] = []
        for tr in tracks:
            if tr.group not in {"cue", "solid", "stripe", "black"}:
                continue
            center = self.calibration.camera_px_to_table_mm(np.asarray([tr.center_px], dtype=np.float32))[0]
            radius_mm = self.calibration.pixel_radius_to_mm(tr.center_px, tr.radius_px)
            if radius_mm <= 1.0 or radius_mm > 80.0:
                radius_mm = 0.5 * self.calibration.table.ball_diameter_mm
            balls.append(
                _Ball(
                    track_id=tr.track_id,
                    group=tr.group,
                    center_mm=center.astype(np.float32),
                    radius_mm=float(radius_mm),
                    quality=float(tr.quality),
                )
            )
        return balls

    def _candidate(
        self,
        cue: _Ball,
        target: _Ball,
        pocket: np.ndarray,
        pocket_index: int,
        balls: List[_Ball],
        inner_polygon: np.ndarray,
    ) -> Optional[ShotCandidate]:
        obj_vec = pocket - target.center_mm
        obj_dist = float(np.linalg.norm(obj_vec))
        if obj_dist < 1.0:
            return None
        obj_dir = unit(obj_vec)
        contact_dist = cue.radius_mm + target.radius_mm
        ghost = target.center_mm - obj_dir * contact_dist
        if not self._inside(inner_polygon, ghost, margin_mm=max(2.0, self.config.collision_padding_mm)):
            return None
        cue_vec = ghost - cue.center_mm
        cue_dist = float(np.linalg.norm(cue_vec))
        if cue_dist < 1.0:
            return None
        cue_dir = unit(cue_vec)
        cut = angle_deg(cue_dir, obj_dir)
        if cut > self.config.max_cut_angle_deg:
            return None
        if not self._segment_inside(inner_polygon, cue.center_mm, ghost, margin_mm=max(0.0, self.config.collision_padding_mm)):
            return None
        if not self._segment_inside_to_pocket(inner_polygon, target.center_mm, pocket, margin_mm=max(0.0, self.config.collision_padding_mm)):
            return None
        cue_clearance = self._path_clearance(
            cue.center_mm,
            ghost,
            balls,
            ignore={cue.track_id, target.track_id},
            moving_radius=cue.radius_mm,
        )
        obj_clearance = self._path_clearance(
            target.center_mm,
            pocket,
            balls,
            ignore={cue.track_id, target.track_id},
            moving_radius=target.radius_mm,
        )
        required_cue = self.config.cue_path_margin_mm + self.config.collision_padding_mm
        required_obj = self.config.object_path_margin_mm + self.config.collision_padding_mm
        if cue_clearance < required_cue or obj_clearance < required_obj:
            return None

        table_diag = math.hypot(self.calibration.table.width_mm, self.calibration.table.height_mm)
        cut_penalty = (cut / max(1.0, self.config.max_cut_angle_deg)) ** 1.8
        dist_penalty = (cue_dist + 0.55 * obj_dist) / max(1.0, table_diag)
        clearance_norm = clamp(min(cue_clearance / 80.0, obj_clearance / 80.0), 0.0, 1.0)
        pocket_angle_penalty = self._pocket_angle_penalty(target.center_mm, pocket, obj_dir)
        risk = clamp(0.45 * cut_penalty + 0.35 * dist_penalty + 0.20 * (1.0 - clearance_norm), 0.0, 1.0)
        score = float(2.0 - 1.3 * cut_penalty - 0.9 * dist_penalty - 0.35 * pocket_angle_penalty + 0.25 * clearance_norm)

        cid = f"f{target.track_id}_p{pocket_index}_{int(cue_dist)}"
        return ShotCandidate(
            candidate_id=cid,
            cue_track_id=cue.track_id,
            target_track_id=target.track_id,
            target_group=target.group,
            pocket_index=pocket_index,
            cue_ball=_pt(cue.center_mm),
            object_ball=_pt(target.center_mm),
            ghost_ball=_pt(ghost),
            pocket_point=_pt(pocket),
            aim_line=[_pt(cue.center_mm), _pt(ghost)],
            object_line=[_pt(target.center_mm), _pt(pocket)],
            cut_angle_deg=float(cut),
            cue_distance_mm=float(cue_dist),
            object_distance_mm=float(obj_dist),
            score=score,
            risk=float(risk),
            explanation={
                "cue_clearance_mm": float(cue_clearance),
                "object_clearance_mm": float(obj_clearance),
                "cut_penalty": float(cut_penalty),
                "distance_penalty": float(dist_penalty),
                "pocket_angle_penalty": float(pocket_angle_penalty),
                "learning_ranker": self.learning_ranker.version,
            },
        )

    def _inside(self, polygon: np.ndarray, point: np.ndarray, margin_mm: float = 0.0) -> bool:
        if polygon.size < 6:
            x, y = point
            return 0.0 - margin_mm <= x <= self.calibration.table.width_mm + margin_mm and 0.0 - margin_mm <= y <= self.calibration.table.height_mm + margin_mm
        dist = cv2.pointPolygonTest(polygon.reshape((-1, 1, 2)).astype(np.float32), (float(point[0]), float(point[1])), True)
        return float(dist) >= float(margin_mm)

    def _segment_inside(self, polygon: np.ndarray, a: np.ndarray, b: np.ndarray, margin_mm: float) -> bool:
        length = float(np.linalg.norm(b - a))
        samples = max(8, int(length / 50.0))
        for i in range(samples + 1):
            t = i / max(1, samples)
            p = a * (1.0 - t) + b * t
            if not self._inside(polygon, p, margin_mm=margin_mm):
                return False
        return True

    def _segment_inside_to_pocket(self, polygon: np.ndarray, a: np.ndarray, b: np.ndarray, margin_mm: float) -> bool:
        length = float(np.linalg.norm(b - a))
        samples = max(8, int(length / 50.0))
        for i in range(samples + 1):
            t = i / max(1, samples)
            p = a * (1.0 - t) + b * t
            remaining = (1.0 - t) * length
            relaxed = -max(18.0, 2.0 * self.calibration.table.ball_diameter_mm) if remaining < 70.0 else margin_mm
            if not self._inside(polygon, p, margin_mm=relaxed):
                return False
        return True

    def _path_clearance(self, a: np.ndarray, b: np.ndarray, balls: List[_Ball], ignore: set[int], moving_radius: float) -> float:
        min_clearance = float("inf")
        for ball in balls:
            if ball.track_id in ignore:
                continue
            d = point_segment_distance(ball.center_mm, a, b)
            clearance = d - ball.radius_mm - moving_radius
            min_clearance = min(min_clearance, float(clearance))
        return min_clearance if np.isfinite(min_clearance) else 9999.0

    def _pocket_angle_penalty(self, target: np.ndarray, pocket: np.ndarray, obj_dir: np.ndarray) -> float:
        table_center = np.asarray([self.calibration.table.width_mm * 0.5, self.calibration.table.height_mm * 0.5], dtype=np.float32)
        inward = unit(table_center - pocket)
        approach = angle_deg(obj_dir, inward)
        return float(max(0.0, (approach - 20.0) / 70.0) ** 1.5)


def _pt(arr: np.ndarray) -> Point:
    return (float(arr[0]), float(arr[1]))

