from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..calibration.service import CalibrationService
from ..config import LearningConfig, PlannerConfig
from ..route_geometry import polygon_contains_with_margin, segment_inside_polygon, segment_inside_polygon_to_pocket
from ..schemas import MatchStateFrame, Point, ShotCandidate, ShotPlan, TrackObservation
from ..utils import angle_deg, clamp, point_segment_distance, unit, wall_time_id
from .aim_context import PlannerAimFrameContext
from .cue_aim import CueStickAimDetector
from .cue_sector import CueSectorCorrection
from .hook_shot import HookShotPlanner
from .learning import create_learning_ranker
from .target_shot import TargetShotDecision, TargetShotModeController, TargetShotPlanner
from .target_lock import TargetLockController, TargetLockDecision


@dataclass
class _Ball:
    track_id: int
    group: str
    center_px: np.ndarray
    center_mm: np.ndarray
    radius_mm: float
    radius_px: float
    quality: float


class GeometryPhysicsPlanner:
    version = "geometry_physics_mvp_v1"

    def __init__(self, config: PlannerConfig, calibration: CalibrationService, learning_config: LearningConfig | None = None):
        self.config = config
        self.calibration = calibration
        self.learning_ranker = create_learning_ranker(
            learning_config,
            table_width_mm=self.calibration.table.width_mm,
            table_height_mm=self.calibration.table.height_mm,
        )
        self.aim_detector = CueStickAimDetector()
        self.cue_sector = CueSectorCorrection(config, calibration, aim_detector=self.aim_detector)
        self.target_lock = TargetLockController(config)
        self.target_shot_mode = TargetShotModeController(config, aim_detector=self.aim_detector)
        self.target_shot_planner = TargetShotPlanner(config, calibration)
        self.hook_shot_planner = HookShotPlanner(config, self.target_shot_planner)
        self.manual_target_id: Optional[int] = None

    def set_manual_target(self, track_id: int) -> None:
        self.manual_target_id = int(track_id)
        self.target_lock.reset()
        self.target_shot_mode.reset()

    def clear_manual_target(self) -> None:
        self.manual_target_id = None
        self.target_lock.reset()

    def plan(
        self,
        state: MatchStateFrame,
        frame_bgr: Optional[np.ndarray] = None,
        *,
        forced_shot_mode: Optional[str] = None,
        forced_turn_target_group: Optional[str] = None,
    ) -> ShotPlan:
        self._release_manual_target_on_shot_start(state)
        shot_mode = self._shot_mode(forced_shot_mode=forced_shot_mode)
        if not self.config.enabled:
            return self._empty_plan(state, shot_mode=shot_mode, hook_status="disabled")
        balls = self._extract_balls(state.layout)
        cue = next((b for b in balls if b.group == "cue"), None)
        if shot_mode == "hook":
            self.target_lock.reset()
            return self._hook_plan(
                state,
                cue,
                balls,
                frame_bgr=frame_bgr,
                turn_target_group=forced_turn_target_group,
            )
        if self.manual_target_id is not None:
            return self._manual_target_plan(state, cue, balls)
        if cue is None:
            return self._empty_plan(state, shot_mode=shot_mode)
        frame_context = self._build_aim_frame_context(
            cue=cue,
            tracks=state.layout,
            frame_bgr=frame_bgr,
        )
        target_shot = self.target_shot_mode.update(
            state=state,
            balls=balls,
            tracks=state.layout,
            frame_bgr=frame_bgr,
            frame_context=frame_context,
        )
        if target_shot.active:
            return self._target_shot_plan(state, cue, balls, target_shot)
        turn_target_group = forced_turn_target_group if forced_turn_target_group is not None else getattr(state, "turn_target_group", None)
        cue_sector_aim = self.cue_sector.detect_aim(
            state,
            cue,
            frame_bgr=frame_bgr,
            frame_context=frame_context,
        )
        target_lock = self.target_lock.update(state=state, cue_ball=cue, balls=balls, aim=cue_sector_aim)
        locked_target = self._locked_target(balls, target_lock)
        if locked_target is not None:
            targets = [locked_target]
        elif cue_sector_aim is not None:
            targets = self.cue_sector.all_object_targets(balls)
        else:
            targets = self._eligible_targets(
                balls,
                turn_target_group=turn_target_group,
            )
        if not targets:
            return self._empty_plan(
                state,
                shot_mode=shot_mode,
                target_lock=target_lock,
            )
        candidates: List[ShotCandidate] = []
        pockets = [np.asarray(p, dtype=np.float32) for p in self.calibration.table.pockets_mm]
        center_polygon = self.calibration.table.center_playable_polygon_mm or self.calibration.table.inner_polygon_mm
        inner = np.asarray(center_polygon, dtype=np.float32)
        for target in targets:
            for pocket_index, pocket in enumerate(pockets):
                candidate = self._candidate(cue, target, pocket, pocket_index, balls, inner)
                if candidate is not None:
                    candidates.append(candidate)
        candidates = self.learning_ranker.rerank(candidates, state)
        if locked_target is not None:
            candidates = self._apply_target_lock(candidates, target_lock)
        elif cue_sector_aim is not None:
            candidates = self.cue_sector.apply(
                state=state,
                cue_ball=cue,
                balls=balls,
                candidates=candidates,
                aim=cue_sector_aim,
                turn_target_group=turn_target_group,
            )
        candidates = candidates[: max(1, int(self.config.top_k))]
        best = candidates[0] if candidates else None
        return ShotPlan(
            plan_id=f"plan_{state.frame_id}_{wall_time_id()}",
            frame_id=state.frame_id,
            ts_cam_ns=state.ts_cam_ns,
            candidates=candidates,
            best=best,
            shot_mode="rule",
            planner_version=self.version,
            locked_target_id=target_lock.locked_target_id,
            target_lock_status=target_lock.status,
            target_shot_status=target_shot.status,
        )

    def _shot_mode(self, *, forced_shot_mode: Optional[str] = None) -> str:
        mode = str(forced_shot_mode if forced_shot_mode is not None else getattr(self.config, "shot_mode", "rule") or "rule").strip().lower()
        return "hook" if mode in {"hook", "hook_shot", "free", "free_shot"} else "rule"

    def _empty_plan(
        self,
        state: MatchStateFrame,
        *,
        shot_mode: str,
        hook_status: str = "off",
        target_lock: TargetLockDecision | None = None,
    ) -> ShotPlan:
        return ShotPlan(
            plan_id=f"plan_{state.frame_id}_{wall_time_id()}",
            frame_id=state.frame_id,
            ts_cam_ns=state.ts_cam_ns,
            shot_mode=shot_mode,
            hook_status=hook_status,
            planner_version=self.version if shot_mode == "rule" else f"{self.version}+{self.hook_shot_planner.version}",
            locked_target_id=target_lock.locked_target_id if target_lock is not None else None,
            target_lock_status=target_lock.status if target_lock is not None else "off",
            target_shot_status="off",
        )

    def _hook_plan(
        self,
        state: MatchStateFrame,
        cue: Optional[_Ball],
        balls: Sequence[_Ball],
        *,
        frame_bgr: Optional[np.ndarray],
        turn_target_group: Optional[str],
    ) -> ShotPlan:
        if cue is None:
            return self._empty_plan(state, shot_mode="hook", hook_status="no_cue_ball")

        manual_target = self._manual_target(balls)
        frame_context = self._build_aim_frame_context(cue=cue, tracks=state.layout, frame_bgr=frame_bgr)
        if manual_target is not None:
            targets = [manual_target]
            selection_source = "manual"
            locked_target_id = int(manual_target.track_id)
            lock_status = "manual"
            target_shot_status = "manual_target"
        else:
            target_shot = self.target_shot_mode.update(
                state=state,
                balls=balls,
                tracks=state.layout,
                frame_bgr=frame_bgr,
                frame_context=frame_context,
            )
            pointed_target = self._target_shot_target(balls, target_shot) if target_shot.active else None
            if pointed_target is not None:
                targets = [pointed_target]
                selection_source = "cue_hold"
                locked_target_id = int(pointed_target.track_id)
                lock_status = f"hook_cue_hold:{target_shot.status}"
            else:
                effective_group = turn_target_group if turn_target_group is not None else getattr(state, "turn_target_group", None)
                targets = self._eligible_targets(balls, turn_target_group=effective_group)
                selection_source = "automatic"
                locked_target_id = None
                lock_status = "hook_global"
            target_shot_status = target_shot.status

        result = self.hook_shot_planner.plan(
            cue_ball=cue,
            targets=targets,
            balls=balls,
            selection_source=selection_source,
        )
        return ShotPlan(
            plan_id=f"plan_{state.frame_id}_{wall_time_id()}",
            frame_id=state.frame_id,
            ts_cam_ns=state.ts_cam_ns,
            candidates=result.candidates,
            best=result.best,
            shot_mode="hook",
            hook_status=result.status,
            planner_version=f"{self.version}+{self.hook_shot_planner.version}+{self.target_shot_planner.version}",
            locked_target_id=locked_target_id,
            target_lock_status=lock_status,
            target_shot_status=target_shot_status,
        )

    def _extract_balls(self, tracks: Sequence[TrackObservation]) -> List[_Ball]:
        balls: List[_Ball] = []
        for tr in tracks:
            if tr.group not in {"cue", "solid", "stripe", "black"}:
                continue
            if str(getattr(tr, "visibility", "visible")).strip().lower() != "visible":
                continue
            if float(getattr(tr, "quality", 0.0)) <= 0.25:
                continue
            center = self.calibration.ball_camera_px_to_table_mm(np.asarray([tr.center_px], dtype=np.float32))[0]
            radius_mm = self.calibration.ball_pixel_radius_to_mm(tr.center_px, tr.radius_px)
            if radius_mm <= 1.0 or radius_mm > 80.0:
                radius_mm = 0.5 * self.calibration.table.ball_diameter_mm
            balls.append(
                _Ball(
                    track_id=tr.track_id,
                    group=tr.group,
                    center_px=np.asarray(tr.center_px, dtype=np.float32).reshape((2,)),
                    center_mm=center.astype(np.float32),
                    radius_mm=float(radius_mm),
                    radius_px=float(max(2.0, tr.radius_px)),
                    quality=float(tr.quality),
                )
            )
        return balls

    def _build_aim_frame_context(
        self,
        *,
        cue: _Ball,
        tracks: Sequence[TrackObservation],
        frame_bgr: Optional[np.ndarray],
    ) -> PlannerAimFrameContext:
        min_stick_quality = min(
            float(getattr(self.config, "target_shot_min_stick_quality", 0.25)),
            float(getattr(self.config, "cue_sector_min_stick_quality", 0.25)),
        )
        return PlannerAimFrameContext(
            frame_bgr=frame_bgr,
            tracks=tracks,
            cue_center_px=cue.center_px,
            cue_radius_px=cue.radius_px,
            inner_polygon_px=self._inner_polygon_px(frame_bgr.shape if frame_bgr is not None else None),
            aim_detector=self.aim_detector,
            min_stick_quality=min_stick_quality,
        )

    def _inner_polygon_px(self, frame_shape: Optional[tuple[int, ...]]) -> Optional[np.ndarray]:
        base_polygon = self.calibration.table.center_playable_polygon_mm or self.calibration.table.inner_polygon_mm
        poly_mm = np.asarray(base_polygon, dtype=np.float32).reshape((-1, 2))
        if poly_mm.shape[0] >= 3:
            try:
                poly_px = self.calibration.table_mm_to_camera_px(poly_mm).astype(np.float32)
                if np.all(np.isfinite(poly_px)):
                    return poly_px
            except Exception:
                pass
        if frame_shape is not None and len(frame_shape) >= 2:
            h, w = frame_shape[:2]
            return np.asarray([(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))], dtype=np.float32)
        return None

    def _release_manual_target_on_shot_start(self, state: MatchStateFrame) -> None:
        if any(str(event.name).strip().upper() == "SHOT_STARTED" for event in getattr(state, "events", ())):
            self.clear_manual_target()

    def _manual_target_plan(
        self,
        state: MatchStateFrame,
        cue: Optional[_Ball],
        balls: Sequence[_Ball],
    ) -> ShotPlan:
        target_id = self.manual_target_id
        target = self._manual_target(balls)
        target_lock = TargetLockDecision(target_id, target.group if target is not None else None, "manual")
        if cue is None or target is None:
            return ShotPlan(
                plan_id=f"plan_{state.frame_id}_{wall_time_id()}",
                frame_id=state.frame_id,
                ts_cam_ns=state.ts_cam_ns,
                shot_mode="target",
                planner_version=f"{self.version}+manual_target",
                locked_target_id=target_lock.locked_target_id,
                target_lock_status="manual_missing",
                target_shot_status="manual_target_missing",
            )

        pockets = [np.asarray(p, dtype=np.float32) for p in self.calibration.table.pockets_mm]
        center_polygon = self.calibration.table.center_playable_polygon_mm or self.calibration.table.inner_polygon_mm
        inner = np.asarray(center_polygon, dtype=np.float32)
        candidates = [
            candidate
            for pocket_index, pocket in enumerate(pockets)
            if (candidate := self._candidate(cue, target, pocket, pocket_index, list(balls), inner)) is not None
        ]
        candidates = self.learning_ranker.rerank(candidates, state)
        candidates = [self._annotate_target_lock(candidate, target_lock) for candidate in candidates]
        candidates = candidates[: max(1, int(self.config.top_k))]
        return ShotPlan(
            plan_id=f"plan_{state.frame_id}_{wall_time_id()}",
            frame_id=state.frame_id,
            ts_cam_ns=state.ts_cam_ns,
            candidates=candidates,
            best=candidates[0] if candidates else None,
            shot_mode="target",
            planner_version=f"{self.version}+manual_target",
            locked_target_id=target_lock.locked_target_id,
            target_lock_status="manual",
            target_shot_status="manual_target",
        )

    def _locked_target(self, balls: Sequence[_Ball], target_lock: TargetLockDecision) -> Optional[_Ball]:
        if target_lock.locked_target_id is None:
            return None
        for ball in balls:
            if int(ball.track_id) == int(target_lock.locked_target_id) and ball.group in {"solid", "stripe", "black"} and ball.quality > 0.25:
                return ball
        return None

    def _manual_target(self, balls: Sequence[_Ball]) -> Optional[_Ball]:
        if self.manual_target_id is None:
            return None
        for ball in balls:
            if int(ball.track_id) == int(self.manual_target_id) and ball.group in {"solid", "stripe", "black"} and ball.quality > 0.25:
                return ball
        return None

    def _target_shot_plan(
        self,
        state: MatchStateFrame,
        cue: _Ball,
        balls: Sequence[_Ball],
        target_shot: TargetShotDecision,
    ) -> ShotPlan:
        target = self._target_shot_target(balls, target_shot)
        best = None
        candidates: List[ShotCandidate] = []
        status = target_shot.status
        if target is None:
            status = f"{status}:target_missing"
            self.target_shot_planner.last_status = "target_missing"
        else:
            best = self.target_shot_planner.plan(cue_ball=cue, target=target, balls=balls, decision=target_shot)
            if best is not None:
                candidates = [best]
            status = f"{status}:{self.target_shot_planner.last_status}"
        return ShotPlan(
            plan_id=f"plan_{state.frame_id}_{wall_time_id()}",
            frame_id=state.frame_id,
            ts_cam_ns=state.ts_cam_ns,
            candidates=candidates,
            best=best,
            shot_mode="target",
            planner_version=f"{self.version}+{self.target_shot_planner.version}",
            locked_target_id=target_shot.active_target_id,
            target_lock_status=f"target_shot:{target_shot.status}",
            target_shot_status=status,
        )

    def _target_shot_target(self, balls: Sequence[_Ball], target_shot: TargetShotDecision) -> Optional[_Ball]:
        if target_shot.active_target_id is None:
            return None
        for ball in balls:
            if int(ball.track_id) == int(target_shot.active_target_id) and ball.group in {"solid", "stripe", "black"} and ball.quality > 0.25:
                return ball
        return None

    def _apply_target_lock(self, candidates: List[ShotCandidate], target_lock: TargetLockDecision) -> List[ShotCandidate]:
        if target_lock.locked_target_id is None:
            return candidates
        locked = [
            candidate
            for candidate in candidates
            if int(candidate.target_track_id) == int(target_lock.locked_target_id)
        ]
        return [self._annotate_target_lock(candidate, target_lock) for candidate in locked]

    def _annotate_target_lock(self, candidate: ShotCandidate, target_lock: TargetLockDecision) -> ShotCandidate:
        explanation = dict(candidate.explanation)
        explanation.update(
            {
                "target_lock": True,
                "target_lock_version": self.target_lock.version,
                "target_lock_status": target_lock.status,
                "target_lock_target_id": target_lock.locked_target_id,
                "target_lock_group": target_lock.locked_group,
            }
        )
        return replace(candidate, explanation=explanation)

    def _eligible_targets(self, balls: Sequence[_Ball], turn_target_group: Optional[str] = None) -> List[_Ball]:
        targets = [b for b in balls if b.group in {"solid", "stripe", "black"} and b.quality > 0.25]
        if not targets:
            return []
        solid_count = sum(1 for b in targets if b.group == "solid")
        stripe_count = sum(1 for b in targets if b.group == "stripe")
        if solid_count == 0 and stripe_count == 0:
            return [b for b in targets if b.group == "black"]
        active_group = str(turn_target_group or "").strip().lower()
        if active_group == "solid":
            return [b for b in targets if b.group == "solid"] if solid_count > 0 else [b for b in targets if b.group == "black"]
        if active_group == "stripe":
            return [b for b in targets if b.group == "stripe"] if stripe_count > 0 else [b for b in targets if b.group == "black"]
        if active_group == "black":
            return [b for b in targets if b.group == "black"]
        return [b for b in targets if b.group in {"solid", "stripe"}]

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
        return polygon_contains_with_margin(polygon, point, margin_mm=margin_mm)

    def _segment_inside(self, polygon: np.ndarray, a: np.ndarray, b: np.ndarray, margin_mm: float) -> bool:
        if polygon.size < 6:
            length = float(np.linalg.norm(b - a))
            samples = max(8, int(length / 50.0))
            for i in range(samples + 1):
                t = i / max(1, samples)
                p = a * (1.0 - t) + b * t
                if not self._inside(polygon, p, margin_mm=margin_mm):
                    return False
            return True
        return segment_inside_polygon(polygon, a, b, margin_mm=margin_mm)

    def _segment_inside_to_pocket(self, polygon: np.ndarray, a: np.ndarray, b: np.ndarray, margin_mm: float) -> bool:
        if polygon.size < 6:
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
        return segment_inside_polygon_to_pocket(
            polygon,
            a,
            b,
            margin_mm=margin_mm,
            pocket_relief_mm=max(18.0, 2.0 * self.calibration.table.ball_diameter_mm),
        )

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
