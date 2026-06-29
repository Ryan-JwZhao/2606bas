from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from ..calibration.service import CalibrationService
from ..config import PlannerConfig
from ..schemas import MatchStateFrame, ShotCandidate, TrackObservation
from ..utils import unit
from .cue_aim import CueStickAimDetector
from .cue_direction_stability import CueDirectionStabilizer


OBJECT_GROUPS = {"solid", "stripe", "black"}


@dataclass(frozen=True)
class CueSectorAim:
    cue_track_id: int
    tip_px: tuple[float, float]
    tail_px: tuple[float, float]
    direction_px: tuple[float, float]
    tip_mm: tuple[float, float]
    tail_mm: tuple[float, float]
    direction_mm: tuple[float, float]
    corridor_width_px: float
    half_width_px: float


@dataclass(frozen=True)
class SectorBall:
    track_id: int
    group: str
    center_px: tuple[float, float]
    lateral_px: float
    forward_px: float
    distance_px: float


@dataclass(frozen=True)
class CueSectorDebugView:
    cue_center_px: tuple[float, float]
    direction_px: tuple[float, float]
    half_width_px: float
    candidate_centers_px: tuple[tuple[float, float], ...] = ()
    candidate_track_ids: tuple[int, ...] = ()
    status: str = "idle"


class CueSectorCorrection:
    version = "cue_corridor_correction_v2"

    def __init__(self, config: PlannerConfig, calibration: CalibrationService):
        self.config = config
        self.calibration = calibration
        self.last_status = "off"
        self._held_target_id: Optional[int] = None
        self._pending_target_id: Optional[int] = None
        self._pending_frames = 0
        self.aim_detector = CueStickAimDetector()
        self.direction_stabilizer = CueDirectionStabilizer(config)
        self.last_debug_view: Optional[CueSectorDebugView] = None

    def reset(self) -> None:
        self.last_status = "off"
        self._held_target_id = None
        self._pending_target_id = None
        self._pending_frames = 0
        self.direction_stabilizer.reset()
        self.last_debug_view = None

    def enabled(self) -> bool:
        return bool(getattr(self.config, "cue_sector_correction_enabled", True))

    def detect_aim(self, state: MatchStateFrame, cue_ball, frame_bgr: Optional[np.ndarray] = None) -> Optional[CueSectorAim]:
        if not self.enabled():
            self.last_status = "disabled"
            self._reset_aim_state()
            return None
        if not self._balls_stationary(state.layout):
            self.last_status = "balls_moving"
            self._reset_aim_state()
            return None
        cue_track = self._cue_track(state.layout, int(cue_ball.track_id))
        if cue_track is None:
            self.last_status = "no_cue_track"
            self._reset_aim_state()
            return None

        cue_center_px = np.asarray(cue_track.center_px, dtype=np.float32).reshape((2,))
        cue_radius_px = max(2.0, float(cue_track.radius_px))
        min_quality = float(getattr(self.config, "cue_sector_min_stick_quality", 0.25))
        aim_px = self.aim_detector.detect(
            frame_bgr=frame_bgr,
            tracks=state.layout,
            cue_center_px=cue_center_px,
            cue_radius_px=cue_radius_px,
            inner_polygon_px=self._inner_polygon_px(frame_bgr.shape if frame_bgr is not None else None),
            min_stick_quality=min_quality,
        )
        if aim_px is None:
            self.last_status = "no_valid_cue_stick"
            self._reset_aim_state()
            return None

        aim = self._to_table_aim(cue_center_px, aim_px.tip_px, aim_px.tail_px, aim_px.direction_px, int(cue_ball.track_id))
        if aim is None:
            self.last_status = "invalid_cue_direction"
            self._reset_aim_state()
            return None
        direction_decision = self.direction_stabilizer.stabilize(aim.direction_px, aim.direction_mm)
        aim = replace(
            aim,
            direction_px=direction_decision.direction_px,
            direction_mm=direction_decision.direction_mm,
        )
        self._store_debug_view(cue_center_px, aim, (), direction_decision.status)
        self.last_status = f"aim_active:{aim_px.source}/{direction_decision.status}"
        return aim

    def apply(
        self,
        *,
        state: MatchStateFrame,
        cue_ball,
        balls: Sequence[object],
        candidates: Sequence[ShotCandidate],
        aim: CueSectorAim,
        turn_target_group: Optional[str],
    ) -> list[ShotCandidate]:
        sector_balls = self._sector_balls(cue_ball, balls, aim)
        cue_center_px = np.asarray(getattr(cue_ball, "center_px"), dtype=np.float32).reshape((2,))
        self._store_debug_view(cue_center_px, aim, sector_balls, "sector_ready")
        if not sector_balls:
            self.last_status = "empty_sector"
            self._reset_target_stability()
            return []

        sector_ids = {ball.track_id for ball in sector_balls}
        sector_candidates = [candidate for candidate in candidates if int(candidate.target_track_id) in sector_ids]
        if not sector_candidates:
            self.last_status = "sector_no_route"
            self._store_debug_view(cue_center_px, aim, sector_balls, self.last_status)
            self._reset_target_stability()
            return []

        allowed_groups, policy = self._allowed_groups(sector_candidates, turn_target_group)
        selected = [candidate for candidate in sector_candidates if str(candidate.target_group) in allowed_groups]
        if not selected:
            self.last_status = f"{policy}_no_route"
            self._store_debug_view(cue_center_px, aim, sector_balls, self.last_status)
            self._reset_target_stability()
            return []

        selected = sorted(selected, key=lambda candidate: candidate.score, reverse=True)
        selected, policy = self._stabilize(selected, list(candidates), policy)
        if not selected:
            self.last_status = "stabilizer_empty"
            self._store_debug_view(cue_center_px, aim, sector_balls, self.last_status)
            return []

        requires_confirmation = policy == "opponent_confirmation"
        confirmation_target_id = int(selected[0].target_track_id) if requires_confirmation else None
        confirmation_target_group = str(selected[0].target_group) if requires_confirmation else None
        self.last_status = policy
        self._store_debug_view(cue_center_px, aim, sector_balls, policy)
        return [
            self._annotate_candidate(
                candidate,
                aim=aim,
                policy=policy,
                sector_balls=sector_balls,
                requires_confirmation=requires_confirmation,
                confirmation_target_id=confirmation_target_id,
                confirmation_target_group=confirmation_target_group,
            )
            for candidate in selected
        ]

    def all_object_targets(self, balls: Sequence[object]) -> list[object]:
        return [
            ball
            for ball in balls
            if str(getattr(ball, "group", "")).strip().lower() in OBJECT_GROUPS
            and float(getattr(ball, "quality", 0.0)) > 0.25
        ]

    def _cue_track(self, tracks: Sequence[TrackObservation], cue_track_id: int) -> Optional[TrackObservation]:
        for track in tracks:
            if int(track.track_id) == int(cue_track_id) and str(track.group).strip().lower() == "cue":
                if str(getattr(track, "visibility", "visible")).strip().lower() == "visible" and float(track.quality) > 0.25:
                    return track
        for track in tracks:
            if str(track.group).strip().lower() == "cue":
                if str(getattr(track, "visibility", "visible")).strip().lower() == "visible" and float(track.quality) > 0.25:
                    return track
        return None

    def _balls_stationary(self, tracks: Sequence[TrackObservation]) -> bool:
        if not bool(getattr(self.config, "cue_sector_require_balls_stationary", True)):
            return True
        mm_threshold = max(0.0, float(getattr(self.config, "cue_sector_stationary_speed_mm_s", 8.0)))
        px_threshold = max(0.0, float(getattr(self.config, "cue_sector_stationary_speed_px_s", 25.0)))
        for track in tracks:
            group = str(track.group or "").strip().lower()
            if group not in {"cue", "solid", "stripe", "black"}:
                continue
            if str(getattr(track, "visibility", "visible")).strip().lower() != "visible":
                continue
            if float(getattr(track, "quality", 0.0)) <= 0.25:
                continue
            if track.velocity_mm_s is not None:
                vx, vy = track.velocity_mm_s
                speed = float(np.hypot(vx, vy))
                if speed > mm_threshold:
                    return False
            else:
                vx, vy = track.velocity_px_s
                speed = float(np.hypot(vx, vy))
                if speed > px_threshold:
                    return False
        return True

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

    def _to_table_aim(
        self,
        cue_center_px: np.ndarray,
        tip_px: np.ndarray,
        tail_px: np.ndarray,
        direction_px: np.ndarray,
        cue_track_id: int,
    ) -> Optional[CueSectorAim]:
        step = 80.0
        points_px = np.asarray([tip_px, tail_px, cue_center_px, cue_center_px + unit(direction_px) * step], dtype=np.float32)
        try:
            points_mm = self.calibration.camera_px_to_table_mm(points_px).astype(np.float32)
        except Exception:
            return None
        if points_mm.shape[0] < 4 or not np.all(np.isfinite(points_mm)):
            return None
        direction_mm = unit(points_mm[3] - points_mm[2])
        if float(np.linalg.norm(direction_mm)) < 1e-6:
            return None
        corridor_width = max(1.0, float(getattr(self.config, "cue_sector_corridor_width_px", 140.0)))
        return CueSectorAim(
            cue_track_id=int(cue_track_id),
            tip_px=(float(tip_px[0]), float(tip_px[1])),
            tail_px=(float(tail_px[0]), float(tail_px[1])),
            direction_px=(float(direction_px[0]), float(direction_px[1])),
            tip_mm=(float(points_mm[0, 0]), float(points_mm[0, 1])),
            tail_mm=(float(points_mm[1, 0]), float(points_mm[1, 1])),
            direction_mm=(float(direction_mm[0]), float(direction_mm[1])),
            corridor_width_px=float(corridor_width),
            half_width_px=float(corridor_width * 0.5),
        )

    def _sector_balls(self, cue_ball, balls: Sequence[object], aim: CueSectorAim) -> list[SectorBall]:
        direction = unit(np.asarray(aim.direction_px, dtype=np.float32))
        cue_center = np.asarray(getattr(cue_ball, "center_px"), dtype=np.float32)
        half_width = max(0.5, float(aim.half_width_px))
        if half_width <= 0.0:
            return []
        normal = np.asarray([-float(direction[1]), float(direction[0])], dtype=np.float32)
        sector: list[SectorBall] = []
        for ball in balls:
            group = str(getattr(ball, "group", "")).strip().lower()
            if group not in OBJECT_GROUPS:
                continue
            if int(getattr(ball, "track_id", -1)) == int(getattr(cue_ball, "track_id", -2)):
                continue
            if float(getattr(ball, "quality", 0.0)) <= 0.25:
                continue
            center = np.asarray(getattr(ball, "center_px"), dtype=np.float32)
            vec = center - cue_center
            forward = float(np.dot(vec, direction))
            if forward <= 0.0:
                continue
            lateral = float(np.dot(vec, normal))
            if abs(lateral) <= half_width:
                sector.append(
                    SectorBall(
                        track_id=int(getattr(ball, "track_id")),
                        group=group,
                        center_px=(float(center[0]), float(center[1])),
                        lateral_px=float(lateral),
                        forward_px=float(forward),
                        distance_px=float(np.linalg.norm(vec)),
                    )
                )
        return sector

    def _allowed_groups(
        self,
        sector_candidates: Sequence[ShotCandidate],
        turn_target_group: Optional[str],
    ) -> tuple[set[str], str]:
        route_groups = {str(candidate.target_group).strip().lower() for candidate in sector_candidates}
        active = str(turn_target_group or "").strip().lower()
        if active in {"solid", "stripe"}:
            if active in route_groups:
                return {active}, "own_group"
            if "black" in route_groups:
                return {"black"}, "black_fallback"
            opponent = "stripe" if active == "solid" else "solid"
            if opponent in route_groups:
                return {opponent}, "opponent_confirmation"
            return set(), "no_group"
        if active == "black":
            return ({"black"}, "black_turn") if "black" in route_groups else (set(), "black_turn")
        object_groups = route_groups & {"solid", "stripe"}
        if object_groups:
            return object_groups, "open_table"
        if "black" in route_groups:
            return {"black"}, "black_only"
        return set(), "no_group"

    def _stabilize(
        self,
        candidates: list[ShotCandidate],
        all_candidates: list[ShotCandidate],
        policy: str,
    ) -> tuple[list[ShotCandidate], str]:
        if not candidates:
            self._reset_target_stability()
            return [], policy
        confirm_frames = max(1, int(getattr(self.config, "cue_sector_switch_confirm_frames", 2)))
        if confirm_frames <= 1 or self._held_target_id is None:
            self._held_target_id = int(candidates[0].target_track_id)
            self._pending_target_id = None
            self._pending_frames = 0
            return candidates, policy

        held_target_id = int(self._held_target_id)
        held_candidates = [candidate for candidate in candidates if int(candidate.target_track_id) == held_target_id]
        held_candidates_anywhere = held_candidates or [
            candidate for candidate in all_candidates if int(candidate.target_track_id) == held_target_id
        ]
        best = candidates[0]
        if int(best.target_track_id) == held_target_id:
            self._pending_target_id = None
            self._pending_frames = 0
            return candidates, policy

        if not held_candidates_anywhere:
            self._held_target_id = int(best.target_track_id)
            self._pending_target_id = None
            self._pending_frames = 0
            return candidates, policy

        held_best = max(held_candidates_anywhere, key=lambda candidate: candidate.score)
        score_delta = float(best.score) - float(held_best.score)
        min_delta = float(getattr(self.config, "cue_sector_switch_min_score_delta", 0.10))
        if score_delta < min_delta:
            self._pending_target_id = None
            self._pending_frames = 0
            return sorted(held_candidates_anywhere, key=lambda candidate: candidate.score, reverse=True), "stable_hold"

        next_target_id = int(best.target_track_id)
        if self._pending_target_id == next_target_id:
            self._pending_frames += 1
        else:
            self._pending_target_id = next_target_id
            self._pending_frames = 1
        if self._pending_frames < confirm_frames:
            return sorted(held_candidates_anywhere, key=lambda candidate: candidate.score, reverse=True), "stable_hold"

        self._held_target_id = next_target_id
        self._pending_target_id = None
        self._pending_frames = 0
        return candidates, policy

    def _annotate_candidate(
        self,
        candidate: ShotCandidate,
        *,
        aim: CueSectorAim,
        policy: str,
        sector_balls: Sequence[SectorBall],
        requires_confirmation: bool,
        confirmation_target_id: Optional[int],
        confirmation_target_group: Optional[str],
    ) -> ShotCandidate:
        explanation = dict(candidate.explanation)
        explanation.update(
            {
                "cue_sector_correction": True,
                "cue_sector_version": self.version,
                "cue_sector_policy": policy,
                "cue_sector_geometry": "corridor",
                "cue_sector_corridor_width_px": float(aim.corridor_width_px),
                "cue_sector_corridor_half_width_px": float(aim.half_width_px),
                "cue_sector_target_ids": [int(ball.track_id) for ball in sector_balls],
                "cue_sector_target_lateral_px": [float(ball.lateral_px) for ball in sector_balls],
                "cue_sector_target_forward_px": [float(ball.forward_px) for ball in sector_balls],
                "cue_sector_requires_confirmation": bool(requires_confirmation),
                "cue_sector_confirmation_target_id": confirmation_target_id,
                "cue_sector_confirmation_target_group": confirmation_target_group,
            }
        )
        return replace(candidate, explanation=explanation)

    def _reset_target_stability(self) -> None:
        self._held_target_id = None
        self._pending_target_id = None
        self._pending_frames = 0

    def _reset_aim_state(self) -> None:
        self._reset_target_stability()
        self.direction_stabilizer.reset()
        self.last_debug_view = None

    def _store_debug_view(
        self,
        cue_center_px: np.ndarray,
        aim: CueSectorAim,
        sector_balls: Sequence[SectorBall],
        status: str,
    ) -> None:
        centers = tuple(tuple(ball.center_px) for ball in sector_balls)
        track_ids = tuple(int(ball.track_id) for ball in sector_balls)
        self.last_debug_view = CueSectorDebugView(
            cue_center_px=(float(cue_center_px[0]), float(cue_center_px[1])),
            direction_px=(float(aim.direction_px[0]), float(aim.direction_px[1])),
            half_width_px=float(aim.half_width_px),
            candidate_centers_px=centers,
            candidate_track_ids=track_ids,
            status=str(status),
        )
