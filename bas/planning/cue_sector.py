from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from ..calibration.service import CalibrationService
from ..config import PlannerConfig
from ..schemas import MatchStateFrame, ShotCandidate, TrackObservation
from ..utils import angle_deg, point_segment_distance, unit


OBJECT_GROUPS = {"solid", "stripe", "black"}


@dataclass(frozen=True)
class CueSectorAim:
    cue_track_id: int
    tip_mm: tuple[float, float]
    tail_mm: tuple[float, float]
    direction_mm: tuple[float, float]
    total_angle_deg: float
    half_angle_deg: float


@dataclass(frozen=True)
class SectorBall:
    track_id: int
    group: str
    angle_deg: float
    distance_mm: float


class CueSectorCorrection:
    version = "cue_sector_correction_v1"

    def __init__(self, config: PlannerConfig, calibration: CalibrationService):
        self.config = config
        self.calibration = calibration
        self.last_status = "off"
        self._held_target_id: Optional[int] = None
        self._pending_target_id: Optional[int] = None
        self._pending_frames = 0

    def reset(self) -> None:
        self.last_status = "off"
        self._held_target_id = None
        self._pending_target_id = None
        self._pending_frames = 0

    def enabled(self) -> bool:
        return bool(getattr(self.config, "cue_sector_correction_enabled", True))

    def detect_aim(self, state: MatchStateFrame, cue_ball) -> Optional[CueSectorAim]:
        if not self.enabled():
            self.last_status = "disabled"
            self._reset_stability()
            return None
        cue_track = self._cue_track(state.layout, int(cue_ball.track_id))
        if cue_track is None:
            self.last_status = "no_cue_track"
            self._reset_stability()
            return None

        cue_center_px = np.asarray(cue_track.center_px, dtype=np.float32).reshape((2,))
        cue_radius_px = max(2.0, float(cue_track.radius_px))
        line_limit = max(18.0, 4.5 * cue_radius_px)
        near_limit = max(line_limit, 8.0 * cue_radius_px)
        min_quality = float(getattr(self.config, "cue_sector_min_stick_quality", 0.25))
        best: Optional[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = None

        for stick in state.layout:
            if str(stick.group).strip().lower() != "cue_stick":
                continue
            if str(getattr(stick, "visibility", "visible")).strip().lower() != "visible":
                continue
            if float(getattr(stick, "quality", 0.0)) < min_quality:
                continue
            endpoints = self._stick_axis_endpoints(stick)
            if endpoints is None:
                continue
            p1, p2 = endpoints
            seg_len = float(np.linalg.norm(p2 - p1))
            if seg_len < max(24.0, 1.6 * cue_radius_px):
                continue
            line_dist = point_segment_distance(cue_center_px, p1, p2)
            near, far = (p1, p2) if float(np.linalg.norm(p1 - cue_center_px)) <= float(np.linalg.norm(p2 - cue_center_px)) else (p2, p1)
            near_dist = float(np.linalg.norm(near - cue_center_px))
            if line_dist > line_limit and near_dist > near_limit:
                continue
            direction_px = unit(cue_center_px - near)
            align_axis = abs(float(np.dot(unit(p2 - p1), direction_px)))
            if align_axis < 0.42:
                continue
            score = float(seg_len + 70.0 * float(stick.confidence) - 1.4 * line_dist - 0.28 * near_dist)
            if best is None or score > best[0]:
                best = (score, near.astype(np.float32), far.astype(np.float32), direction_px.astype(np.float32))

        if best is None:
            self.last_status = "no_valid_cue_stick"
            self._reset_stability()
            return None

        _score, tip_px, tail_px, direction_px = best
        aim = self._to_table_aim(cue_center_px, tip_px, tail_px, direction_px, int(cue_ball.track_id))
        if aim is None:
            self.last_status = "invalid_cue_direction"
            self._reset_stability()
            return None
        self.last_status = "aim_active"
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
        if not sector_balls:
            self.last_status = "empty_sector"
            self._reset_stability()
            return []

        sector_ids = {ball.track_id for ball in sector_balls}
        sector_candidates = [candidate for candidate in candidates if int(candidate.target_track_id) in sector_ids]
        if not sector_candidates:
            self.last_status = "sector_no_route"
            self._reset_stability()
            return []

        allowed_groups, policy = self._allowed_groups(sector_candidates, turn_target_group)
        selected = [candidate for candidate in sector_candidates if str(candidate.target_group) in allowed_groups]
        if not selected:
            self.last_status = f"{policy}_no_route"
            self._reset_stability()
            return []

        selected = sorted(selected, key=lambda candidate: candidate.score, reverse=True)
        selected = self._stabilize(selected)
        if not selected:
            self.last_status = "stabilizer_empty"
            return []

        requires_confirmation = policy == "opponent_confirmation"
        confirmation_target_id = int(selected[0].target_track_id) if requires_confirmation else None
        confirmation_target_group = str(selected[0].target_group) if requires_confirmation else None
        self.last_status = policy
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

    @staticmethod
    def _stick_axis_endpoints(track: TrackObservation) -> Optional[tuple[np.ndarray, np.ndarray]]:
        x1, y1, x2, y2 = [float(v) for v in track.bbox]
        w = x2 - x1
        h = y2 - y1
        if w <= 1.0 or h <= 1.0:
            return None
        if abs(w) >= abs(h):
            p1 = np.asarray([x1, (y1 + y2) * 0.5], dtype=np.float32)
            p2 = np.asarray([x2, (y1 + y2) * 0.5], dtype=np.float32)
        else:
            p1 = np.asarray([(x1 + x2) * 0.5, y1], dtype=np.float32)
            p2 = np.asarray([(x1 + x2) * 0.5, y2], dtype=np.float32)
        return p1, p2

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
        total_angle = max(1.0, min(120.0, float(getattr(self.config, "cue_sector_angle_deg", 15.0))))
        return CueSectorAim(
            cue_track_id=int(cue_track_id),
            tip_mm=(float(points_mm[0, 0]), float(points_mm[0, 1])),
            tail_mm=(float(points_mm[1, 0]), float(points_mm[1, 1])),
            direction_mm=(float(direction_mm[0]), float(direction_mm[1])),
            total_angle_deg=float(total_angle),
            half_angle_deg=float(total_angle * 0.5),
        )

    def _sector_balls(self, cue_ball, balls: Sequence[object], aim: CueSectorAim) -> list[SectorBall]:
        direction = np.asarray(aim.direction_mm, dtype=np.float32)
        cue_center = np.asarray(getattr(cue_ball, "center_mm"), dtype=np.float32)
        edge_margin = max(0.0, float(getattr(self.config, "cue_sector_edge_margin_deg", 1.0)))
        effective_half = max(0.0, float(aim.half_angle_deg) - edge_margin)
        if effective_half <= 0.0:
            return []
        sector: list[SectorBall] = []
        for ball in balls:
            group = str(getattr(ball, "group", "")).strip().lower()
            if group not in OBJECT_GROUPS:
                continue
            if int(getattr(ball, "track_id", -1)) == int(getattr(cue_ball, "track_id", -2)):
                continue
            if float(getattr(ball, "quality", 0.0)) <= 0.25:
                continue
            center = np.asarray(getattr(ball, "center_mm"), dtype=np.float32)
            vec = center - cue_center
            dist = float(np.linalg.norm(vec))
            if dist <= 1.0:
                continue
            vec_dir = unit(vec)
            if float(np.dot(vec_dir, direction)) <= 0.0:
                continue
            angle = angle_deg(direction, vec)
            if angle <= effective_half:
                sector.append(SectorBall(track_id=int(getattr(ball, "track_id")), group=group, angle_deg=float(angle), distance_mm=dist))
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

    def _stabilize(self, candidates: list[ShotCandidate]) -> list[ShotCandidate]:
        if not candidates:
            self._reset_stability()
            return []
        confirm_frames = max(1, int(getattr(self.config, "cue_sector_switch_confirm_frames", 2)))
        if confirm_frames <= 1 or self._held_target_id is None:
            self._held_target_id = int(candidates[0].target_track_id)
            self._pending_target_id = None
            self._pending_frames = 0
            return candidates

        held_candidates = [candidate for candidate in candidates if int(candidate.target_track_id) == int(self._held_target_id)]
        if not held_candidates:
            self._held_target_id = int(candidates[0].target_track_id)
            self._pending_target_id = None
            self._pending_frames = 0
            return candidates

        best = candidates[0]
        if int(best.target_track_id) == int(self._held_target_id):
            self._pending_target_id = None
            self._pending_frames = 0
            return candidates

        held_best = max(held_candidates, key=lambda candidate: candidate.score)
        score_delta = float(best.score) - float(held_best.score)
        min_delta = float(getattr(self.config, "cue_sector_switch_min_score_delta", 0.10))
        if score_delta < min_delta:
            self._pending_target_id = None
            self._pending_frames = 0
            return self._promote_target(candidates, self._held_target_id)

        next_target_id = int(best.target_track_id)
        if self._pending_target_id == next_target_id:
            self._pending_frames += 1
        else:
            self._pending_target_id = next_target_id
            self._pending_frames = 1
        if self._pending_frames < confirm_frames:
            return self._promote_target(candidates, self._held_target_id)

        self._held_target_id = next_target_id
        self._pending_target_id = None
        self._pending_frames = 0
        return candidates

    @staticmethod
    def _promote_target(candidates: list[ShotCandidate], target_id: int) -> list[ShotCandidate]:
        return sorted(
            candidates,
            key=lambda candidate: (
                0 if int(candidate.target_track_id) == int(target_id) else 1,
                -float(candidate.score),
            ),
        )

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
                "cue_sector_angle_deg": float(aim.total_angle_deg),
                "cue_sector_half_angle_deg": float(aim.half_angle_deg),
                "cue_sector_edge_margin_deg": float(getattr(self.config, "cue_sector_edge_margin_deg", 1.0)),
                "cue_sector_target_ids": [int(ball.track_id) for ball in sector_balls],
                "cue_sector_requires_confirmation": bool(requires_confirmation),
                "cue_sector_confirmation_target_id": confirmation_target_id,
                "cue_sector_confirmation_target_group": confirmation_target_group,
            }
        )
        return replace(candidate, explanation=explanation)

    def _reset_stability(self) -> None:
        self._held_target_id = None
        self._pending_target_id = None
        self._pending_frames = 0
