from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..config import PlannerConfig
from ..schemas import MatchStateFrame
from ..utils import unit
from .corridor_targeting import rank_object_balls_in_corridor


OBJECT_GROUPS = {"solid", "stripe", "black"}
AIM_UPDATE_PHASES = {"STABLE_IDLE", "PRE_SHOT_ARMED"}
MOTION_HOLD_PHASES = {"SHOT_ACTIVE", "SETTLING"}


@dataclass(frozen=True)
class TargetLockDecision:
    locked_target_id: Optional[int]
    locked_group: Optional[str]
    status: str
    candidate_target_id: Optional[int] = None
    pending_target_id: Optional[int] = None
    active: bool = False
    switched: bool = False


@dataclass(frozen=True)
class _AimTarget:
    track_id: int
    group: str
    center_px: tuple[float, float]
    lateral_px: float
    forward_px: float


class TargetLockController:
    """Tracks the user's intended object ball across cue jitter and shot motion."""

    version = "target_lock_v1"

    def __init__(self, config: PlannerConfig):
        self.config = config
        self.last_status = "off"
        self.reset()

    def reset(self) -> None:
        self._locked_target_id: Optional[int] = None
        self._locked_group: Optional[str] = None
        self._anchor_px: Optional[tuple[float, float]] = None
        self._seed_target_id: Optional[int] = None
        self._seed_frames = 0
        self._pending_target_id: Optional[int] = None
        self._pending_frames = 0
        self._missing_frames = 0
        self.last_status = "off"

    def update(
        self,
        *,
        state: MatchStateFrame,
        cue_ball: object,
        balls: Sequence[object],
        aim: object | None,
    ) -> TargetLockDecision:
        if not bool(getattr(self.config, "target_lock_enabled", True)):
            self.reset()
            return self._decision("disabled")

        phase = str(getattr(state, "phase", "") or "").strip().upper()
        objects = self._object_balls(balls)
        self._refresh_locked_ball(objects, phase)

        aim_target = self._aim_target(cue_ball, objects, aim) if phase in AIM_UPDATE_PHASES else None
        if self._locked_target_id is None:
            return self._update_seed(aim_target)
        return self._update_locked(aim_target, phase)

    def _update_seed(self, aim_target: _AimTarget | None) -> TargetLockDecision:
        if aim_target is None:
            self._clear_seed()
            return self._decision("unlocked_no_aim")
        if self._seed_target_id == aim_target.track_id:
            self._seed_frames += 1
        else:
            self._seed_target_id = aim_target.track_id
            self._seed_frames = 1
        confirm_frames = max(1, int(getattr(self.config, "target_lock_confirm_frames", 3)))
        if self._seed_frames < confirm_frames:
            return self._decision(
                f"seed_pending {self._seed_frames}/{confirm_frames}",
                candidate_target_id=aim_target.track_id,
            )
        self._lock(aim_target.track_id, aim_target.group, aim_target.center_px)
        self._clear_seed()
        return self._decision("locked_new", candidate_target_id=aim_target.track_id)

    def _update_locked(self, aim_target: _AimTarget | None, phase: str) -> TargetLockDecision:
        if aim_target is None:
            self._clear_pending_switch()
            status = "locked_hold_motion" if phase in MOTION_HOLD_PHASES else "locked_hold_no_aim"
            return self._decision(status)

        if int(aim_target.track_id) == int(self._locked_target_id):
            self._clear_pending_switch()
            self._missing_frames = 0
            self._anchor_px = aim_target.center_px
            return self._decision("locked_hold_same", candidate_target_id=aim_target.track_id)

        if not self._is_large_switch(aim_target):
            self._clear_pending_switch()
            return self._decision("locked_hold_near_anchor", candidate_target_id=aim_target.track_id)

        if self._pending_target_id == aim_target.track_id:
            self._pending_frames += 1
        else:
            self._pending_target_id = aim_target.track_id
            self._pending_frames = 1

        confirm_frames = max(1, int(getattr(self.config, "target_lock_switch_confirm_frames", 8)))
        if self._pending_frames < confirm_frames:
            return self._decision(
                f"switch_pending {self._pending_frames}/{confirm_frames}",
                candidate_target_id=aim_target.track_id,
                pending_target_id=aim_target.track_id,
            )

        self._lock(aim_target.track_id, aim_target.group, aim_target.center_px)
        self._clear_pending_switch()
        return self._decision("switch_commit", candidate_target_id=aim_target.track_id, switched=True)

    def _refresh_locked_ball(self, objects: Sequence[object], phase: str) -> None:
        if self._locked_target_id is None:
            return
        locked = self._find_ball(objects, self._locked_target_id)
        if locked is not None:
            self._locked_group = str(getattr(locked, "group", self._locked_group or "")).strip().lower() or self._locked_group
            self._anchor_px = self._center_px(locked)
            self._missing_frames = 0
            return

        reacquired = self._reacquire_ball(objects)
        if reacquired is not None:
            self._locked_target_id = int(getattr(reacquired, "track_id"))
            self._locked_group = str(getattr(reacquired, "group", self._locked_group or "")).strip().lower() or self._locked_group
            self._anchor_px = self._center_px(reacquired)
            self._missing_frames = 0
            return

        if phase in MOTION_HOLD_PHASES:
            return
        self._missing_frames += 1
        release_frames = max(1, int(getattr(self.config, "target_lock_missing_release_frames", 45)))
        if self._missing_frames >= release_frames:
            self._release()

    def _aim_target(
        self,
        cue_ball: object,
        balls: Sequence[object],
        aim: object | None,
    ) -> _AimTarget | None:
        if aim is None:
            return None
        direction = unit(np.asarray(getattr(aim, "direction_px", (0.0, 0.0)), dtype=np.float32))
        if float(np.linalg.norm(direction)) < 1e-6:
            return None
        half_width = 0.5 * max(
            1.0,
            float(
                getattr(
                    self.config,
                    "target_lock_corridor_width_px",
                    getattr(self.config, "cue_sector_corridor_width_px", 140.0),
                )
            ),
        )
        ranked = rank_object_balls_in_corridor(
            cue_ball=cue_ball,
            balls=balls,
            direction_px=direction,
            half_width_px=half_width,
        )
        if not ranked:
            return None
        best = ranked[0]
        return _AimTarget(
            track_id=int(best.track_id),
            group=str(best.group).strip().lower(),
            center_px=best.center_px,
            lateral_px=float(best.lateral_px),
            forward_px=float(best.forward_px),
        )

    def _object_balls(self, balls: Sequence[object]) -> list[object]:
        objects: list[object] = []
        for ball in balls:
            group = str(getattr(ball, "group", "")).strip().lower()
            if group not in OBJECT_GROUPS:
                continue
            if float(getattr(ball, "quality", 0.0)) <= 0.25:
                continue
            objects.append(ball)
        return objects

    @staticmethod
    def _find_ball(balls: Sequence[object], track_id: int | None) -> object | None:
        if track_id is None:
            return None
        for ball in balls:
            if int(getattr(ball, "track_id", -1)) == int(track_id):
                return ball
        return None

    def _reacquire_ball(self, balls: Sequence[object]) -> object | None:
        if self._anchor_px is None:
            return None
        anchor = np.asarray(self._anchor_px, dtype=np.float32)
        max_distance = max(0.0, float(getattr(self.config, "target_lock_reacquire_radius_px", 90.0)))
        best: tuple[float, object] | None = None
        for ball in balls:
            group = str(getattr(ball, "group", "")).strip().lower()
            if self._locked_group is not None and group != self._locked_group:
                continue
            distance = float(np.linalg.norm(np.asarray(self._center_px(ball), dtype=np.float32) - anchor))
            if distance > max_distance:
                continue
            if best is None or distance < best[0]:
                best = (distance, ball)
        return best[1] if best is not None else None

    def _is_large_switch(self, aim_target: _AimTarget) -> bool:
        if self._anchor_px is None:
            return True
        distance = float(np.linalg.norm(np.asarray(aim_target.center_px, dtype=np.float32) - np.asarray(self._anchor_px, dtype=np.float32)))
        min_distance = max(0.0, float(getattr(self.config, "target_lock_switch_min_distance_px", 70.0)))
        return distance >= min_distance

    @staticmethod
    def _center_px(ball: object) -> tuple[float, float]:
        center = getattr(ball, "center_px")
        return (float(center[0]), float(center[1]))

    def _lock(self, track_id: int, group: str, center_px: tuple[float, float]) -> None:
        self._locked_target_id = int(track_id)
        self._locked_group = str(group).strip().lower()
        self._anchor_px = (float(center_px[0]), float(center_px[1]))
        self._missing_frames = 0

    def _release(self) -> None:
        self._locked_target_id = None
        self._locked_group = None
        self._anchor_px = None
        self._clear_seed()
        self._clear_pending_switch()
        self._missing_frames = 0

    def _clear_seed(self) -> None:
        self._seed_target_id = None
        self._seed_frames = 0

    def _clear_pending_switch(self) -> None:
        self._pending_target_id = None
        self._pending_frames = 0

    def _decision(
        self,
        status: str,
        *,
        candidate_target_id: Optional[int] = None,
        pending_target_id: Optional[int] = None,
        switched: bool = False,
    ) -> TargetLockDecision:
        self.last_status = str(status)
        return TargetLockDecision(
            locked_target_id=self._locked_target_id,
            locked_group=self._locked_group,
            status=self.last_status,
            candidate_target_id=candidate_target_id,
            pending_target_id=pending_target_id,
            active=self._locked_target_id is not None,
            switched=bool(switched),
        )
