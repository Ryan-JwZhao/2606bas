from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

from .config import PlannerConfig
from .schemas import MatchStateFrame, ProjectionOverlay, ShotCandidate, ShotPlan


STABLE_ROUTE_PHASES = {"STABLE_IDLE", "PRE_SHOT_ARMED"}


@dataclass
class RouteDisplayDecision:
    plan: ShotPlan
    overlay: ProjectionOverlay
    frozen: bool
    status_text: str


@dataclass
class _RouteSnapshot:
    plan: ShotPlan
    overlay: ProjectionOverlay
    signature: tuple


class MotionRouteFreezeController:
    def __init__(self, config: PlannerConfig):
        self.config = config
        self.last_status_text = "off"
        self.reset()

    def reset(self) -> None:
        self._display: Optional[_RouteSnapshot] = None
        self._frozen = False
        self._moving_frames = 0
        self._stable_frames = 0
        self._pending_signature: Optional[tuple] = None
        self._pending_frames = 0
        self.last_status_text = "off"

    def force(self, state: MatchStateFrame, plan: ShotPlan, overlay: ProjectionOverlay) -> RouteDisplayDecision:
        self._remember(plan, overlay)
        self._frozen = False
        self._moving_frames = 0
        self._stable_frames = 1 if self._is_stable_phase(state.phase) else 0
        self._pending_signature = None
        self._pending_frames = 0
        self.last_status_text = "forced"
        return RouteDisplayDecision(plan=plan, overlay=overlay, frozen=False, status_text=self.last_status_text)

    def update(self, state: MatchStateFrame, plan: ShotPlan, overlay: ProjectionOverlay) -> RouteDisplayDecision:
        if not bool(getattr(self.config, "route_freeze_enabled", False)):
            self._remember(plan, overlay)
            self._frozen = False
            self._moving_frames = 0
            self._stable_frames = 0
            self._pending_signature = None
            self._pending_frames = 0
            self.last_status_text = "off"
            return RouteDisplayDecision(plan=plan, overlay=overlay, frozen=False, status_text=self.last_status_text)

        if self._is_stable_phase(state.phase):
            self._stable_frames += 1
            self._moving_frames = 0
            return self._handle_stable_phase(state, plan, overlay)

        self._moving_frames += 1
        self._stable_frames = 0
        return self._handle_motion_phase(state, plan, overlay)

    def _handle_motion_phase(
        self,
        state: MatchStateFrame,
        plan: ShotPlan,
        overlay: ProjectionOverlay,
    ) -> RouteDisplayDecision:
        if self._display is None:
            self._remember(plan, overlay)
            self.last_status_text = "motion_anchor_missing"
            return RouteDisplayDecision(plan=plan, overlay=overlay, frozen=False, status_text=self.last_status_text)

        enter_frames = max(1, int(getattr(self.config, "route_freeze_enter_frames", 1)))
        if self._moving_frames >= enter_frames:
            self._frozen = True
            self.last_status_text = f"frozen_motion {self._moving_frames}/{enter_frames}"
        else:
            self.last_status_text = f"freeze_pending {self._moving_frames}/{enter_frames}"

        bound_plan, bound_overlay = self._bind_to_frame(state, self._display.plan, self._display.overlay)
        return RouteDisplayDecision(
            plan=bound_plan,
            overlay=bound_overlay,
            frozen=self._frozen,
            status_text=self.last_status_text,
        )

    def _handle_stable_phase(
        self,
        state: MatchStateFrame,
        plan: ShotPlan,
        overlay: ProjectionOverlay,
    ) -> RouteDisplayDecision:
        if _is_target_no_route(plan):
            self._remember(plan, overlay)
            self._frozen = False
            self._pending_signature = None
            self._pending_frames = 0
            self.last_status_text = "target_no_route_clear"
            return RouteDisplayDecision(plan=plan, overlay=overlay, frozen=False, status_text=self.last_status_text)

        release_frames = max(1, int(getattr(self.config, "route_freeze_release_frames", 1)))
        if self._frozen and self._stable_frames < release_frames and self._display is not None:
            self.last_status_text = f"release_pending {self._stable_frames}/{release_frames}"
            bound_plan, bound_overlay = self._bind_to_frame(state, self._display.plan, self._display.overlay)
            return RouteDisplayDecision(
                plan=bound_plan,
                overlay=bound_overlay,
                frozen=True,
                status_text=self.last_status_text,
            )

        self._frozen = False
        if self._display is None:
            self._remember(plan, overlay)
            self.last_status_text = "stable_seed"
            return RouteDisplayDecision(plan=plan, overlay=overlay, frozen=False, status_text=self.last_status_text)

        current = self._display
        current_score = _plan_score(current.plan)
        next_score = _plan_score(plan)
        signature = _plan_signature(plan)
        route_delta_mm = _route_delta_mm(current.plan, plan)

        if signature == current.signature:
            self._pending_signature = None
            self._pending_frames = 0
            score_delta = abs(next_score - current_score)
            if (
                route_delta_mm < float(getattr(self.config, "route_freeze_same_route_refresh_mm", 0.0))
                and score_delta < float(getattr(self.config, "route_freeze_same_route_refresh_score_delta", 0.0))
            ):
                self.last_status_text = f"stable_hold {route_delta_mm:.1f}mm"
                bound_plan, bound_overlay = self._bind_to_frame(state, current.plan, current.overlay)
                return RouteDisplayDecision(
                    plan=bound_plan,
                    overlay=bound_overlay,
                    frozen=False,
                    status_text=self.last_status_text,
                )
            self._remember(plan, overlay)
            self.last_status_text = f"stable_refresh {route_delta_mm:.1f}mm"
            return RouteDisplayDecision(plan=plan, overlay=overlay, frozen=False, status_text=self.last_status_text)

        if self._pending_signature == signature:
            self._pending_frames += 1
        else:
            self._pending_signature = signature
            self._pending_frames = 1

        confirm_frames = max(1, int(getattr(self.config, "route_freeze_switch_confirm_frames", 1)))
        if self._pending_frames < confirm_frames:
            self.last_status_text = f"switch_pending {self._pending_frames}/{confirm_frames}"
            bound_plan, bound_overlay = self._bind_to_frame(state, current.plan, current.overlay)
            return RouteDisplayDecision(
                plan=bound_plan,
                overlay=bound_overlay,
                frozen=False,
                status_text=self.last_status_text,
            )

        score_gain = next_score - current_score
        if (
            route_delta_mm < float(getattr(self.config, "route_freeze_switch_min_distance_mm", 0.0))
            and score_gain < float(getattr(self.config, "route_freeze_switch_min_score_delta", 0.0))
        ):
            self.last_status_text = f"switch_suppressed {route_delta_mm:.1f}mm"
            bound_plan, bound_overlay = self._bind_to_frame(state, current.plan, current.overlay)
            return RouteDisplayDecision(
                plan=bound_plan,
                overlay=bound_overlay,
                frozen=False,
                status_text=self.last_status_text,
            )

        self._remember(plan, overlay)
        self._pending_signature = None
        self._pending_frames = 0
        self.last_status_text = f"switch_commit {route_delta_mm:.1f}mm"
        return RouteDisplayDecision(plan=plan, overlay=overlay, frozen=False, status_text=self.last_status_text)

    def _remember(self, plan: ShotPlan, overlay: ProjectionOverlay) -> None:
        self._display = _RouteSnapshot(plan=plan, overlay=overlay, signature=_plan_signature(plan))

    def _bind_to_frame(
        self,
        state: MatchStateFrame,
        plan: ShotPlan,
        overlay: ProjectionOverlay,
    ) -> tuple[ShotPlan, ProjectionOverlay]:
        bound_plan = plan
        if plan.frame_id != state.frame_id or plan.ts_cam_ns != state.ts_cam_ns:
            bound_plan = replace(plan, frame_id=state.frame_id, ts_cam_ns=state.ts_cam_ns)
        bound_overlay = overlay if overlay.frame_id == state.frame_id else replace(overlay, frame_id=state.frame_id)
        return bound_plan, bound_overlay

    def _is_stable_phase(self, phase: str) -> bool:
        return str(phase or "").strip().upper() in STABLE_ROUTE_PHASES


def _plan_signature(plan: ShotPlan) -> tuple:
    shot_mode = str(getattr(plan, "shot_mode", "rule") or "rule").strip().lower()
    if shot_mode == "target":
        best = getattr(plan, "best", None)
        if best is None:
            return ("target", getattr(plan, "locked_target_id", None), None, 0)
        return (
            "target",
            int(getattr(best, "target_track_id", -1)),
            int(getattr(best, "pocket_index", -1)),
            len(getattr(best, "object_line", []) or []),
            int(dict(getattr(best, "explanation", {}) or {}).get("target_shot_rebounds", 0)),
        )
    best = getattr(plan, "best", None)
    if best is None:
        return ("rule", None, None, None)
    return (
        "rule",
        int(getattr(best, "target_track_id", -1)),
        int(getattr(best, "pocket_index", -1)),
        str(getattr(best, "target_group", "")),
    )


def _is_target_no_route(plan: ShotPlan) -> bool:
    shot_mode = str(getattr(plan, "shot_mode", "rule") or "rule").strip().lower()
    return shot_mode == "target" and getattr(plan, "best", None) is None


def _plan_score(plan: ShotPlan) -> float:
    best = getattr(plan, "best", None)
    if best is not None:
        try:
            return float(best.score)
        except (TypeError, ValueError):
            return float("-inf")
    return float("-inf")


def _route_delta_mm(first: ShotPlan, second: ShotPlan) -> float:
    first_mode = str(getattr(first, "shot_mode", "rule") or "rule").strip().lower()
    second_mode = str(getattr(second, "shot_mode", "rule") or "rule").strip().lower()
    if first_mode != second_mode:
        return float("inf")
    if first_mode == "target":
        return _target_route_delta_mm(getattr(first, "best", None), getattr(second, "best", None))
    return _rule_route_delta_mm(getattr(first, "best", None), getattr(second, "best", None))


def _rule_route_delta_mm(first: Optional[ShotCandidate], second: Optional[ShotCandidate]) -> float:
    if first is None and second is None:
        return 0.0
    if first is None or second is None:
        return float("inf")
    distances = [
        _point_delta_mm(first.cue_ball, second.cue_ball),
        _point_delta_mm(first.object_ball, second.object_ball),
        _point_delta_mm(first.ghost_ball, second.ghost_ball),
        _point_delta_mm(first.pocket_point, second.pocket_point),
    ]
    return max(distances)


def _target_route_delta_mm(first: Optional[ShotCandidate], second: Optional[ShotCandidate]) -> float:
    if first is None and second is None:
        return 0.0
    if first is None or second is None:
        return float("inf")
    first_points = list(getattr(first, "object_line", []) or [])
    second_points = list(getattr(second, "object_line", []) or [])
    if len(first_points) != len(second_points):
        return float("inf")
    distances = [
        _point_delta_mm(first.cue_ball, second.cue_ball),
        _point_delta_mm(first.object_ball, second.object_ball),
        _point_delta_mm(first.ghost_ball, second.ghost_ball),
        _point_delta_mm(first.pocket_point, second.pocket_point),
    ]
    for first_point, second_point in zip(first_points, second_points):
        distances.append(_point_delta_mm(first_point, second_point))
    return max(distances)


def _point_delta_mm(first, second) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))
