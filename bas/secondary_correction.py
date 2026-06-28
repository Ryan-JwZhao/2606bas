from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import PlannerConfig
from .schemas import Event, MatchStateFrame, ShotPlan, TrackObservation


ACTIVE_SHOT_PHASES = {"SHOT_ACTIVE", "SETTLING", "TURN_RESOLVE"}
ARMING_PHASES = {"STABLE_IDLE", "PRE_SHOT_ARMED"}


@dataclass
class PendingSecondaryConfirmation:
    cue_track_id: int
    target_track_id: int
    target_group: str
    plan_frame_id: int
    shot_started: bool = False


class SecondaryCorrectionController:
    def __init__(self, config: PlannerConfig):
        self.config = config
        self.pending: Optional[PendingSecondaryConfirmation] = None
        self.last_status = "idle"

    def reset(self) -> None:
        self.pending = None
        self.last_status = "idle"

    def enabled(self) -> bool:
        return bool(getattr(self.config, "cue_sector_correction_enabled", True))

    def arm_from_plan(self, state: MatchStateFrame, plan: ShotPlan) -> None:
        if not self.enabled():
            self.reset()
            self.last_status = "disabled"
            return
        phase = str(state.phase or "").strip().upper()
        if phase not in ARMING_PHASES:
            return
        best = getattr(plan, "best", None)
        if best is None or str(getattr(plan, "shot_mode", "rule") or "rule").strip().lower() != "rule":
            self.pending = None
            self.last_status = "no_confirmation_route"
            return
        explanation = dict(getattr(best, "explanation", {}) or {})
        if not bool(explanation.get("cue_sector_requires_confirmation")):
            self.pending = None
            self.last_status = "no_confirmation_needed"
            return
        target_id = explanation.get("cue_sector_confirmation_target_id", best.target_track_id)
        target_group = explanation.get("cue_sector_confirmation_target_group", best.target_group)
        try:
            target_id_int = int(target_id)
        except (TypeError, ValueError):
            self.pending = None
            self.last_status = "invalid_confirmation_target"
            return
        group = str(target_group or "").strip().lower()
        if group not in {"solid", "stripe"}:
            self.pending = None
            self.last_status = "invalid_confirmation_group"
            return
        self.pending = PendingSecondaryConfirmation(
            cue_track_id=int(best.cue_track_id),
            target_track_id=target_id_int,
            target_group=group,
            plan_frame_id=int(plan.frame_id),
        )
        self.last_status = f"armed:{target_id_int}:{group}"

    def advance_from_state(self, state: MatchStateFrame, state_machine: object) -> Optional[str]:
        if not self.enabled():
            self.reset()
            self.last_status = "disabled"
            return None
        pending = self.pending
        if pending is None:
            return None

        phase = str(state.phase or "").strip().upper()
        if phase in ACTIVE_SHOT_PHASES or self._has_shot_start(state.events):
            pending.shot_started = True
        if not pending.shot_started:
            return None

        for event in state.events:
            if str(event.name).strip().upper() != "BALL_COLLISION_CANDIDATE":
                continue
            other_track_id = self._cue_collision_other_track_id(event, pending.cue_track_id)
            if other_track_id is None:
                continue
            if int(other_track_id) == int(pending.target_track_id):
                group = pending.target_group
                self.pending = None
                setter = getattr(state_machine, "set_turn_target_group", None)
                if callable(setter):
                    setter(group, frame_id=state.frame_id, ts_cam_ns=state.ts_cam_ns, reason="cue_sector_first_contact")
                self.last_status = f"confirmed:{other_track_id}:{group}"
                return group
            self.pending = None
            other_group = self._track_group(state.layout, int(other_track_id))
            self.last_status = f"rejected:first_hit={other_track_id}:{other_group or 'unknown'}"
            return None

        if any(str(event.name).strip().upper() == "TURN_RESOLVE" for event in state.events):
            self.pending = None
            self.last_status = "expired:turn_resolve"
        return None

    @staticmethod
    def _has_shot_start(events: list[Event]) -> bool:
        return any(str(event.name).strip().upper() in {"SHOT_STARTED", "SHOT_START_VOTED"} for event in events)

    @staticmethod
    def _cue_collision_other_track_id(event: Event, cue_track_id: int) -> Optional[int]:
        payload = event.payload or {}
        track_a = _optional_int(payload.get("track_a"))
        track_b = _optional_int(payload.get("track_b"))
        if track_a is None or track_b is None:
            return None
        if int(track_a) == int(cue_track_id):
            return int(track_b)
        if int(track_b) == int(cue_track_id):
            return int(track_a)
        return None

    @staticmethod
    def _track_group(tracks: list[TrackObservation], track_id: int) -> Optional[str]:
        for track in tracks:
            if int(track.track_id) == int(track_id):
                group = str(track.group or "").strip().lower()
                return group or None
        return None


def _optional_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
