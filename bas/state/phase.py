from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..config import StateConfig
from ..schemas import Event, MatchPhase, TracksFrame


@dataclass
class PhaseSignals:
    moving: bool
    stable: bool
    anomaly: bool
    cue_stick_seen: bool
    cue_motion: bool
    shot_start_voted: bool


class ShotPhaseMachine:
    """Timing-only state machine for shot lifecycle transitions."""

    def __init__(self, config: StateConfig):
        self.config = config
        self.phase = MatchPhase.STABLE_IDLE
        self.stable_count = 0
        self.moving_count = 0
        self.armed_count = 0
        self.settle_count = 0
        self.anomaly_count = 0

    def reset(self) -> None:
        self.phase = MatchPhase.STABLE_IDLE
        self.stable_count = 0
        self.moving_count = 0
        self.armed_count = 0
        self.settle_count = 0
        self.anomaly_count = 0

    def force(self, phase: MatchPhase) -> None:
        self.phase = phase
        self.stable_count = 0
        self.moving_count = 0
        self.armed_count = 0
        self.settle_count = 0
        self.anomaly_count = 0

    def update(self, frame: TracksFrame, signals: PhaseSignals, events: List[Event]) -> MatchPhase:
        if signals.moving:
            self.moving_count += 1
            self.stable_count = 0
        elif signals.stable:
            self.stable_count += 1
            self.moving_count = 0
        else:
            self.moving_count = max(0, self.moving_count - 1)
            self.stable_count = max(0, self.stable_count - 1)

        if signals.cue_stick_seen and self.phase == MatchPhase.STABLE_IDLE:
            self.armed_count += 1
        else:
            self.armed_count = max(0, self.armed_count - 1)

        if signals.anomaly:
            self.anomaly_count += 1
        else:
            self.anomaly_count = max(0, self.anomaly_count - 1)

        if self.anomaly_count >= self.config.anomaly_frames:
            self._transition(MatchPhase.ANOMALY_RECOVERY, frame, events, "ANOMALY")
        elif self.phase == MatchPhase.STABLE_IDLE:
            if self.armed_count >= self.config.armed_frames:
                self._transition(MatchPhase.PRE_SHOT_ARMED, frame, events, "SHOT_ARMED")
            elif signals.shot_start_voted or signals.cue_motion or self.moving_count >= 2:
                self._transition(MatchPhase.SHOT_ACTIVE, frame, events, "SHOT_STARTED")
        elif self.phase == MatchPhase.PRE_SHOT_ARMED:
            if signals.shot_start_voted or signals.cue_motion or self.moving_count >= 2:
                self._transition(MatchPhase.SHOT_ACTIVE, frame, events, "SHOT_STARTED")
            elif self.stable_count >= self.config.stable_frames and not signals.cue_stick_seen:
                self._transition(MatchPhase.STABLE_IDLE, frame, events, "SHOT_DISARMED")
        elif self.phase == MatchPhase.SHOT_ACTIVE:
            if self.stable_count >= max(2, self.config.stable_frames // 2):
                self.settle_count = 0
                self._transition(MatchPhase.SETTLING, frame, events, "SHOT_SETTLING")
        elif self.phase == MatchPhase.SETTLING:
            if signals.stable:
                self.settle_count += 1
            else:
                self.settle_count = 0
            if self.settle_count >= self.config.settle_frames:
                self._transition(MatchPhase.TURN_RESOLVE, frame, events, "TURN_RESOLVE")
        elif self.phase == MatchPhase.TURN_RESOLVE:
            if self.stable_count >= max(2, self.config.stable_frames // 2):
                self._transition(MatchPhase.STABLE_IDLE, frame, events, "LAYOUT_STABLE")
        elif self.phase == MatchPhase.ANOMALY_RECOVERY:
            if self.stable_count >= self.config.stable_frames:
                self._transition(MatchPhase.STABLE_IDLE, frame, events, "ANOMALY_RECOVERED")
        return self.phase

    def counters(self) -> dict[str, int]:
        return {
            "stable_count": int(self.stable_count),
            "moving_count": int(self.moving_count),
            "armed_count": int(self.armed_count),
            "settle_count": int(self.settle_count),
            "anomaly_count": int(self.anomaly_count),
        }

    def _transition(self, phase: MatchPhase, frame: TracksFrame, events: List[Event], name: str) -> None:
        if self.phase == phase and name not in {"ANOMALY"}:
            return
        self.phase = phase
        events.append(Event(name=name, ts_cam_ns=frame.ts_cam_ns, frame_id=frame.frame_id, confidence=1.0))
