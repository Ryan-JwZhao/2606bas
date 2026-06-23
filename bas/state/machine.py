from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set

import numpy as np

from ..config import StateConfig
from ..schemas import Event, MatchPhase, MatchStateFrame, TrackObservation, TracksFrame


@dataclass
class _TrackMemory:
    track_id: int
    group: str
    last_center_px: tuple[float, float]
    last_seen_frame: int
    last_seen_ts_ns: int
    last_quality: float


class MatchStateMachine:
    version = "temporal_state_v1"

    def __init__(self, config: StateConfig):
        self.config = config
        self.phase = MatchPhase.STABLE_IDLE
        self._stable_count = 0
        self._moving_count = 0
        self._armed_count = 0
        self._settle_count = 0
        self._anomaly_count = 0
        self._snapshot_layout: List[TrackObservation] = []
        self._last_layout: List[TrackObservation] = []
        self._memory: Dict[int, _TrackMemory] = {}
        self._potted_ids: Set[int] = set()
        self._recent_events: Deque[Event] = deque(maxlen=128)
        self._operator_hold = False
        self._operator_lock_frames = 0
        self._pending_operator_events: Deque[Event] = deque(maxlen=32)

    def reset(self) -> None:
        self.phase = MatchPhase.STABLE_IDLE
        self._stable_count = 0
        self._moving_count = 0
        self._armed_count = 0
        self._settle_count = 0
        self._anomaly_count = 0
        self._snapshot_layout = []
        self._last_layout = []
        self._memory.clear()
        self._potted_ids.clear()
        self._recent_events.clear()
        self._operator_hold = False
        self._operator_lock_frames = 0
        self._pending_operator_events.clear()

    def force_phase(self, phase: MatchPhase | str, *, frame_id: int = 0, ts_cam_ns: int = 0, reason: str = "operator") -> None:
        target = phase if isinstance(phase, MatchPhase) else MatchPhase(str(phase))
        self.phase = target
        self._stable_count = 0
        self._moving_count = 0
        self._armed_count = 0
        self._settle_count = 0
        self._anomaly_count = 0
        self._operator_lock_frames = max(self._operator_lock_frames, 2)
        self._queue_operator_event(
            "OPERATOR_FORCE_PHASE",
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            payload={"phase": target.value, "reason": reason},
        )

    def set_operator_hold(self, enabled: bool, *, frame_id: int = 0, ts_cam_ns: int = 0, reason: str = "operator") -> None:
        self._operator_hold = bool(enabled)
        name = "OPERATOR_HOLD_ENABLED" if enabled else "OPERATOR_HOLD_DISABLED"
        self._queue_operator_event(name, frame_id=frame_id, ts_cam_ns=ts_cam_ns, payload={"reason": reason})

    def snapshot_layout(self, tracks_frame: Optional[TracksFrame] = None, *, frame_id: int = 0, ts_cam_ns: int = 0) -> None:
        if tracks_frame is not None:
            self._snapshot_layout = list(tracks_frame.tracks)
            frame_id = tracks_frame.frame_id
            ts_cam_ns = tracks_frame.ts_cam_ns
        else:
            self._snapshot_layout = list(self._last_layout)
        self._queue_operator_event(
            "OPERATOR_SNAPSHOT_LAYOUT",
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            payload={"tracks": len(self._snapshot_layout)},
        )

    def clear_review_flags(self, *, frame_id: int = 0, ts_cam_ns: int = 0) -> None:
        self._potted_ids.clear()
        self._anomaly_count = 0
        self._queue_operator_event("OPERATOR_CLEAR_REVIEW_FLAGS", frame_id=frame_id, ts_cam_ns=ts_cam_ns)

    @property
    def operator_hold(self) -> bool:
        return self._operator_hold

    def update(self, tracks_frame: TracksFrame) -> MatchStateFrame:
        tracks = tracks_frame.tracks
        self._last_layout = list(tracks)
        events: List[Event] = self._drain_operator_events(tracks_frame)
        moving = self._is_moving(tracks)
        stable = self._is_stable(tracks)
        anomaly = self._detect_anomaly(tracks)
        events.extend(self._detect_disappearances(tracks_frame))
        self._update_memory(tracks_frame)

        if self._operator_hold:
            for event in events:
                self._recent_events.append(event)
            layout = self._snapshot_layout if self._snapshot_layout else tracks
            return MatchStateFrame(
                frame_id=tracks_frame.frame_id,
                ts_cam_ns=tracks_frame.ts_cam_ns,
                phase=self.phase.value,
                events=events,
                layout=layout,
                confidence=0.70,
                state_version=f"{self.version}+operator_hold",
            )

        if self._operator_lock_frames > 0:
            self._operator_lock_frames -= 1
            for event in events:
                self._recent_events.append(event)
            return MatchStateFrame(
                frame_id=tracks_frame.frame_id,
                ts_cam_ns=tracks_frame.ts_cam_ns,
                phase=self.phase.value,
                events=events,
                layout=tracks,
                confidence=0.85,
                state_version=f"{self.version}+operator_override",
            )

        if moving:
            self._moving_count += 1
            self._stable_count = 0
        elif stable:
            self._stable_count += 1
            self._moving_count = 0
        else:
            self._moving_count = max(0, self._moving_count - 1)
            self._stable_count = max(0, self._stable_count - 1)

        cue_stick_seen = any(t.group == "cue_stick" and t.visibility == "visible" and t.quality > 0.35 for t in tracks)
        cue_motion = any(t.group == "cue" and self._speed(t) > self.config.moving_speed_px_s for t in tracks)
        if cue_stick_seen and self.phase == MatchPhase.STABLE_IDLE:
            self._armed_count += 1
        else:
            self._armed_count = max(0, self._armed_count - 1)

        if anomaly:
            self._anomaly_count += 1
        else:
            self._anomaly_count = max(0, self._anomaly_count - 1)

        if self._anomaly_count >= self.config.anomaly_frames:
            self._transition(MatchPhase.ANOMALY_RECOVERY, tracks_frame, events, "ANOMALY")
        elif self.phase == MatchPhase.STABLE_IDLE:
            if self._armed_count >= self.config.armed_frames:
                self._snapshot_layout = list(tracks)
                self._transition(MatchPhase.PRE_SHOT_ARMED, tracks_frame, events, "SHOT_ARMED")
            elif cue_motion or self._moving_count >= 2:
                self._snapshot_layout = list(tracks)
                self._transition(MatchPhase.SHOT_ACTIVE, tracks_frame, events, "SHOT_STARTED")
        elif self.phase == MatchPhase.PRE_SHOT_ARMED:
            if cue_motion or self._moving_count >= 2:
                self._transition(MatchPhase.SHOT_ACTIVE, tracks_frame, events, "SHOT_STARTED")
            elif self._stable_count >= self.config.stable_frames and not cue_stick_seen:
                self._transition(MatchPhase.STABLE_IDLE, tracks_frame, events, "SHOT_DISARMED")
        elif self.phase == MatchPhase.SHOT_ACTIVE:
            if self._stable_count >= max(2, self.config.stable_frames // 2):
                self._settle_count = 0
                self._transition(MatchPhase.SETTLING, tracks_frame, events, "SHOT_SETTLING")
        elif self.phase == MatchPhase.SETTLING:
            if stable:
                self._settle_count += 1
            else:
                self._settle_count = 0
            if self._settle_count >= self.config.settle_frames:
                self._transition(MatchPhase.TURN_RESOLVE, tracks_frame, events, "TURN_RESOLVE")
        elif self.phase == MatchPhase.TURN_RESOLVE:
            if self._stable_count >= max(2, self.config.stable_frames // 2):
                self._transition(MatchPhase.STABLE_IDLE, tracks_frame, events, "LAYOUT_STABLE")
        elif self.phase == MatchPhase.ANOMALY_RECOVERY:
            if self._stable_count >= self.config.stable_frames:
                self._transition(MatchPhase.STABLE_IDLE, tracks_frame, events, "ANOMALY_RECOVERED")

        for event in events:
            self._recent_events.append(event)

        confidence = 1.0
        if self.phase == MatchPhase.ANOMALY_RECOVERY:
            confidence = 0.35
        elif anomaly:
            confidence = 0.65
        return MatchStateFrame(
            frame_id=tracks_frame.frame_id,
            ts_cam_ns=tracks_frame.ts_cam_ns,
            phase=self.phase.value,
            events=events,
            layout=tracks,
            confidence=float(confidence),
            state_version=self.version,
        )

    def _transition(self, phase: MatchPhase, frame: TracksFrame, events: List[Event], name: str) -> None:
        if self.phase == phase and name not in {"ANOMALY"}:
            return
        self.phase = phase
        events.append(Event(name=name, ts_cam_ns=frame.ts_cam_ns, frame_id=frame.frame_id, confidence=1.0))

    def _queue_operator_event(
        self,
        name: str,
        *,
        frame_id: int = 0,
        ts_cam_ns: int = 0,
        payload: Optional[dict] = None,
    ) -> None:
        self._pending_operator_events.append(
            Event(
                name=name,
                ts_cam_ns=int(ts_cam_ns),
                frame_id=int(frame_id),
                payload=payload or {},
                confidence=1.0,
            )
        )

    def _drain_operator_events(self, frame: TracksFrame) -> List[Event]:
        events: List[Event] = []
        while self._pending_operator_events:
            event = self._pending_operator_events.popleft()
            if event.frame_id == 0:
                event.frame_id = frame.frame_id
            if event.ts_cam_ns == 0:
                event.ts_cam_ns = frame.ts_cam_ns
            events.append(event)
        return events

    def _speed(self, track: TrackObservation) -> float:
        vx, vy = track.velocity_px_s
        return float(np.hypot(vx, vy))

    def _is_moving(self, tracks: List[TrackObservation]) -> bool:
        return any(self._speed(t) >= self.config.moving_speed_px_s and t.quality > 0.25 for t in tracks)

    def _is_stable(self, tracks: List[TrackObservation]) -> bool:
        ball_tracks = [t for t in tracks if t.group in {"cue", "solid", "stripe", "black"}]
        if not ball_tracks:
            return False
        return all(self._speed(t) <= self.config.still_speed_px_s or t.quality < 0.25 for t in ball_tracks)

    def _detect_anomaly(self, tracks: List[TrackObservation]) -> bool:
        visible_balls = [t for t in tracks if t.group in {"cue", "solid", "stripe", "black"} and t.visibility == "visible"]
        cue_count = sum(1 for t in visible_balls if t.group == "cue")
        if cue_count > 1:
            return True
        if len(visible_balls) > 16:
            return True
        centers = np.asarray([t.center_px for t in visible_balls], dtype=np.float32)
        if centers.shape[0] >= 2:
            for i in range(centers.shape[0]):
                d = np.linalg.norm(centers[i + 1 :] - centers[i], axis=1)
                if np.any(d < 3.0):
                    return True
        return False

    def _detect_disappearances(self, tracks_frame: TracksFrame) -> List[Event]:
        events: List[Event] = []
        visible_ids = {t.track_id for t in tracks_frame.tracks if t.visibility == "visible"}
        for tid, memory in list(self._memory.items()):
            if tid in visible_ids or tid in self._potted_ids:
                continue
            missing = tracks_frame.frame_id - memory.last_seen_frame
            if missing < max(4, self.config.stable_frames // 2):
                continue
            if self.phase in {MatchPhase.SHOT_ACTIVE, MatchPhase.SETTLING, MatchPhase.TURN_RESOLVE} and memory.group != "cue_stick":
                self._potted_ids.add(tid)
                events.append(
                    Event(
                        name="BALL_DISAPPEARED",
                        ts_cam_ns=tracks_frame.ts_cam_ns,
                        frame_id=tracks_frame.frame_id,
                        payload={"track_id": tid, "group": memory.group, "last_center_px": list(memory.last_center_px)},
                        confidence=0.55,
                    )
                )
        return events

    def _update_memory(self, tracks_frame: TracksFrame) -> None:
        for t in tracks_frame.tracks:
            if t.visibility != "visible":
                continue
            self._memory[t.track_id] = _TrackMemory(
                track_id=t.track_id,
                group=t.group,
                last_center_px=t.center_px,
                last_seen_frame=tracks_frame.frame_id,
                last_seen_ts_ns=tracks_frame.ts_cam_ns,
                last_quality=t.quality,
            )
