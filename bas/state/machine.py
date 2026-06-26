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
    last_center_mm: Optional[tuple[float, float]]
    last_velocity: tuple[float, float]
    last_radius: float
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
        self._event_cooldowns: Dict[str, int] = {}
        self._table_inner_polygon_mm: List[tuple[float, float]] = []
        self._pockets_mm: List[tuple[float, float]] = []
        self._ball_diameter_mm = 57.15

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
        self._event_cooldowns.clear()

    def set_table_context(
        self,
        *,
        inner_polygon_mm: Optional[List[tuple[float, float]]] = None,
        pockets_mm: Optional[List[tuple[float, float]]] = None,
        ball_diameter_mm: Optional[float] = None,
    ) -> None:
        if inner_polygon_mm:
            self._table_inner_polygon_mm = [(float(x), float(y)) for x, y in inner_polygon_mm]
        if pockets_mm:
            self._pockets_mm = [(float(x), float(y)) for x, y in pockets_mm]
        if ball_diameter_mm is not None:
            self._ball_diameter_mm = float(ball_diameter_mm)

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
        if self._operator_hold:
            self._update_memory(tracks_frame)
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
            self._update_memory(tracks_frame)
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

        moving = self._is_moving(tracks)
        stable = self._is_stable(tracks)
        anomaly = self._detect_anomaly(tracks)
        self._tick_event_cooldowns()
        events.extend(self._detect_ball_collisions(tracks_frame))
        events.extend(self._detect_rail_collisions(tracks_frame))
        shot_vote_event = self._detect_shot_start_vote(tracks_frame)
        shot_start_voted = shot_vote_event is not None
        if shot_vote_event is not None:
            events.append(shot_vote_event)
        events.extend(self._detect_disappearances(tracks_frame))
        self._update_memory(tracks_frame)

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
            elif shot_start_voted or cue_motion or self._moving_count >= 2:
                self._snapshot_layout = list(tracks)
                self._transition(MatchPhase.SHOT_ACTIVE, tracks_frame, events, "SHOT_STARTED")
        elif self.phase == MatchPhase.PRE_SHOT_ARMED:
            if shot_start_voted or cue_motion or self._moving_count >= 2:
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
        vx, vy = track.velocity_mm_s if track.velocity_mm_s is not None else track.velocity_px_s
        return float(np.hypot(vx, vy))

    def _is_moving(self, tracks: List[TrackObservation]) -> bool:
        return any(self._speed(t) >= self._moving_threshold(t) and t.quality > 0.25 for t in tracks)

    def _is_stable(self, tracks: List[TrackObservation]) -> bool:
        ball_tracks = [t for t in tracks if t.group in {"cue", "solid", "stripe", "black"}]
        if not ball_tracks:
            return False
        return all(self._speed(t) <= self._still_threshold(t) or t.quality < 0.25 for t in ball_tracks)

    def _moving_threshold(self, track: TrackObservation) -> float:
        return float(self.config.moving_speed_mm_s if track.velocity_mm_s is not None else self.config.moving_speed_px_s)

    def _still_threshold(self, track: TrackObservation) -> float:
        return float(self.config.still_speed_mm_s if track.velocity_mm_s is not None else self.config.still_speed_px_s)

    def _detect_anomaly(self, tracks: List[TrackObservation]) -> bool:
        visible_balls = [
            t
            for t in tracks
            if t.group in {"cue", "solid", "stripe", "black"} and t.visibility == "visible" and t.quality > 0.25
        ]
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
                pot_payload = self._pot_payload(memory)
                if pot_payload is not None:
                    events.append(
                        Event(
                            name="POT_PROBABLE",
                            ts_cam_ns=tracks_frame.ts_cam_ns,
                            frame_id=tracks_frame.frame_id,
                            payload=pot_payload,
                            confidence=0.75,
                        )
                    )
                    continue
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
                last_center_mm=t.center_mm,
                last_velocity=self._velocity(t),
                last_radius=self._radius(t),
                last_seen_frame=tracks_frame.frame_id,
                last_seen_ts_ns=tracks_frame.ts_cam_ns,
                last_quality=t.quality,
            )

    def _detect_ball_collisions(self, tracks_frame: TracksFrame) -> List[Event]:
        events: List[Event] = []
        balls = [t for t in tracks_frame.tracks if t.group in {"cue", "solid", "stripe", "black"} and t.visibility == "visible" and t.quality > 0.25]
        for i, a in enumerate(balls):
            pa = self._position(a)
            va = np.asarray(self._velocity(a), dtype=np.float32)
            ra = self._radius(a)
            for b in balls[i + 1 :]:
                key = self._cooldown_key("BALL_COLLISION_CANDIDATE", a.track_id, b.track_id)
                if key in self._event_cooldowns:
                    continue
                pb = self._position(b)
                vb = np.asarray(self._velocity(b), dtype=np.float32)
                rb = self._radius(b)
                delta = pb - pa
                dist = float(np.linalg.norm(delta))
                if dist <= 1e-6:
                    continue
                contact_limit = ra + rb + float(self.config.collision_epsilon_mm if a.center_mm is not None and b.center_mm is not None else 6.0)
                if dist > contact_limit:
                    continue
                normal = delta / dist
                closing_speed = float(np.dot(va - vb, normal))
                heading_change = self._heading_changed(a) or self._heading_changed(b)
                if closing_speed < max(5.0, self._still_threshold(a)) and not heading_change:
                    continue
                events.append(
                    Event(
                        name="BALL_COLLISION_CANDIDATE",
                        ts_cam_ns=tracks_frame.ts_cam_ns,
                        frame_id=tracks_frame.frame_id,
                        payload={
                            "track_a": a.track_id,
                            "track_b": b.track_id,
                            "distance": dist,
                            "contact_limit": float(contact_limit),
                            "closing_speed": closing_speed,
                            "heading_change": bool(heading_change),
                        },
                        confidence=0.70 if heading_change else 0.55,
                    )
                )
                self._event_cooldowns[key] = int(self.config.event_cooldown_frames)
        return events

    def _detect_rail_collisions(self, tracks_frame: TracksFrame) -> List[Event]:
        events: List[Event] = []
        bounds = self._table_bounds()
        if bounds is None:
            return events
        x_min, y_min, x_max, y_max = bounds
        for track in tracks_frame.tracks:
            if track.group not in {"cue", "solid", "stripe", "black"} or track.visibility != "visible" or track.quality <= 0.25:
                continue
            key = self._cooldown_key("RAIL_COLLISION_CANDIDATE", track.track_id)
            if key in self._event_cooldowns:
                continue
            pos = self._position(track)
            vel = np.asarray(self._velocity(track), dtype=np.float32)
            radius = self._radius(track)
            distances = [
                ("left", float(pos[0] - x_min), np.asarray([1.0, 0.0], dtype=np.float32)),
                ("right", float(x_max - pos[0]), np.asarray([-1.0, 0.0], dtype=np.float32)),
                ("top", float(pos[1] - y_min), np.asarray([0.0, 1.0], dtype=np.float32)),
                ("bottom", float(y_max - pos[1]), np.asarray([0.0, -1.0], dtype=np.float32)),
            ]
            side, dist, inward = min(distances, key=lambda item: item[1])
            limit = radius + float(self.config.rail_epsilon_mm if track.center_mm is not None else 8.0)
            if dist > limit:
                continue
            memory = self._memory.get(track.track_id)
            prev_normal = float(np.dot(np.asarray(memory.last_velocity, dtype=np.float32), inward)) if memory is not None else 0.0
            now_normal = float(np.dot(vel, inward))
            if abs(now_normal) < self._still_threshold(track) and abs(prev_normal) < self._still_threshold(track):
                continue
            bounced = prev_normal < -self._still_threshold(track) * 0.5 and now_normal > self._still_threshold(track) * 0.5
            events.append(
                Event(
                    name="RAIL_COLLISION_CANDIDATE",
                    ts_cam_ns=tracks_frame.ts_cam_ns,
                    frame_id=tracks_frame.frame_id,
                    payload={
                        "track_id": track.track_id,
                        "side": side,
                        "distance_to_rail": dist,
                        "limit": float(limit),
                        "normal_before": prev_normal,
                        "normal_after": now_normal,
                        "bounced": bool(bounced),
                    },
                    confidence=0.75 if bounced else 0.55,
                )
            )
            self._event_cooldowns[key] = int(self.config.event_cooldown_frames)
        return events

    def _detect_shot_start_vote(self, tracks_frame: TracksFrame) -> Optional[Event]:
        cue = next((t for t in tracks_frame.tracks if t.group == "cue" and t.visibility == "visible" and t.quality > 0.25), None)
        if cue is None:
            return None
        cue_pos = self._position(cue)
        cue_speed = self._speed(cue)
        cue_memory = self._memory.get(cue.track_id)
        prev_speed = float(np.linalg.norm(np.asarray(cue_memory.last_velocity, dtype=np.float32))) if cue_memory is not None else 0.0
        dt = max(1e-3, (tracks_frame.ts_cam_ns - cue_memory.last_seen_ts_ns) / 1e9) if cue_memory is not None else 1.0

        tip_distance = None
        tip_near = False
        for stick in [t for t in tracks_frame.tracks if t.group == "cue_stick" and t.visibility == "visible" and t.quality > 0.25]:
            tip = self._cue_tip_candidate(stick, cue_pos)
            dist = float(np.linalg.norm(tip - cue_pos))
            tip_distance = dist if tip_distance is None else min(tip_distance, dist)
            limit = max(self._radius(cue) * float(self.config.shot_tip_radius_multiplier), 60.0 if cue.center_mm is None else 0.0)
            if dist <= limit:
                tip_near = True
                break

        speed_jump = cue_speed - prev_speed >= float(self.config.shot_speed_jump_mm_s if cue.velocity_mm_s is not None else self.config.moving_speed_px_s * 0.35)
        accel = (cue_speed - prev_speed) / dt
        accel_peak = accel >= float(self.config.shot_accel_mm_s2 if cue.velocity_mm_s is not None else self.config.moving_speed_px_s * 3.0)
        votes = {
            "tip_near_cue_ball": bool(tip_near),
            "cue_ball_speed_jump": bool(speed_jump),
            "cue_ball_accel_peak": bool(accel_peak),
        }
        if sum(1 for value in votes.values() if value) < 2:
            return None
        key = self._cooldown_key("SHOT_START_VOTED", cue.track_id)
        if key in self._event_cooldowns:
            return None
        self._event_cooldowns[key] = int(max(2, self.config.event_cooldown_frames))
        return Event(
            name="SHOT_START_VOTED",
            ts_cam_ns=tracks_frame.ts_cam_ns,
            frame_id=tracks_frame.frame_id,
            payload={
                "cue_track_id": cue.track_id,
                "votes": votes,
                "vote_count": sum(1 for value in votes.values() if value),
                "cue_speed": cue_speed,
                "prev_cue_speed": prev_speed,
                "cue_accel": accel,
                "tip_distance": tip_distance,
            },
            confidence=0.82,
        )

    def _cue_tip_candidate(self, stick: TrackObservation, cue_pos: np.ndarray) -> np.ndarray:
        if stick.center_mm is not None and stick.radius_mm is not None:
            center = self._position(stick)
            velocity = np.asarray(self._velocity(stick), dtype=np.float32)
            if float(np.linalg.norm(velocity)) > 1e-6:
                axis = velocity / max(1e-6, float(np.linalg.norm(velocity)))
                candidates = [center + axis * stick.radius_mm * 2.0, center - axis * stick.radius_mm * 2.0]
                return min(candidates, key=lambda p: float(np.linalg.norm(p - cue_pos)))
        x1, y1, x2, y2 = stick.bbox
        if abs(x2 - x1) >= abs(y2 - y1):
            candidates = [np.asarray([x1, (y1 + y2) * 0.5], dtype=np.float32), np.asarray([x2, (y1 + y2) * 0.5], dtype=np.float32)]
        else:
            candidates = [np.asarray([(x1 + x2) * 0.5, y1], dtype=np.float32), np.asarray([(x1 + x2) * 0.5, y2], dtype=np.float32)]
        return min(candidates, key=lambda p: float(np.linalg.norm(p - cue_pos)))

    def _pot_payload(self, memory: _TrackMemory) -> Optional[dict]:
        if memory.last_center_mm is None or not self._pockets_mm:
            return None
        pos = np.asarray(memory.last_center_mm, dtype=np.float32)
        vel = np.asarray(memory.last_velocity, dtype=np.float32)
        pockets = np.asarray(self._pockets_mm, dtype=np.float32).reshape((-1, 2))
        distances = np.linalg.norm(pockets - pos.reshape((1, 2)), axis=1)
        idx = int(np.argmin(distances))
        dist = float(distances[idx])
        if dist > float(self.config.pocket_funnel_radius_mm):
            return None
        toward = 0.0
        speed = float(np.linalg.norm(vel))
        if speed > 1e-6:
            direction = pockets[idx] - pos
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                toward = float(np.dot(vel / speed, direction / norm))
        if speed >= self.config.still_speed_mm_s and toward < 0.2:
            return None
        return {
            "track_id": memory.track_id,
            "group": memory.group,
            "pocket_index": idx,
            "distance_to_pocket_mm": dist,
            "last_center_mm": list(memory.last_center_mm),
            "last_center_px": list(memory.last_center_px),
            "speed": speed,
            "toward_pocket": toward,
        }

    def _position(self, track: TrackObservation) -> np.ndarray:
        point = track.center_mm if track.center_mm is not None else track.center_px
        return np.asarray(point, dtype=np.float32)

    def _velocity(self, track: TrackObservation) -> tuple[float, float]:
        velocity = track.velocity_mm_s if track.velocity_mm_s is not None else track.velocity_px_s
        return (float(velocity[0]), float(velocity[1]))

    def _radius(self, track: TrackObservation) -> float:
        if track.radius_mm is not None and track.radius_mm > 0:
            return float(track.radius_mm)
        if track.center_mm is not None:
            return float(self._ball_diameter_mm * 0.5)
        return float(track.radius_px)

    def _heading_changed(self, track: TrackObservation) -> bool:
        memory = self._memory.get(track.track_id)
        if memory is None:
            return False
        prev = np.asarray(memory.last_velocity, dtype=np.float32)
        now = np.asarray(self._velocity(track), dtype=np.float32)
        if float(np.linalg.norm(prev)) < self._still_threshold(track) or float(np.linalg.norm(now)) < self._still_threshold(track):
            return False
        cos = float(np.dot(prev, now) / max(1e-6, float(np.linalg.norm(prev) * np.linalg.norm(now))))
        return cos < 0.86

    def _table_bounds(self) -> Optional[tuple[float, float, float, float]]:
        if len(self._table_inner_polygon_mm) < 3:
            return None
        pts = np.asarray(self._table_inner_polygon_mm, dtype=np.float32).reshape((-1, 2))
        return (float(np.min(pts[:, 0])), float(np.min(pts[:, 1])), float(np.max(pts[:, 0])), float(np.max(pts[:, 1])))

    def _cooldown_key(self, name: str, *ids: int) -> str:
        return f"{name}:{':'.join(str(i) for i in sorted(ids))}"

    def _tick_event_cooldowns(self) -> None:
        for key in list(self._event_cooldowns.keys()):
            self._event_cooldowns[key] -= 1
            if self._event_cooldowns[key] <= 0:
                del self._event_cooldowns[key]
