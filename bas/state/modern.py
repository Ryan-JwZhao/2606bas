from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

from ..config import StateConfig
from ..schemas import Event, MatchPhase, MatchStateFrame, TrackObservation, TracksFrame
from .models import InventoryLedger, MatchRuleState, normalize_object_group, other_object_group
from .phase import PhaseSignals, ShotPhaseMachine
from .pocket import PerBallPocketFSM
from .reconcile import ObservationReconciler
from .referee import RefereeAdapter, ShotContextAggregator, shot_context_payload


class ModernMatchStateMachine:
    version = "billiards_state_new_v2"

    def __init__(self, config: StateConfig):
        self.config = config
        self.phase_machine = ShotPhaseMachine(config)
        self.pocket_fsm = PerBallPocketFSM(config)
        self.aggregator = ShotContextAggregator()
        self.ledger = InventoryLedger()
        self.rule_state = MatchRuleState()
        self.referee = RefereeAdapter()
        self.reconciler = ObservationReconciler(config)
        self._turn_target_group: Optional[str] = None
        self._snapshot_layout: List[TrackObservation] = []
        self._last_layout: List[TrackObservation] = []
        self._operator_hold = False
        self._operator_lock_frames = 0
        self._pending_operator_events: Deque[Event] = deque(maxlen=32)
        self._recent_events: Deque[Event] = deque(maxlen=160)
        self._event_cooldowns: Dict[str, int] = {}
        self._last_debug_snapshot: Dict[str, object] = {}
        self._last_referee_payload: dict[str, object] = {}
        self._last_shot_payload: dict[str, object] = {}
        self._pending_turn_resolve: Optional[dict[str, int]] = None
        self._table_inner_polygon_mm: list[tuple[float, float]] = []
        self._pockets_mm: list[tuple[float, float]] = []
        self._ball_diameter_mm = 57.15

    @property
    def phase(self) -> MatchPhase:
        return self.phase_machine.phase

    @phase.setter
    def phase(self, value: MatchPhase | str) -> None:
        target = value if isinstance(value, MatchPhase) else MatchPhase(str(value))
        self.phase_machine.force(target)

    @property
    def operator_hold(self) -> bool:
        return self._operator_hold

    @property
    def turn_target_group(self) -> Optional[str]:
        return self._normalized_turn_target_group()

    def reset(self) -> None:
        self.phase_machine.reset()
        self.pocket_fsm.reset()
        self.aggregator.reset()
        self.ledger.reset()
        self.rule_state.reset()
        self.reconciler.reset()
        self._turn_target_group = None
        self._snapshot_layout = []
        self._last_layout = []
        self._operator_hold = False
        self._operator_lock_frames = 0
        self._pending_operator_events.clear()
        self._recent_events.clear()
        self._event_cooldowns.clear()
        self._last_debug_snapshot = {}
        self._last_referee_payload = {}
        self._last_shot_payload = {}
        self._pending_turn_resolve = None

    def set_table_context(
        self,
        *,
        inner_polygon_mm: Optional[List[tuple[float, float]]] = None,
        pockets_mm: Optional[List[tuple[float, float]]] = None,
        ball_diameter_mm: Optional[float] = None,
    ) -> None:
        if inner_polygon_mm is not None:
            self._table_inner_polygon_mm = [(float(x), float(y)) for x, y in inner_polygon_mm]
        if pockets_mm is not None:
            self._pockets_mm = [(float(x), float(y)) for x, y in pockets_mm]
        if ball_diameter_mm is not None:
            self._ball_diameter_mm = float(ball_diameter_mm)
        self.pocket_fsm.set_table_context(
            inner_polygon_mm=self._table_inner_polygon_mm,
            pockets_mm=self._pockets_mm,
            ball_diameter_mm=self._ball_diameter_mm,
        )

    def force_phase(self, phase: MatchPhase | str, *, frame_id: int = 0, ts_cam_ns: int = 0, reason: str = "operator") -> None:
        target = phase if isinstance(phase, MatchPhase) else MatchPhase(str(phase))
        self.phase_machine.force(target)
        self._operator_lock_frames = max(self._operator_lock_frames, 2)
        self._queue_operator_event(
            "OPERATOR_FORCE_PHASE",
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            payload={"phase": target.value, "reason": reason},
        )
        if target == MatchPhase.TURN_RESOLVE:
            self._queue_operator_event("TURN_RESOLVE", frame_id=frame_id, ts_cam_ns=ts_cam_ns, payload={"source": "operator"})

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
        self._queue_operator_event("OPERATOR_CLEAR_REVIEW_FLAGS", frame_id=frame_id, ts_cam_ns=ts_cam_ns)
        self._last_referee_payload = {}

    def set_turn_target_group(
        self,
        group: Optional[str],
        *,
        frame_id: int = 0,
        ts_cam_ns: int = 0,
        reason: str = "operator",
    ) -> None:
        normalized = str(group or "").strip().lower()
        self._turn_target_group = normalized if normalized in {"solid", "stripe", "black"} else None
        object_group = normalize_object_group(normalized)
        if object_group is not None:
            self.rule_state.table_state = "closed"
            self.rule_state.actor_group = object_group
            self.rule_state.opponent_group = other_object_group(object_group)
        self._queue_operator_event(
            "OPERATOR_SET_TURN_TARGET_GROUP",
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            payload={"group": self._turn_target_group, "reason": reason},
        )

    def debug_snapshot(self) -> Dict[str, object]:
        snapshot = dict(self._last_debug_snapshot)
        snapshot["signals"] = dict(snapshot.get("signals", {}))
        snapshot["counters"] = dict(snapshot.get("counters", {}))
        snapshot["cooldowns"] = dict(snapshot.get("cooldowns", {}))
        snapshot["visible_group_counts"] = dict(snapshot.get("visible_group_counts", {}))
        snapshot["visible_track_ids"] = list(snapshot.get("visible_track_ids", []))
        snapshot["ledger"] = dict(self.ledger.remaining)
        snapshot["removed_confirmed"] = dict(self.ledger.removed_confirmed)
        snapshot["rule_state"] = {
            "table_state": self.rule_state.table_state,
            "actor_group": self.rule_state.actor_group,
            "opponent_group": self.rule_state.opponent_group,
            "shot_number": self.rule_state.shot_number,
            "game_status": self.rule_state.game_status,
        }
        snapshot["referee_intent"] = dict(self._last_referee_payload)
        snapshot["last_shot_context"] = dict(self._last_shot_payload)
        snapshot["pocket_fsm"] = self.pocket_fsm.debug_snapshot()
        snapshot["observation_reconcile"] = self.reconciler.event_payload(self.reconciler.last_result)
        snapshot["pending_turn_resolve"] = dict(self._pending_turn_resolve or {})
        return snapshot

    def update(self, tracks_frame: TracksFrame) -> MatchStateFrame:
        tracks = tracks_frame.tracks
        self._last_layout = list(tracks)
        events: List[Event] = self._drain_operator_events(tracks_frame)
        if self._operator_hold:
            self.reconciler.update_observation(tracks_frame)
            for event in events:
                self._recent_events.append(event)
            layout = self._snapshot_layout if self._snapshot_layout else tracks
            state = self._state_frame(tracks_frame, events, layout, confidence=0.70, suffix="+operator_hold")
            self._last_debug_snapshot = self._build_debug_snapshot(
                tracks_frame,
                state,
                self._empty_signals(),
                operator_hold_short_circuit=True,
            )
            return state

        if self._operator_lock_frames > 0:
            self._operator_lock_frames -= 1
            self._tick_event_cooldowns()
            self.reconciler.update_observation(tracks_frame)
            self._process_turn_resolve_if_needed(tracks_frame, events)
            for event in events:
                self._recent_events.append(event)
            state = self._state_frame(tracks_frame, events, tracks, confidence=0.85, suffix="+operator_override")
            self._last_debug_snapshot = self._build_debug_snapshot(
                tracks_frame,
                state,
                self._empty_signals(),
                operator_override_short_circuit=True,
            )
            return state

        self._tick_event_cooldowns()
        signals, sensed_events = self._sense(tracks_frame)
        events.extend(sensed_events)
        phase_before = self.phase_machine.phase
        phase = self.phase_machine.update(tracks_frame, signals, events)
        if self._pending_turn_resolve is not None and phase != MatchPhase.ANOMALY_RECOVERY:
            self.phase_machine.force(MatchPhase.TURN_RESOLVE)
            phase = MatchPhase.TURN_RESOLVE
        if any(event.name == "SHOT_STARTED" for event in events):
            self._snapshot_layout = list(tracks)
        self.reconciler.update_observation(tracks_frame)
        pocket_phase = MatchPhase.TURN_RESOLVE if self._pending_turn_resolve is not None else phase
        pocket_events = self.pocket_fsm.update(tracks_frame, pocket_phase)
        events.extend(pocket_events)
        ts_ms = self._ts_ms(tracks_frame.ts_cam_ns)
        self.aggregator.ingest(events, ts_ms=ts_ms, rule_state=self.rule_state)
        self._process_turn_resolve_if_needed(tracks_frame, events)

        for event in events:
            self._recent_events.append(event)

        confidence = 1.0
        if phase == MatchPhase.ANOMALY_RECOVERY:
            confidence = 0.35
        elif signals.anomaly:
            confidence = 0.65
        state = self._state_frame(tracks_frame, events, tracks, confidence=confidence)
        self._last_debug_snapshot = self._build_debug_snapshot(tracks_frame, state, signals)
        return state

    def _process_turn_resolve_if_needed(self, tracks_frame: TracksFrame, events: List[Event]) -> None:
        resolve_events = [event for event in events if event.name == "TURN_RESOLVE"]
        if resolve_events and self._should_defer_turn_resolve(tracks_frame, resolve_events):
            self._start_pending_turn_resolve(tracks_frame, events)
            return
        if not resolve_events and self._pending_turn_resolve is None:
            return
        if self._pending_turn_resolve is not None and self._should_keep_waiting_for_pockets(tracks_frame):
            return
        pending = self._pending_turn_resolve
        self._pending_turn_resolve = None
        ts_ms = self._ts_ms(tracks_frame.ts_cam_ns)
        shot_ctx = self.aggregator.finalize(ts_ms=ts_ms, rule_state=self.rule_state)
        self.ledger.apply(shot_ctx)
        reconcile_result = self.reconciler.reconcile(
            self.ledger,
            supporting_events=[*self._recent_events, *events],
        )
        review_reasons = [str(item.get("mode", "")) for item in reconcile_result.mismatches if item.get("mode")]
        intent = self.referee.evaluate(
            shot_ctx,
            self.ledger,
            self.rule_state,
            effective_remaining=reconcile_result.effective_remaining,
            review_reasons=review_reasons,
        )
        self.rule_state.table_state = intent.table_state_after
        self.rule_state.actor_group = intent.actor_group_after
        self.rule_state.opponent_group = intent.opponent_group_after
        self.rule_state.game_status = intent.game_status
        self.rule_state.shot_number += 1
        self._turn_target_group = intent.next_group_hint
        self._last_shot_payload = shot_context_payload(shot_ctx)
        self._last_referee_payload = intent.to_payload()
        if reconcile_result.mismatches:
            events.append(
                Event(
                    name="LEDGER_OBSERVATION_MISMATCH",
                    ts_cam_ns=tracks_frame.ts_cam_ns,
                    frame_id=tracks_frame.frame_id,
                    payload=self.reconciler.event_payload(reconcile_result),
                    confidence=0.70,
                )
            )
        if pending is not None:
            events.append(
                Event(
                    name="TURN_RESOLVE_COMMITTED",
                    ts_cam_ns=tracks_frame.ts_cam_ns,
                    frame_id=tracks_frame.frame_id,
                    payload={
                        "deferred_from_frame_id": int(pending.get("frame_id", tracks_frame.frame_id)),
                        "deferred_ms": self._elapsed_ms(tracks_frame.ts_cam_ns, int(pending.get("ts_cam_ns", tracks_frame.ts_cam_ns))),
                    },
                    confidence=1.0,
                )
            )
        events.append(
            Event(
                name="SHOT_CONTEXT_FINALIZED",
                ts_cam_ns=tracks_frame.ts_cam_ns,
                frame_id=tracks_frame.frame_id,
                payload=self._last_shot_payload,
                confidence=0.95,
            )
        )
        events.append(
            Event(
                name="REFEREE_INTENT",
                ts_cam_ns=tracks_frame.ts_cam_ns,
                frame_id=tracks_frame.frame_id,
                payload=self._last_referee_payload,
                confidence=0.85,
            )
        )
        if intent.game_status != "in_progress":
            events.append(
                Event(
                    name="GAME_OVER_CANDIDATE",
                    ts_cam_ns=tracks_frame.ts_cam_ns,
                    frame_id=tracks_frame.frame_id,
                    payload={"game_status": intent.game_status, "reason": "black_confirmed"},
                    confidence=0.80,
                )
            )

    def _should_defer_turn_resolve(self, tracks_frame: TracksFrame, resolve_events: List[Event]) -> bool:
        if self._pending_turn_resolve is not None:
            return False
        if any(str((event.payload or {}).get("source", "")).strip().lower() == "operator" for event in resolve_events):
            return False
        if not self.pocket_fsm.has_pending_resolution(tracks_frame.ts_cam_ns):
            return False
        return int(getattr(self.config, "turn_resolve_grace_ms", 900)) > 0

    def _start_pending_turn_resolve(self, tracks_frame: TracksFrame, events: List[Event]) -> None:
        grace_ms = max(1, int(getattr(self.config, "turn_resolve_grace_ms", 900)))
        self._pending_turn_resolve = {
            "frame_id": int(tracks_frame.frame_id),
            "ts_cam_ns": int(tracks_frame.ts_cam_ns),
            "deadline_ns": int(tracks_frame.ts_cam_ns) + grace_ms * 1_000_000,
        }
        events.append(
            Event(
                name="TURN_RESOLVE_DEFERRED",
                ts_cam_ns=tracks_frame.ts_cam_ns,
                frame_id=tracks_frame.frame_id,
                payload={
                    "grace_ms": grace_ms,
                    "pending_pockets": self.pocket_fsm.pending_candidates(tracks_frame.ts_cam_ns),
                },
                confidence=1.0,
            )
        )

    def _should_keep_waiting_for_pockets(self, tracks_frame: TracksFrame) -> bool:
        pending = self._pending_turn_resolve
        if pending is None:
            return False
        if int(tracks_frame.ts_cam_ns) >= int(pending.get("deadline_ns", tracks_frame.ts_cam_ns)):
            return False
        return self.pocket_fsm.has_pending_resolution(tracks_frame.ts_cam_ns)

    def _sense(self, tracks_frame: TracksFrame) -> tuple[PhaseSignals, List[Event]]:
        tracks = tracks_frame.tracks
        moving = self._is_moving(tracks)
        stable = self._is_stable(tracks)
        anomaly = self._detect_anomaly(tracks)
        cue_stick_seen = any(t.group == "cue_stick" and t.visibility == "visible" and t.quality > 0.35 for t in tracks)
        cue_motion = any(t.group == "cue" and self._speed(t) > self._moving_threshold(t) for t in tracks)
        events: List[Event] = []
        events.extend(self._detect_ball_collisions(tracks_frame))
        events.extend(self._detect_rail_collisions(tracks_frame))
        shot_vote = self._detect_shot_start_vote(tracks_frame)
        if shot_vote is not None:
            events.append(shot_vote)
        return PhaseSignals(
            moving=bool(moving),
            stable=bool(stable),
            anomaly=bool(anomaly),
            cue_stick_seen=bool(cue_stick_seen),
            cue_motion=bool(cue_motion),
            shot_start_voted=shot_vote is not None,
        ), events

    def _state_frame(
        self,
        tracks_frame: TracksFrame,
        events: List[Event],
        layout: List[TrackObservation],
        *,
        confidence: float,
        suffix: str = "",
    ) -> MatchStateFrame:
        return MatchStateFrame(
            frame_id=tracks_frame.frame_id,
            ts_cam_ns=tracks_frame.ts_cam_ns,
            phase=self.phase_machine.phase.value,
            events=events,
            layout=layout,
            turn_target_group=self._turn_target_group,
            confidence=float(confidence),
            state_version=f"{self.version}{suffix}",
        )

    def _queue_operator_event(
        self,
        name: str,
        *,
        frame_id: int = 0,
        ts_cam_ns: int = 0,
        payload: Optional[dict] = None,
    ) -> None:
        self._pending_operator_events.append(
            Event(name=name, ts_cam_ns=int(ts_cam_ns), frame_id=int(frame_id), payload=payload or {}, confidence=1.0)
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

    def _detect_ball_collisions(self, tracks_frame: TracksFrame) -> List[Event]:
        events: List[Event] = []
        balls = [
            t
            for t in tracks_frame.tracks
            if t.group in {"cue", "solid", "stripe", "black"} and t.visibility == "visible" and t.quality > 0.25
        ]
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
                both_mm = a.center_mm is not None and b.center_mm is not None
                contact_limit = ra + rb + float(self.config.collision_epsilon_mm if both_mm else 6.0)
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
                            "group_a": a.group,
                            "group_b": b.group,
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
            now_normal = float(np.dot(vel, inward))
            if abs(now_normal) < self._still_threshold(track):
                continue
            events.append(
                Event(
                    name="RAIL_COLLISION_CANDIDATE",
                    ts_cam_ns=tracks_frame.ts_cam_ns,
                    frame_id=tracks_frame.frame_id,
                    payload={"track_id": track.track_id, "side": side, "distance_to_rail": dist, "limit": float(limit)},
                    confidence=0.58,
                )
            )
            self._event_cooldowns[key] = int(self.config.event_cooldown_frames)
        return events

    def _detect_shot_start_vote(self, tracks_frame: TracksFrame) -> Optional[Event]:
        cue = next((t for t in tracks_frame.tracks if t.group == "cue" and t.visibility == "visible" and t.quality > 0.25), None)
        if cue is None:
            return None
        cue_speed = self._speed(cue)
        votes = {
            "cue_ball_motion": cue_speed >= self._moving_threshold(cue),
            "cue_ball_accel_peak": cue_speed >= float(self.config.shot_speed_jump_mm_s if cue.velocity_mm_s is not None else self.config.moving_speed_px_s),
            "cue_stick_seen": any(t.group == "cue_stick" and t.visibility == "visible" and t.quality > 0.25 for t in tracks_frame.tracks),
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
            payload={"cue_track_id": cue.track_id, "votes": votes, "cue_speed": cue_speed},
            confidence=0.78,
        )

    def _is_moving(self, tracks: List[TrackObservation]) -> bool:
        return any(
            t.group in {"cue", "solid", "stripe", "black"}
            and self._speed(t) >= self._moving_threshold(t)
            and t.quality > 0.25
            for t in tracks
        )

    def _is_stable(self, tracks: List[TrackObservation]) -> bool:
        ball_tracks = [t for t in tracks if t.group in {"cue", "solid", "stripe", "black"}]
        if not ball_tracks:
            return False
        return all(self._speed(t) <= self._still_threshold(t) or t.quality < 0.25 for t in ball_tracks)

    def _detect_anomaly(self, tracks: List[TrackObservation]) -> bool:
        visible_balls = [
            t
            for t in tracks
            if t.group in {"cue", "solid", "stripe", "black"} and t.visibility == "visible" and t.quality > 0.25
        ]
        if sum(1 for t in visible_balls if t.group == "cue") > 1:
            return True
        if len(visible_balls) > 16:
            return True
        return False

    def _speed(self, track: TrackObservation) -> float:
        vx, vy = self._velocity(track)
        return float(np.hypot(vx, vy))

    def _moving_threshold(self, track: TrackObservation) -> float:
        return float(self.config.moving_speed_mm_s if track.velocity_mm_s is not None else self.config.moving_speed_px_s)

    def _still_threshold(self, track: TrackObservation) -> float:
        return float(self.config.still_speed_mm_s if track.velocity_mm_s is not None else self.config.still_speed_px_s)

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

    def _heading_changed(self, _track: TrackObservation) -> bool:
        return False

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

    def _build_debug_snapshot(
        self,
        tracks_frame: TracksFrame,
        state: MatchStateFrame,
        signals: PhaseSignals,
        *,
        operator_hold_short_circuit: bool = False,
        operator_override_short_circuit: bool = False,
    ) -> Dict[str, object]:
        visible_tracks = [track for track in tracks_frame.tracks if track.visibility == "visible" and track.quality > 0.25]
        return {
            "frame_id": int(tracks_frame.frame_id),
            "ts_cam_ns": int(tracks_frame.ts_cam_ns),
            "phase": str(state.phase),
            "state_version": str(state.state_version),
            "operator_hold": bool(self._operator_hold),
            "turn_target_group": self._normalized_turn_target_group(),
            "signals": {
                "moving": bool(signals.moving),
                "stable": bool(signals.stable),
                "anomaly": bool(signals.anomaly),
                "cue_stick_seen": bool(signals.cue_stick_seen),
                "cue_motion": bool(signals.cue_motion),
                "shot_start_voted": bool(signals.shot_start_voted),
                "operator_hold_short_circuit": bool(operator_hold_short_circuit),
                "operator_override_short_circuit": bool(operator_override_short_circuit),
            },
            "counters": {**self.phase_machine.counters(), "operator_lock_frames": int(self._operator_lock_frames)},
            "cooldowns": {str(key): int(value) for key, value in sorted(self._event_cooldowns.items())},
            "visible_group_counts": self._debug_visible_group_counts(visible_tracks),
            "visible_track_ids": [int(track.track_id) for track in visible_tracks],
        }

    def _empty_signals(self) -> PhaseSignals:
        return PhaseSignals(False, False, False, False, False, False)

    @staticmethod
    def _debug_visible_group_counts(tracks: List[TrackObservation]) -> Dict[str, int]:
        counts = {"cue": 0, "solid": 0, "stripe": 0, "black": 0, "cue_stick": 0, "other": 0}
        for track in tracks:
            group = str(track.group).strip().lower()
            counts[group if group in counts else "other"] += 1
        return counts

    def _normalized_turn_target_group(self) -> Optional[str]:
        group = str(self._turn_target_group or "").strip().lower()
        return group if group in {"solid", "stripe", "black"} else None

    @staticmethod
    def _ts_ms(ts_cam_ns: int) -> int:
        return int(round(int(ts_cam_ns) / 1_000_000.0))

    @staticmethod
    def _elapsed_ms(now_ns: int, since_ns: int) -> int:
        return max(0, int(round((int(now_ns) - int(since_ns)) / 1_000_000.0)))
