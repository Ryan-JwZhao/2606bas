from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from ..config import StateConfig
from ..schemas import Event, MatchPhase, MatchStateFrame, TrackObservation, TracksFrame
from .models import InventoryLedger, MatchRuleState, RefereeIntent, ShotContext, normalize_group, normalize_object_group, other_object_group
from .phase import PhaseSignals, ShotPhaseMachine
from .pocket import PerBallPocketFSM
from .reconcile import ObservationReconciler
from .referee import RefereeAdapter, ShotContextAggregator, shot_context_payload
from .targeting import TargetGroupResolution, resolve_turn_target_group


@dataclass
class _PendingReviewDecision:
    confirm_shot_ctx: ShotContext
    confirm_ledger: InventoryLedger
    confirm_intent: RefereeIntent
    reject_shot_ctx: ShotContext
    reject_intent: RefereeIntent
    review_reasons: list[str]
    decision_ids: list[str]

    def to_payload(self) -> dict[str, object]:
        return {
            "decision_ids": list(self.decision_ids),
            "review_reasons": list(self.review_reasons),
            "group_choice_required": bool(self.confirm_intent.group_choice_required),
            "review_pockets": list(self.confirm_shot_ctx.review_pockets),
            "committed_pockets": list(self.confirm_shot_ctx.committed_pockets),
        }


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
        self._pending_review: Optional[_PendingReviewDecision] = None
        self._table_inner_polygon_mm: list[tuple[float, float]] = []
        self._pockets_mm: list[tuple[float, float]] = []
        self._pocket_curves_mm: list[list[tuple[float, float]]] = []
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
        return self._target_resolution().target_group

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
        self._pending_review = None

    def set_table_context(
        self,
        *,
        inner_polygon_mm: Optional[List[tuple[float, float]]] = None,
        pockets_mm: Optional[List[tuple[float, float]]] = None,
        ball_diameter_mm: Optional[float] = None,
        pocket_curves_mm: Optional[List[List[tuple[float, float]]]] = None,
    ) -> None:
        if inner_polygon_mm is not None:
            self._table_inner_polygon_mm = [(float(x), float(y)) for x, y in inner_polygon_mm]
        if pockets_mm is not None:
            self._pockets_mm = [(float(x), float(y)) for x, y in pockets_mm]
        if ball_diameter_mm is not None:
            self._ball_diameter_mm = float(ball_diameter_mm)
        if pocket_curves_mm is not None:
            self._pocket_curves_mm = [
                [(float(point[0]), float(point[1])) for point in list(curve or [])]
                for curve in pocket_curves_mm
            ]
        self.pocket_fsm.set_table_context(
            inner_polygon_mm=self._table_inner_polygon_mm,
            pockets_mm=self._pockets_mm,
            ball_diameter_mm=self._ball_diameter_mm,
            pocket_curves_mm=self._pocket_curves_mm,
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
        self.reject_episode(frame_id=frame_id, ts_cam_ns=ts_cam_ns, reason="operator_clear_review_flags")

    def confirm_episode(
        self,
        *,
        decision_id: Optional[str] = None,
        frame_id: int = 0,
        ts_cam_ns: int = 0,
        reason: str = "operator",
    ) -> bool:
        pending = self._pending_review
        if pending is None or not self._decision_matches_pending(pending, decision_id):
            self._queue_operator_event(
                "OPERATOR_CONFIRM_EPISODE_SKIPPED",
                frame_id=frame_id,
                ts_cam_ns=ts_cam_ns,
                payload={"decision_id": decision_id, "reason": reason},
            )
            return False
        previous_status = self.rule_state.game_status
        self.ledger = pending.confirm_ledger.clone()
        self._apply_intent(pending.confirm_intent)
        self.rule_state.shot_number += 1
        self._last_shot_payload = shot_context_payload(pending.confirm_shot_ctx)
        self._last_referee_payload = pending.confirm_intent.to_payload()
        self._pending_review = None
        for pocket in pending.confirm_shot_ctx.committed_pockets:
            confirmed_payload = self._confirmed_pocket_payload(pocket, reason_code="operator_confirmed_review")
            self._queue_operator_event("POCKET_CONFIRMED", frame_id=frame_id, ts_cam_ns=ts_cam_ns, payload=confirmed_payload)
            self._queue_operator_event("POT_PROBABLE", frame_id=frame_id, ts_cam_ns=ts_cam_ns, payload=confirmed_payload)
        self._queue_operator_event(
            "OPERATOR_CONFIRM_EPISODE",
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            payload={"decision_id": decision_id, "reason": reason},
        )
        self._queue_commit_status_events(
            previous_status=previous_status,
            intent=pending.confirm_intent,
            shot_ctx=pending.confirm_shot_ctx,
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
        )
        return True

    def reject_episode(
        self,
        *,
        decision_id: Optional[str] = None,
        frame_id: int = 0,
        ts_cam_ns: int = 0,
        reason: str = "operator",
    ) -> bool:
        pending = self._pending_review
        if pending is None or not self._decision_matches_pending(pending, decision_id):
            self._queue_operator_event(
                "OPERATOR_REJECT_EPISODE_SKIPPED",
                frame_id=frame_id,
                ts_cam_ns=ts_cam_ns,
                payload={"decision_id": decision_id, "reason": reason},
            )
            return False
        previous_status = self.rule_state.game_status
        self._apply_intent(pending.reject_intent)
        self.rule_state.shot_number += 1
        self._last_shot_payload = shot_context_payload(pending.reject_shot_ctx)
        self._last_referee_payload = pending.reject_intent.to_payload()
        self._pending_review = None
        self._queue_operator_event(
            "OPERATOR_REJECT_EPISODE",
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            payload={"decision_id": decision_id, "reason": reason},
        )
        self._queue_commit_status_events(
            previous_status=previous_status,
            intent=pending.reject_intent,
            shot_ctx=pending.reject_shot_ctx,
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
        )
        return True

    def resolve_open_table_group(
        self,
        group: Optional[str],
        *,
        frame_id: int = 0,
        ts_cam_ns: int = 0,
        reason: str = "operator",
    ) -> bool:
        pending = self._pending_review
        chosen = normalize_object_group(group)
        if pending is None or chosen is None or not pending.confirm_intent.group_choice_required:
            self._queue_operator_event(
                "OPERATOR_RESOLVE_OPEN_TABLE_GROUP_SKIPPED",
                frame_id=frame_id,
                ts_cam_ns=ts_cam_ns,
                payload={"group": group, "reason": reason},
            )
            return False
        previous_status = self.rule_state.game_status
        self.ledger = pending.confirm_ledger.clone()
        next_group = self.referee._target_for_actor(chosen, self.ledger)  # type: ignore[attr-defined]
        resolved_intent = RefereeIntent(
            next_group_hint=next_group,
            next_actor_changed=False,
            table_state_after="closed",
            actor_group_after=chosen,
            opponent_group_after=other_object_group(chosen),
            ball_in_hand_scope=pending.confirm_intent.ball_in_hand_scope,
            foul_flags=dict(pending.confirm_intent.foul_flags),
            review_required=False,
            group_choice_required=False,
            game_status=pending.confirm_intent.game_status,
            reasons=[str(reason_code) for reason_code in pending.confirm_intent.reasons if str(reason_code).strip() != "open_table_group_choice_required"],
        )
        self._apply_intent(resolved_intent)
        self.rule_state.shot_number += 1
        self._last_shot_payload = shot_context_payload(pending.confirm_shot_ctx)
        self._last_referee_payload = resolved_intent.to_payload()
        self._pending_review = None
        for pocket in pending.confirm_shot_ctx.committed_pockets:
            confirmed_payload = self._confirmed_pocket_payload(pocket, reason_code="operator_resolved_open_table_group")
            self._queue_operator_event("POCKET_CONFIRMED", frame_id=frame_id, ts_cam_ns=ts_cam_ns, payload=confirmed_payload)
            self._queue_operator_event("POT_PROBABLE", frame_id=frame_id, ts_cam_ns=ts_cam_ns, payload=confirmed_payload)
        self._queue_operator_event(
            "OPERATOR_RESOLVE_OPEN_TABLE_GROUP",
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
            payload={"group": chosen, "reason": reason},
        )
        self._queue_commit_status_events(
            previous_status=previous_status,
            intent=resolved_intent,
            shot_ctx=pending.confirm_shot_ctx,
            frame_id=frame_id,
            ts_cam_ns=ts_cam_ns,
        )
        return True

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
        snapshot["pending_review"] = {} if self._pending_review is None else self._pending_review.to_payload()
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
        self._annotate_shot_ids(events, ts_ms=ts_ms)
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
        commit_shot_ctx = self._confirmed_shot_context(shot_ctx)
        commit_ledger = self.ledger.applied_copy(commit_shot_ctx)
        reconcile_result = self.reconciler.reconcile(
            commit_ledger,
            supporting_events=[*self._recent_events, *events],
        )
        target_resolution = self._target_resolution(reconcile_result=reconcile_result)
        review_reasons = [str(item.get("mode", "")) for item in reconcile_result.mismatches if item.get("mode")]
        review_reasons.extend(target_resolution.reasons)
        commit_intent = self.referee.evaluate(
            commit_shot_ctx,
            commit_ledger,
            self.rule_state,
            effective_remaining=target_resolution.effective_remaining,
            review_reasons=[],
        )
        if commit_intent.group_choice_required and "open_table_group_choice_required" not in review_reasons:
            review_reasons.append("open_table_group_choice_required")
        current_target_resolution = self._target_resolution(
            reconcile_result=self.reconciler.current_observation_result(self.ledger)
        )
        intent = self.referee.evaluate(
            shot_ctx,
            self.ledger,
            self.rule_state,
            effective_remaining=current_target_resolution.effective_remaining,
            review_reasons=review_reasons,
        )
        turn_frozen = bool(shot_ctx.review_required or review_reasons or commit_intent.review_required or commit_intent.group_choice_required)
        previous_status = self.rule_state.game_status
        if turn_frozen:
            reject_shot_ctx = self._rejected_shot_context(shot_ctx)
            reject_intent = self.referee.evaluate(
                reject_shot_ctx,
                self.ledger,
                self.rule_state,
                effective_remaining=current_target_resolution.effective_remaining,
                review_reasons=[],
            )
            self._pending_review = _PendingReviewDecision(
                confirm_shot_ctx=commit_shot_ctx,
                confirm_ledger=commit_ledger,
                confirm_intent=commit_intent,
                reject_shot_ctx=reject_shot_ctx,
                reject_intent=reject_intent,
                review_reasons=list(review_reasons),
                decision_ids=self._review_decision_ids(commit_shot_ctx, shot_ctx),
            )
        else:
            self.ledger = commit_ledger.clone()
            self._apply_intent(commit_intent)
            self.rule_state.shot_number += 1
            self._pending_review = None
            for pocket in commit_shot_ctx.committed_pockets:
                confirmed_payload = self._confirmed_pocket_payload(pocket)
                events.append(
                    Event(
                        name="POCKET_CONFIRMED",
                        ts_cam_ns=tracks_frame.ts_cam_ns,
                        frame_id=tracks_frame.frame_id,
                        payload=confirmed_payload,
                        confidence=0.92,
                    )
                )
                events.append(
                    Event(
                        name="POT_PROBABLE",
                        ts_cam_ns=tracks_frame.ts_cam_ns,
                        frame_id=tracks_frame.frame_id,
                        payload=confirmed_payload,
                        confidence=0.92,
                    )
                )
            intent = commit_intent
        self._last_shot_payload = shot_context_payload(commit_shot_ctx if not turn_frozen else shot_ctx)
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
        if previous_status != intent.game_status:
            transition_payload = {
                "shot_id": (commit_shot_ctx if not turn_frozen else shot_ctx).shot_id,
                "decision_id": f"game-status:{(commit_shot_ctx if not turn_frozen else shot_ctx).shot_id}:{previous_status}->{intent.game_status}",
                "from_status": previous_status,
                "to_status": intent.game_status,
                "review_required": bool(intent.review_required),
                "reason_codes": [str(reason) for reason in intent.reasons if str(reason).strip()],
            }
            events.append(
                Event(
                    name="GAME_STATUS_CHANGED",
                    ts_cam_ns=tracks_frame.ts_cam_ns,
                    frame_id=tracks_frame.frame_id,
                    payload=transition_payload,
                    confidence=0.90,
                )
            )
        if previous_status != intent.game_status and intent.game_status != "in_progress":
            events.append(
                Event(
                    name="GAME_OVER_CANDIDATE",
                    ts_cam_ns=tracks_frame.ts_cam_ns,
                    frame_id=tracks_frame.frame_id,
                    payload={
                        "game_status": intent.game_status,
                        "reason": "black_confirmed",
                        "shot_id": (commit_shot_ctx if not turn_frozen else shot_ctx).shot_id,
                    },
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
            turn_target_group=self._target_resolution().target_group,
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
        target_resolution = self._target_resolution()
        return {
            "frame_id": int(tracks_frame.frame_id),
            "ts_cam_ns": int(tracks_frame.ts_cam_ns),
            "phase": str(state.phase),
            "state_version": str(state.state_version),
            "operator_hold": bool(self._operator_hold),
            "turn_target_group": target_resolution.target_group,
            "raw_turn_target_group": self._normalized_turn_target_group(),
            "target_resolution": target_resolution.to_payload(),
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

    def _apply_intent(self, intent: RefereeIntent) -> None:
        self.rule_state.table_state = intent.table_state_after
        self.rule_state.actor_group = intent.actor_group_after
        self.rule_state.opponent_group = intent.opponent_group_after
        self.rule_state.game_status = intent.game_status
        self._turn_target_group = intent.next_group_hint

    def _confirmed_shot_context(self, shot_ctx: ShotContext) -> ShotContext:
        confirmed = deepcopy(shot_ctx)
        for pocket in list(confirmed.review_pockets):
            promoted = dict(pocket)
            promoted["decision"] = "confirmed"
            promoted["review_required"] = False
            promoted["reason_codes"] = ["operator_confirmed_review"]
            confirmed.committed_pockets.append(promoted)
            group = normalize_group(promoted.get("group"))
            if group is not None:
                confirmed.potted_confirmed[group] = int(confirmed.potted_confirmed.get(group, 0)) + 1
                if group == "cue":
                    confirmed.cue_scratch_candidate = True
        confirmed.review_pockets = []
        confirmed.review_required = False
        confirmed.tentative_pockets = []
        confirmed.reasons = []
        return confirmed

    def _rejected_shot_context(self, shot_ctx: ShotContext) -> ShotContext:
        rejected = deepcopy(shot_ctx)
        rejected.committed_pockets = []
        rejected.review_pockets = []
        rejected.tentative_pockets = []
        rejected.review_required = False
        rejected.reasons = []
        rejected.potted_confirmed = {group: 0 for group in rejected.potted_confirmed}
        return rejected

    @staticmethod
    def _review_decision_ids(confirm_shot_ctx: ShotContext, original_shot_ctx: ShotContext) -> list[str]:
        decision_ids: list[str] = []
        for source in (
            original_shot_ctx.review_pockets,
            original_shot_ctx.committed_pockets,
            confirm_shot_ctx.committed_pockets,
        ):
            for pocket in source:
                decision_id = str(pocket.get("decision_id") or "").strip()
                if decision_id and decision_id not in decision_ids:
                    decision_ids.append(decision_id)
        return decision_ids

    @staticmethod
    def _decision_matches_pending(pending: _PendingReviewDecision, decision_id: Optional[str]) -> bool:
        if decision_id is None:
            return True
        normalized = str(decision_id).strip()
        return bool(normalized) and normalized in pending.decision_ids

    @staticmethod
    def _confirmed_pocket_payload(pocket: dict[str, object], *, reason_code: str = "committed_pocket") -> dict[str, object]:
        payload = dict(pocket)
        payload["decision"] = "confirmed"
        payload["review_required"] = False
        payload["reason_codes"] = [reason_code]
        return payload

    def _queue_commit_status_events(
        self,
        *,
        previous_status: str,
        intent: RefereeIntent,
        shot_ctx: ShotContext,
        frame_id: int,
        ts_cam_ns: int,
    ) -> None:
        self._queue_operator_event("SHOT_CONTEXT_FINALIZED", frame_id=frame_id, ts_cam_ns=ts_cam_ns, payload=self._last_shot_payload)
        self._queue_operator_event("REFEREE_INTENT", frame_id=frame_id, ts_cam_ns=ts_cam_ns, payload=self._last_referee_payload)
        if previous_status != intent.game_status:
            transition_payload = {
                "shot_id": shot_ctx.shot_id,
                "decision_id": f"game-status:{shot_ctx.shot_id}:{previous_status}->{intent.game_status}",
                "from_status": previous_status,
                "to_status": intent.game_status,
                "review_required": bool(intent.review_required),
                "reason_codes": [str(reason) for reason in intent.reasons if str(reason).strip()],
            }
            self._queue_operator_event("GAME_STATUS_CHANGED", frame_id=frame_id, ts_cam_ns=ts_cam_ns, payload=transition_payload)
            if intent.game_status != "in_progress":
                self._queue_operator_event(
                    "GAME_OVER_CANDIDATE",
                    frame_id=frame_id,
                    ts_cam_ns=ts_cam_ns,
                    payload={"game_status": intent.game_status, "reason": "black_confirmed", "shot_id": shot_ctx.shot_id},
                )

    def _annotate_shot_ids(self, events: List[Event], *, ts_ms: int) -> None:
        needs_shot_context = any(
            event.name
            in {
                "POCKET_CANDIDATE",
                "POCKET_TENTATIVE",
                "POCKET_COMMIT_READY",
                "POCKET_CONFIRMED",
                "POCKET_REVIEW_REQUIRED",
                "POCKET_REJECTED",
            }
            for event in events
        )
        if not needs_shot_context:
            return
        ctx = self.aggregator.active
        if ctx is None:
            ctx = self.aggregator.begin_if_needed(ts_ms=ts_ms, rule_state=self.rule_state)
        for event in events:
            if event.name not in {
                "POCKET_CANDIDATE",
                "POCKET_TENTATIVE",
                "POCKET_COMMIT_READY",
                "POCKET_CONFIRMED",
                "POCKET_REVIEW_REQUIRED",
                "POCKET_REJECTED",
            }:
                continue
            event.payload = dict(event.payload or {})
            event.payload.setdefault("shot_id", ctx.shot_id)

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

    def _target_resolution(self, *, reconcile_result: object | None = None) -> TargetGroupResolution:
        result = reconcile_result if reconcile_result is not None else self.reconciler.current_observation_result(self.ledger)
        return resolve_turn_target_group(
            self._normalized_turn_target_group(),
            actor_group=self.rule_state.actor_group,
            ledger_remaining=self.ledger.remaining,
            observation_effective_remaining=getattr(result, "effective_remaining", {}),
            visible_counts=getattr(result, "visible_counts", {}),
            stable_frames=getattr(result, "stable_frames", {}),
            stable_frames_required=int(getattr(self.config, "observation_reconcile_stable_frames", 12)),
            observation_review_required=bool(getattr(result, "review_required", False)),
        )

    @staticmethod
    def _ts_ms(ts_cam_ns: int) -> int:
        return int(round(int(ts_cam_ns) / 1_000_000.0))

    @staticmethod
    def _elapsed_ms(now_ns: int, since_ns: int) -> int:
        return max(0, int(round((int(now_ns) - int(since_ns)) / 1_000_000.0)))
