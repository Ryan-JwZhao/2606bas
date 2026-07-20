from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import StateConfig
from ..schemas import Event, MatchPhase, TrackObservation, TracksFrame
from .models import Group, normalize_group
from .pocket_geometry import PocketGeometryContext, PocketGeometryModel, PocketSample


@dataclass
class PocketEvidence:
    decision_id: str
    pocket_index: int
    candidate_since_ns: int
    last_evidence_ts_ns: int
    candidate_reason: Optional[str] = None
    crossed_mouth: bool = False
    crossed_throat: bool = False
    entered_interior: bool = False


@dataclass
class _PocketMemory:
    track_id: int
    group: Group
    last_center_px: tuple[float, float]
    last_center_mm: tuple[float, float]
    last_velocity: tuple[float, float]
    last_seen_frame: int
    last_seen_ts_ns: int
    last_quality: float
    state: str = "on_table"
    decision: str = "none"
    resolved: bool = False
    evidence: Optional[PocketEvidence] = None
    tentative_since_ns: Optional[int] = None
    commit_ready_since_ns: Optional[int] = None
    nonvisible_since_ns: Optional[int] = None
    absent_since_ns: Optional[int] = None
    resting_mouth_since_ns: Optional[int] = None
    tentative_emitted: bool = False
    commit_ready_emitted: bool = False
    last_zone: Optional[str] = None
    last_visibility: str = "visible"
    last_lost_frames: int = 0
    last_distance_mm: Optional[float] = None
    last_depth_mm: float = 0.0
    last_lateral_mm: float = 0.0
    last_inward_speed_mm_s: float = 0.0
    reason_codes: list[str] = field(default_factory=list)
    reappear_veto: bool = False
    reappear_track_id: Optional[int] = None
    reappear_group: Optional[Group] = None


class PerBallPocketFSM:
    """Per-ball pocket evidence with retractable commit-ready decisions."""

    _REAPPEAR_STATES = {"candidate", "tentative", "commit_ready"}
    _TERMINAL_STATES = {"confirmed", "review_required", "rejected"}

    def __init__(self, config: StateConfig):
        self.config = config
        self.inner_polygon_mm: list[tuple[float, float]] = []
        self.table_edge_polygon_mm: list[tuple[float, float]] = []
        self.ball_center_reachable_polygon_mm: list[tuple[float, float]] = []
        self.pockets_mm: list[tuple[float, float]] = []
        self.pocket_curves_mm: list[list[tuple[float, float]]] = []
        self.ball_diameter_mm = 57.15
        self._memory: Dict[int, _PocketMemory] = {}
        self._decision_seq = 1
        self._geometry_context_fingerprint: Optional[tuple[object, ...]] = None
        self._geometry_model = PocketGeometryModel.build(
            PocketGeometryContext(),
            config,
            log_diagnostics=False,
        )

    def reset(self) -> None:
        self._memory.clear()
        self._decision_seq = 1

    def set_table_context(
        self,
        *,
        inner_polygon_mm: Optional[list[tuple[float, float]]] = None,
        table_edge_polygon_mm: Optional[list[tuple[float, float]]] = None,
        ball_center_reachable_polygon_mm: Optional[list[tuple[float, float]]] = None,
        pockets_mm: Optional[list[tuple[float, float]]] = None,
        ball_diameter_mm: Optional[float] = None,
        pocket_curves_mm: Optional[list[list[tuple[float, float]]]] = None,
    ) -> None:
        if inner_polygon_mm is not None:
            self.inner_polygon_mm = [(float(x), float(y)) for x, y in inner_polygon_mm]
        if table_edge_polygon_mm is not None:
            self.table_edge_polygon_mm = [(float(x), float(y)) for x, y in table_edge_polygon_mm]
        elif inner_polygon_mm is not None and not self.table_edge_polygon_mm:
            self.table_edge_polygon_mm = list(self.inner_polygon_mm)
        if ball_center_reachable_polygon_mm is not None:
            self.ball_center_reachable_polygon_mm = [
                (float(x), float(y)) for x, y in ball_center_reachable_polygon_mm
            ]
        elif inner_polygon_mm is not None and not self.ball_center_reachable_polygon_mm:
            self.ball_center_reachable_polygon_mm = list(self.inner_polygon_mm)
        if pockets_mm is not None:
            self.pockets_mm = [(float(x), float(y)) for x, y in pockets_mm]
        if ball_diameter_mm is not None:
            self.ball_diameter_mm = float(ball_diameter_mm)
        if pocket_curves_mm is not None:
            self.pocket_curves_mm = [
                [(float(point[0]), float(point[1])) for point in list(curve or []) if len(point) >= 2]
                for curve in pocket_curves_mm
            ]
        context = PocketGeometryContext(
            table_edge_polygon_mm=self.table_edge_polygon_mm or self.inner_polygon_mm,
            ball_center_reachable_polygon_mm=self.ball_center_reachable_polygon_mm or self.inner_polygon_mm,
            pockets_mm=self.pockets_mm,
            pocket_curves_mm=self.pocket_curves_mm,
            ball_diameter_mm=self.ball_diameter_mm,
        )
        fingerprint = context.fingerprint()
        if fingerprint == self._geometry_context_fingerprint:
            return
        self._geometry_context_fingerprint = fingerprint
        self._geometry_model = PocketGeometryModel.build(context, self.config)

    def update(self, frame: TracksFrame, phase: MatchPhase) -> List[Event]:
        events: List[Event] = []
        active = phase in {MatchPhase.SHOT_ACTIVE, MatchPhase.SETTLING, MatchPhase.TURN_RESOLVE}
        seen_ids: set[int] = set()

        for track in frame.tracks:
            group = normalize_group(track.group)
            if group is None or not self._track_relevant(track):
                continue
            memory = self._memory.get(track.track_id)
            memory_created_this_frame = False
            if memory is None:
                if track.visibility != "visible" or float(track.quality) <= 0.25:
                    continue
                memory = self._new_memory(track, group, frame)
                self._memory[track.track_id] = memory
                memory_created_this_frame = True
            elif memory.resolved and track.visibility == "visible":
                memory = self._new_memory(track, group, frame)
                self._memory[track.track_id] = memory
                memory_created_this_frame = True

            seen_ids.add(track.track_id)
            if track.visibility == "visible" and float(track.quality) > 0.25:
                sample = self._sample(track)
                cross_track_vetoed = bool(
                    memory_created_this_frame
                    and self._apply_cross_track_reappear_veto(memory, track, sample, frame, events)
                )
                self._handle_visible(memory, track, sample, frame, active and not cross_track_vetoed, events)
            elif active:
                self._handle_nonvisible(memory, track, frame, events)

        if active:
            for memory in list(self._memory.values()):
                if memory.track_id in seen_ids or memory.resolved:
                    continue
                self._handle_absent(memory, frame, events)

        return events

    def debug_snapshot(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for memory in self._memory.values():
            evidence = memory.evidence
            rows.append(
                {
                    "track_id": int(memory.track_id),
                    "group": memory.group,
                    "state": memory.state,
                    "decision": memory.decision,
                    "decision_id": evidence.decision_id if evidence is not None else None,
                    "pocket_index": evidence.pocket_index if evidence is not None else None,
                    "crossed_mouth": bool(evidence and evidence.crossed_mouth),
                    "crossed_throat": bool(evidence and evidence.crossed_throat),
                    "entered_interior": bool(evidence and evidence.entered_interior),
                    "last_zone": memory.last_zone,
                    "zone": memory.last_zone,
                    "inward_speed_mm_s": float(memory.last_inward_speed_mm_s),
                    "candidate_reason": evidence.candidate_reason if evidence is not None else None,
                    "missing_ms": self._memory_missing_ms(memory, memory.last_seen_ts_ns),
                    "reappear_veto": bool(memory.reappear_veto),
                    "reappear_group": memory.reappear_group,
                    "reason_codes": list(memory.reason_codes),
                }
            )
        return rows

    def geometry_diagnostics(self) -> dict[str, object]:
        return self._geometry_model.diagnostics()

    def review_pending(
        self,
        frame: TracksFrame,
        *,
        reason_codes: Optional[list[str]] = None,
    ) -> List[Event]:
        events: List[Event] = []
        reasons = list(reason_codes or ["final_confirmation_window_not_reached"])
        for memory in self._memory.values():
            if memory.resolved or memory.evidence is None:
                continue
            if memory.state not in {"candidate", "tentative", "commit_ready"}:
                continue
            self._review(memory, frame, events, reason_codes=reasons)
        return events

    def has_pending_resolution(self, now_ns: int) -> bool:
        return bool(self.pending_candidates(now_ns))

    def pending_candidates(self, now_ns: int) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for memory in self._memory.values():
            evidence = memory.evidence
            if memory.resolved or evidence is None:
                continue
            if memory.state not in {"candidate", "tentative", "commit_ready"}:
                continue
            missing_ms = self._memory_missing_ms(memory, now_ns)
            if memory.state == "commit_ready" and missing_ms >= self._final_confirmation_missing_ms():
                continue
            rows.append(
                {
                    "track_id": int(memory.track_id),
                    "group": memory.group,
                    "state": memory.state,
                    "decision_id": evidence.decision_id,
                    "pocket_index": evidence.pocket_index,
                    "missing_ms": int(missing_ms),
                    "crossed_mouth": bool(evidence.crossed_mouth),
                    "crossed_throat": bool(evidence.crossed_throat),
                    "entered_interior": bool(evidence.entered_interior),
                    "candidate_reason": evidence.candidate_reason,
                    "finalizable_at_missing_ms": self._final_confirmation_missing_ms(),
                }
            )
        return rows

    def mark_confirmed(self, decision_ids: list[str]) -> None:
        wanted = {str(value) for value in decision_ids if str(value).strip()}
        for memory in self._memory.values():
            evidence = memory.evidence
            if evidence is None or evidence.decision_id not in wanted or memory.state == "rejected":
                continue
            memory.state = "confirmed"
            memory.decision = "confirmed"
            memory.resolved = True
            memory.reason_codes = ["confirmed_at_turn_resolve"]

    def mark_review_required(self, decision_ids: list[str]) -> None:
        wanted = {str(value) for value in decision_ids if str(value).strip()}
        for memory in self._memory.values():
            evidence = memory.evidence
            if evidence is None or evidence.decision_id not in wanted or memory.resolved:
                continue
            memory.state = "review_required"
            memory.decision = "review_required"
            memory.resolved = True
            memory.reason_codes = ["state_frozen_pending_review"]

    def mark_rejected(self, decision_ids: list[str]) -> None:
        wanted = {str(value) for value in decision_ids if str(value).strip()}
        for memory in self._memory.values():
            evidence = memory.evidence
            if evidence is None or evidence.decision_id not in wanted or memory.state == "confirmed":
                continue
            memory.state = "rejected"
            memory.decision = "rejected"
            memory.resolved = True
            memory.reason_codes = ["operator_rejected_review"]

    def _new_memory(self, track: TrackObservation, group: Group, frame: TracksFrame) -> _PocketMemory:
        return _PocketMemory(
            track_id=track.track_id,
            group=group,
            last_center_px=track.center_px,
            last_center_mm=self._center_mm(track),
            last_velocity=self._velocity(track),
            last_seen_frame=frame.frame_id,
            last_seen_ts_ns=frame.ts_cam_ns,
            last_quality=float(track.quality),
        )

    def _handle_visible(
        self,
        memory: _PocketMemory,
        track: TrackObservation,
        sample: PocketSample,
        frame: TracksFrame,
        active: bool,
        events: List[Event],
    ) -> None:
        reappeared_now = False
        if (
            memory.nonvisible_since_ns is not None
            and not memory.resolved
            and memory.evidence is not None
            and memory.state in {"on_table", *self._REAPPEAR_STATES}
        ):
            missing_ms = self._elapsed_ms(frame.ts_cam_ns, memory.nonvisible_since_ns)
            reappeared_group = normalize_group(track.group)
            reason_codes = ["reappeared_same_track"]
            self._emit_reappeared(memory, track.track_id, frame, missing_ms, events)
            if reappeared_group is not None and reappeared_group != memory.group:
                memory.reappear_group = reappeared_group
                reason_codes.append("reappeared_group_changed")
                self._review(memory, frame, events, reason_codes=reason_codes)
            else:
                self._reject(memory, frame, events, reason_codes=reason_codes, candidate_reason="reappeared_visible")
            reappeared_now = True

        if active and not reappeared_now:
            self._update_candidate_evidence(memory, track, sample, frame, events)

        if active and sample.zone is None and memory.evidence is not None and not memory.resolved:
            if memory.state in {"candidate", "tentative", "commit_ready"}:
                self._reject(memory, frame, events, reason_codes=["back_to_table"], candidate_reason="back_to_table")
            elif not memory.tentative_emitted:
                memory.evidence = None
                memory.resting_mouth_since_ns = None

        if active and sample.zone == "mouth" and self._speed(track) <= self._still_speed(track) * 1.25:
            if memory.resting_mouth_since_ns is None:
                memory.resting_mouth_since_ns = frame.ts_cam_ns
        elif sample.zone not in {"mouth", "throat", "interior"}:
            memory.resting_mouth_since_ns = None

        if memory.evidence is None:
            memory.group = normalize_group(track.group) or memory.group
        memory.last_center_px = track.center_px
        memory.last_center_mm = self._center_mm(track)
        memory.last_velocity = self._velocity(track)
        memory.last_seen_frame = frame.frame_id
        memory.last_seen_ts_ns = frame.ts_cam_ns
        memory.last_quality = float(track.quality)
        memory.last_zone = sample.zone
        memory.last_visibility = "visible"
        memory.last_lost_frames = 0
        memory.last_distance_mm = sample.distance_mm
        memory.last_depth_mm = float(sample.depth_mm)
        memory.last_lateral_mm = float(sample.lateral_mm)
        memory.last_inward_speed_mm_s = float(sample.pocketward_speed_mm_s)
        memory.nonvisible_since_ns = None
        memory.absent_since_ns = None

    def _handle_nonvisible(self, memory: _PocketMemory, track: TrackObservation, frame: TracksFrame, events: List[Event]) -> None:
        memory.last_seen_frame = frame.frame_id
        memory.last_seen_ts_ns = frame.ts_cam_ns
        memory.last_visibility = str(track.visibility or "occluded")
        memory.last_lost_frames = int(getattr(track, "lost_frames", 0))
        if not self._has_pocket_interest(memory):
            return
        if memory.nonvisible_since_ns is None:
            memory.nonvisible_since_ns = frame.ts_cam_ns
        self._advance_missing(memory, frame, events, allow_occluded_commit=self._occluded_commit_allowed(memory))

    def _handle_absent(self, memory: _PocketMemory, frame: TracksFrame, events: List[Event]) -> None:
        if not self._has_pocket_interest(memory):
            return
        if memory.absent_since_ns is None:
            memory.absent_since_ns = frame.ts_cam_ns
        if memory.nonvisible_since_ns is None:
            memory.nonvisible_since_ns = memory.absent_since_ns
        self._advance_missing(memory, frame, events, allow_occluded_commit=True)

    def _update_candidate_evidence(
        self,
        memory: _PocketMemory,
        track: TrackObservation,
        sample: PocketSample,
        frame: TracksFrame,
        events: List[Event],
    ) -> None:
        if sample.zone not in {"mouth", "throat", "interior"} or sample.pocket_index is None:
            return
        reason = self._candidate_reason(track, sample)
        if memory.evidence is None and reason is None:
            return
        if memory.evidence is not None and memory.evidence.pocket_index != sample.pocket_index:
            self._reject(
                memory,
                frame,
                events,
                reason_codes=["pocket_changed_before_resolution"],
                candidate_reason="pocket_changed_before_resolution",
            )
            return
        if memory.evidence is None:
            memory.evidence = PocketEvidence(
                decision_id=self._new_decision_id(),
                pocket_index=int(sample.pocket_index),
                candidate_since_ns=frame.ts_cam_ns,
                last_evidence_ts_ns=frame.ts_cam_ns,
            )
        evidence = memory.evidence
        evidence.last_evidence_ts_ns = frame.ts_cam_ns
        evidence.crossed_mouth = True
        if sample.zone in {"throat", "interior"}:
            evidence.crossed_throat = True
        if sample.zone == "interior":
            evidence.entered_interior = True

        if reason is None:
            return
        if evidence.candidate_reason is None:
            evidence.candidate_reason = reason
        if memory.state == "on_table":
            memory.state = "candidate"
            events.append(
                Event(
                    name="POCKET_CANDIDATE",
                    ts_cam_ns=frame.ts_cam_ns,
                    frame_id=frame.frame_id,
                    payload={
                        "track_id": memory.track_id,
                        "logical_id": f"track:{memory.track_id}",
                        "group": memory.group,
                        "pocket_index": evidence.pocket_index,
                        "decision_id": evidence.decision_id,
                        "candidate_reason": evidence.candidate_reason,
                        "distance_to_pocket_mm": sample.distance_mm,
                        "evidence": self._evidence_payload(memory, frame.ts_cam_ns),
                    },
                    confidence=0.72 if sample.zone in {"throat", "interior"} else 0.58,
                )
            )

    def _advance_missing(
        self,
        memory: _PocketMemory,
        frame: TracksFrame,
        events: List[Event],
        *,
        allow_occluded_commit: bool,
    ) -> None:
        missing_ms = self._memory_missing_ms(memory, frame.ts_cam_ns)
        if missing_ms >= self._tentative_missing_ms() and not memory.tentative_emitted:
            self._tentative(memory, frame, events, reason_codes=self._tentative_reason_codes(memory))
        if missing_ms < self._commit_ready_missing_ms() or memory.state == "commit_ready" or memory.resolved:
            return
        if self._strong_confirmation_evidence(memory) and not memory.reappear_veto:
            if allow_occluded_commit:
                self._commit_ready(memory, frame, events)
            return
        self._review(memory, frame, events, reason_codes=self._review_reason_codes(memory))

    def _tentative(
        self,
        memory: _PocketMemory,
        frame: TracksFrame,
        events: List[Event],
        *,
        reason_codes: list[str],
    ) -> None:
        if memory.tentative_emitted or memory.evidence is None:
            return
        memory.state = "tentative"
        memory.decision = "tentative"
        memory.tentative_since_ns = frame.ts_cam_ns
        memory.tentative_emitted = True
        memory.reason_codes = list(reason_codes)
        events.append(
            Event(
                name="POCKET_TENTATIVE",
                ts_cam_ns=frame.ts_cam_ns,
                frame_id=frame.frame_id,
                payload=self._decision_payload(
                    memory,
                    frame.ts_cam_ns,
                    decision="tentative",
                    review_required=False,
                    reason_codes=reason_codes,
                ),
                confidence=0.62,
            )
        )

    def _commit_ready(self, memory: _PocketMemory, frame: TracksFrame, events: List[Event]) -> None:
        if memory.commit_ready_emitted or memory.evidence is None:
            return
        memory.state = "commit_ready"
        memory.decision = "commit_ready"
        memory.commit_ready_since_ns = frame.ts_cam_ns
        memory.commit_ready_emitted = True
        memory.reason_codes = ["ready_for_turn_commit"]
        payload = self._decision_payload(
            memory,
            frame.ts_cam_ns,
            decision="commit_ready",
            review_required=False,
            reason_codes=["ready_for_turn_commit"],
        )
        events.append(
            Event(
                name="POCKET_COMMIT_READY",
                ts_cam_ns=frame.ts_cam_ns,
                frame_id=frame.frame_id,
                payload=payload,
                confidence=0.92,
            )
        )

    def _review(self, memory: _PocketMemory, frame: TracksFrame, events: List[Event], *, reason_codes: list[str]) -> None:
        if memory.resolved or memory.evidence is None:
            return
        memory.state = "review_required"
        memory.decision = "review_required"
        memory.resolved = True
        memory.reason_codes = list(reason_codes)
        events.append(
            Event(
                name="POCKET_REVIEW_REQUIRED",
                ts_cam_ns=frame.ts_cam_ns,
                frame_id=frame.frame_id,
                payload=self._decision_payload(
                    memory,
                    frame.ts_cam_ns,
                    decision="review_required",
                    review_required=True,
                    reason_codes=reason_codes,
                ),
                confidence=0.70,
            )
        )

    def _reject(
        self,
        memory: _PocketMemory,
        frame: TracksFrame,
        events: List[Event],
        *,
        reason_codes: list[str],
        candidate_reason: str,
    ) -> None:
        if memory.resolved or memory.evidence is None:
            return
        memory.state = "rejected"
        memory.decision = "rejected"
        memory.resolved = True
        memory.evidence.candidate_reason = candidate_reason
        memory.reason_codes = list(reason_codes)
        events.append(
            Event(
                name="POCKET_REJECTED",
                ts_cam_ns=frame.ts_cam_ns,
                frame_id=frame.frame_id,
                payload=self._decision_payload(
                    memory,
                    frame.ts_cam_ns,
                    decision="rejected",
                    review_required=False,
                    reason_codes=reason_codes,
                ),
                confidence=0.82,
            )
        )

    def _apply_cross_track_reappear_veto(
        self,
        memory: _PocketMemory,
        track: TrackObservation,
        sample: PocketSample,
        frame: TracksFrame,
        events: List[Event],
    ) -> bool:
        if sample.pocket_index is None:
            return False
        vetoed = False
        center_mm = self._center_mm(track)
        for candidate in self._memory.values():
            evidence = candidate.evidence
            if candidate.track_id == memory.track_id or candidate.resolved or evidence is None:
                continue
            if candidate.state not in {"on_table", *self._REAPPEAR_STATES}:
                continue
            if evidence.pocket_index != sample.pocket_index or candidate.nonvisible_since_ns is None:
                continue
            if self._distance(candidate.last_center_mm, center_mm) > self._reappear_match_distance_mm():
                continue
            candidate.reappear_veto = True
            candidate.reappear_track_id = int(track.track_id)
            candidate.reappear_group = memory.group
            self._emit_reappeared(
                candidate,
                track.track_id,
                frame,
                self._memory_missing_ms(candidate, frame.ts_cam_ns),
                events,
            )
            if candidate.group != memory.group:
                self._review(
                    candidate,
                    frame,
                    events,
                    reason_codes=["reappeared_near_pocket", "track_relinked", "reappeared_group_changed"],
                )
            else:
                self._reject(
                    candidate,
                    frame,
                    events,
                    reason_codes=["reappeared_new_track", "track_relinked"],
                    candidate_reason="reappeared_near_pocket",
                )
            vetoed = True
        return vetoed

    def _emit_reappeared(
        self,
        memory: _PocketMemory,
        track_id: int,
        frame: TracksFrame,
        missing_ms: int,
        events: List[Event],
    ) -> None:
        evidence = memory.evidence
        if evidence is None:
            return
        events.append(
            Event(
                name="POCKET_REAPPEARED",
                ts_cam_ns=frame.ts_cam_ns,
                frame_id=frame.frame_id,
                payload={
                    "track_id": memory.track_id,
                    "group": memory.group,
                    "reappeared_track_id": int(track_id),
                    "reappeared_group": memory.reappear_group,
                    "pocket_index": evidence.pocket_index,
                    "decision_id": evidence.decision_id,
                    "missing_ms": int(missing_ms),
                },
                confidence=0.88,
            )
        )

    def _decision_payload(
        self,
        memory: _PocketMemory,
        now_ns: int,
        *,
        decision: str,
        review_required: bool,
        reason_codes: list[str],
    ) -> dict[str, object]:
        evidence = memory.evidence
        if evidence is None:
            return {}
        return {
            "track_id": memory.track_id,
            "logical_id": f"track:{memory.track_id}",
            "group": memory.group,
            "pocket_index": evidence.pocket_index,
            "decision_id": evidence.decision_id,
            "decision": decision,
            "review_required": bool(review_required),
            "reason_codes": list(reason_codes),
            "candidate_reason": evidence.candidate_reason,
            "last_center_mm": list(memory.last_center_mm),
            "last_center_px": list(memory.last_center_px),
            "missing_ms": self._memory_missing_ms(memory, now_ns),
            "confirmation": "two_stage_commit",
            "evidence": self._evidence_payload(memory, now_ns),
        }

    def _evidence_payload(self, memory: _PocketMemory, now_ns: int) -> dict[str, object]:
        evidence = memory.evidence
        return {
            "decision_id": evidence.decision_id if evidence is not None else None,
            "candidate_since_ns": evidence.candidate_since_ns if evidence is not None else None,
            "last_evidence_ts_ns": evidence.last_evidence_ts_ns if evidence is not None else None,
            "zone": memory.last_zone,
            "inward_speed_mm_s": float(memory.last_inward_speed_mm_s),
            "candidate_reason": evidence.candidate_reason if evidence is not None else None,
            "missing_ms": self._memory_missing_ms(memory, now_ns),
            "reappear_veto": bool(memory.reappear_veto),
            "reappear_track_id": memory.reappear_track_id,
            "reappear_group": memory.reappear_group,
            "crossed_mouth": bool(evidence and evidence.crossed_mouth),
            "crossed_throat": bool(evidence and evidence.crossed_throat),
            "entered_interior": bool(evidence and evidence.entered_interior),
            "pocket_index": evidence.pocket_index if evidence is not None else None,
            "distance_mm": memory.last_distance_mm,
            "depth_mm": float(memory.last_depth_mm),
            "lateral_mm": float(memory.last_lateral_mm),
            "last_visibility": memory.last_visibility,
            "lost_frames": int(memory.last_lost_frames),
            "decision": memory.decision,
        }

    def _sample(self, track: TrackObservation) -> PocketSample:
        return self._geometry_model.sample(self._center_mm(track), self._velocity(track))

    def _candidate_reason(self, track: TrackObservation, sample: PocketSample) -> Optional[str]:
        if sample.zone == "interior":
            return "entered_interior"
        if sample.zone == "throat":
            return "crossed_throat"
        if sample.zone == "mouth" and sample.pocketward_speed_mm_s >= self._inward_speed_threshold(track):
            return "mouth_inward_trend"
        return None

    def _tentative_reason_codes(self, memory: _PocketMemory) -> list[str]:
        evidence = memory.evidence
        if evidence is not None and evidence.entered_interior:
            return ["interior_missing"]
        if evidence is not None and evidence.crossed_throat:
            return ["throat_missing"]
        if evidence is not None and evidence.crossed_mouth:
            return ["mouth_missing"]
        return ["near_pocket_missing"]

    def _review_reason_codes(self, memory: _PocketMemory) -> list[str]:
        evidence = memory.evidence
        if memory.reappear_veto:
            reasons = ["reappeared_near_pocket"]
            if memory.reappear_group is not None and memory.reappear_group != memory.group:
                reasons.append("reappeared_group_changed")
            return reasons
        if memory.resting_mouth_since_ns is not None and not bool(evidence and evidence.crossed_throat):
            return ["mouth_rest_disappear_requires_review"]
        if not self._strong_confirmation_evidence(memory):
            return ["insufficient_pocket_evidence"]
        return ["pocket_review_required"]

    @staticmethod
    def _strong_confirmation_evidence(memory: _PocketMemory) -> bool:
        evidence = memory.evidence
        return bool(evidence and (evidence.entered_interior or evidence.crossed_throat))

    def _occluded_commit_allowed(self, memory: _PocketMemory) -> bool:
        if memory.absent_since_ns is not None:
            return True
        return int(memory.last_lost_frames) >= self._occluded_lost_frames_threshold()

    @staticmethod
    def _has_pocket_interest(memory: _PocketMemory) -> bool:
        return bool(memory.evidence is not None and not memory.resolved)

    def _memory_missing_ms(self, memory: _PocketMemory, now_ns: int) -> int:
        since_ns = memory.nonvisible_since_ns or memory.absent_since_ns
        if since_ns is None:
            return 0
        return self._elapsed_ms(now_ns, since_ns)

    def _new_decision_id(self) -> str:
        decision_id = f"pocket:{self._decision_seq}"
        self._decision_seq += 1
        return decision_id

    def _center_mm(self, track: TrackObservation) -> tuple[float, float]:
        point = track.center_mm if track.center_mm is not None else track.center_px
        return (float(point[0]), float(point[1]))

    @staticmethod
    def _velocity(track: TrackObservation) -> tuple[float, float]:
        velocity = track.velocity_mm_s if track.velocity_mm_s is not None else track.velocity_px_s
        return (float(velocity[0]), float(velocity[1]))

    def _speed(self, track: TrackObservation) -> float:
        vx, vy = self._velocity(track)
        return float(np.hypot(vx, vy))

    def _still_speed(self, track: TrackObservation) -> float:
        if track.velocity_mm_s is not None:
            return float(self.config.still_speed_mm_s)
        return float(self.config.still_speed_px_s)

    def _inward_speed_threshold(self, track: TrackObservation) -> float:
        return self._still_speed(track) * 1.15

    @staticmethod
    def _track_relevant(track: TrackObservation) -> bool:
        if track.visibility == "visible":
            return float(track.quality) > 0.25
        return True

    def _tentative_missing_ms(self) -> int:
        return max(1, int(getattr(self.config, "pocket_tentative_missing_ms", 300) or 300))

    def _commit_ready_missing_ms(self) -> int:
        explicit = getattr(self.config, "pocket_commit_ready_missing_ms", None)
        legacy = getattr(self.config, "pocket_confirm_missing_ms", None)
        selected = explicit if explicit is not None else legacy if legacy is not None else 700
        return max(self._tentative_missing_ms(), int(selected))

    def _final_confirmation_missing_ms(self) -> int:
        reappear = max(1, int(getattr(self.config, "pocket_reappear_window_ms", 800) or 800))
        return max(self._commit_ready_missing_ms(), reappear)

    def _reappear_match_distance_mm(self) -> float:
        return max(
            self.ball_diameter_mm * 2.2,
            float(getattr(self.config, "pocket_mouth_radius_mm", 125.0)) * 0.75,
        )

    @staticmethod
    def _occluded_lost_frames_threshold() -> int:
        return 4

    @staticmethod
    def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return float(np.linalg.norm(np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32)))

    @staticmethod
    def _elapsed_ms(now_ns: int, since_ns: int) -> int:
        return max(0, int(round((int(now_ns) - int(since_ns)) / 1_000_000.0)))


__all__ = ["PerBallPocketFSM", "PocketEvidence"]
