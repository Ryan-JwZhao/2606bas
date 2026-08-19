from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from ..config import StateConfig
from ..schemas import Event, MatchPhase, PocketVisualObservation, PocketVisualObservationFrame, TrackObservation, TracksFrame
from .models import Group, normalize_group
from .pocket_geometry import PocketApproachProbe, PocketGeometryContext, PocketGeometryModel, PocketSample
from .pocket_trajectory import (
    PocketEntryAssessment,
    PocketTrajectoryLimits,
    assess_reported_entry,
    assess_track_handoff,
    projected_entry_has_reversed,
)


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
    projected_entry: bool = False
    entry_source_track_id: Optional[int] = None
    entry_speed_mm_s: float = 0.0
    best_entry_depth_mm: Optional[float] = None
    entry_lateral_mm: float = 0.0
    projected_lateral_mm: float = 0.0
    observer_active: bool = False
    visual_inward: bool = False
    visual_outward: bool = False
    visual_status: str = "unobserved"
    observer_latency_ms: float = 0.0
    associated_track_ids: list[int] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)


@dataclass
class _PocketMemory:
    track_id: int
    group: Group
    last_center_px: tuple[float, float]
    last_center_mm: tuple[float, float]
    last_velocity: tuple[float, float]
    last_seen_frame: int
    last_seen_ts_ns: int
    last_visible_frame: int
    last_visible_ts_ns: int
    last_quality: float
    first_seen_frame: int = 0
    first_seen_ts_ns: int = 0
    last_bbox_aspect: float = 1.0
    last_bbox_horizontal: bool = True
    visible_observation_count: int = 0
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
    detected_emitted: bool = False
    last_zone: Optional[str] = None
    last_visibility: str = "visible"
    last_lost_frames: int = 0
    last_distance_mm: Optional[float] = None
    last_depth_mm: float = 0.0
    last_lateral_mm: float = 0.0
    last_inward_speed_mm_s: float = 0.0
    last_approach_probe: Optional[PocketApproachProbe] = None
    trajectory_anchor_ts_ns: Optional[int] = None
    trajectory_anchor_probe: Optional[PocketApproachProbe] = None
    projected_reversal_since_ns: Optional[int] = None
    reason_codes: list[str] = field(default_factory=list)
    reappear_veto: bool = False
    reappear_track_id: Optional[int] = None
    reappear_group: Optional[Group] = None
    trajectory_history: Deque[tuple[int, PocketApproachProbe]] = field(default_factory=deque)
    visual_lip_since_ns: Optional[int] = None
    track_lip_veto: bool = False


class PerBallPocketFSM:
    """Per-ball pocket evidence with retractable commit-ready decisions."""

    _REAPPEAR_STATES = {"candidate", "tentative", "commit_ready"}
    _TERMINAL_STATES = {"confirmed", "rejected"}

    def __init__(self, config: StateConfig):
        self.config = config
        self.inner_polygon_mm: list[tuple[float, float]] = []
        self.table_edge_polygon_mm: list[tuple[float, float]] = []
        self.ball_center_reachable_polygon_mm: list[tuple[float, float]] = []
        self.pockets_mm: list[tuple[float, float]] = []
        self.pocket_curves_mm: list[list[tuple[float, float]]] = []
        self.ball_diameter_mm = 57.15
        self._memory: Dict[int, _PocketMemory] = {}
        self._deferred_visual_crossings: list[tuple[int, int, PocketVisualObservation, float]] = []
        self._decision_seq = 1
        self._geometry_context_fingerprint: Optional[tuple[object, ...]] = None
        self._geometry_model = PocketGeometryModel.build(
            PocketGeometryContext(),
            config,
            log_diagnostics=False,
        )

    def reset(self) -> None:
        self._memory.clear()
        self._deferred_visual_crossings.clear()
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
            self.pockets_mm = self._canonical_pocket_points([(float(x), float(y)) for x, y in pockets_mm])
        if ball_diameter_mm is not None:
            self.ball_diameter_mm = float(ball_diameter_mm)
        if pocket_curves_mm is not None:
            curves = [
                [(float(point[0]), float(point[1])) for point in list(curve or []) if len(point) >= 2]
                for curve in pocket_curves_mm
            ]
            self.pocket_curves_mm = self._canonical_pocket_curves(curves)
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

    def update(
        self,
        frame: TracksFrame,
        phase: MatchPhase,
        pocket_observations: PocketVisualObservationFrame | None = None,
    ) -> List[Event]:
        events: List[Event] = []
        active = phase in {MatchPhase.SHOT_ACTIVE, MatchPhase.SETTLING, MatchPhase.TURN_RESOLVE}
        seen_ids: set[int] = set()
        visible_ids = {
            int(track.track_id)
            for track in frame.tracks
            if track.visibility == "visible" and float(track.quality) > 0.25
        }

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
                approach = self._approach_probe(track)
                cross_track_vetoed = bool(
                    memory_created_this_frame
                    and self._apply_cross_track_reappear_veto(memory, track, sample, frame, events)
                )
                trajectory: Optional[PocketEntryAssessment] = None
                if active and not cross_track_vetoed and sample.zone is None:
                    trajectory = self._entry_trajectory(
                        memory,
                        track,
                        approach,
                        frame,
                        memory_created_this_frame=memory_created_this_frame,
                        visible_ids=visible_ids,
                    )
                self._handle_visible(
                    memory,
                    track,
                    sample,
                    approach,
                    trajectory,
                    frame,
                    active and not cross_track_vetoed,
                    events,
                )
            elif active:
                self._handle_nonvisible(
                    memory,
                    track,
                    frame,
                    events,
                    visual_mode=pocket_observations is not None,
                )

        if pocket_observations is not None:
            self._prune_deferred_visual_crossings(frame.ts_cam_ns)
            if not active:
                self._remember_deferred_visual_crossings(pocket_observations)
            else:
                if phase in {MatchPhase.SHOT_ACTIVE, MatchPhase.SETTLING}:
                    self._apply_deferred_visual_crossings(frame, events)
                self._apply_visual_observations(frame, pocket_observations, events)

        if active:
            for memory in list(self._memory.values()):
                if memory.track_id in seen_ids or memory.resolved:
                    continue
                self._handle_absent(memory, frame, events)

        self._expire_stale_observed_candidates(frame, events)

        return events

    def _remember_deferred_visual_crossings(self, frame: PocketVisualObservationFrame) -> None:
        for observed in frame.observations:
            if not self._strong_deferred_visual_crossing(observed):
                continue
            track_ids = tuple(sorted(int(value) for value in observed.associated_track_ids))
            key = (int(observed.pocket_index), track_ids)
            self._deferred_visual_crossings = [
                row
                for row in self._deferred_visual_crossings
                if (int(row[2].pocket_index), tuple(sorted(int(value) for value in row[2].associated_track_ids)))
                != key
            ]
            self._deferred_visual_crossings.append(
                (int(frame.frame_id), int(frame.ts_cam_ns), observed, float(frame.latency_ms))
            )

    def _apply_deferred_visual_crossings(self, frame: TracksFrame, events: List[Event]) -> None:
        pending = list(self._deferred_visual_crossings)
        self._deferred_visual_crossings.clear()
        for frame_id, ts_ns, observed, latency_ms in pending:
            proxy = TracksFrame(
                frame_id=int(frame_id),
                ts_cam_ns=int(ts_ns),
                tracks=frame.tracks,
            )
            self._start_visual_candidate(proxy, observed, latency_ms, events)

    def _prune_deferred_visual_crossings(self, now_ns: int) -> None:
        handoff_ns = self._trajectory_limits().handoff_ms * 1_000_000
        self._deferred_visual_crossings = [
            row for row in self._deferred_visual_crossings if int(now_ns) - int(row[1]) <= handoff_ns
        ]

    @staticmethod
    def _strong_deferred_visual_crossing(observed: PocketVisualObservation) -> bool:
        return bool(
            observed.inward_crossing
            and normalize_group(observed.group) is not None
            and observed.associated_track_ids
            and float(observed.motion_score) >= 0.35
            and float(observed.foreground_score) >= 0.20
            and observed.foreground_depth_diameters is not None
            and float(observed.foreground_depth_diameters) >= -0.15
            and {"foreground_motion", "ball_sized_motion"}.issubset(observed.evidence_sources)
        )

    def _expire_stale_observed_candidates(self, frame: TracksFrame, events: List[Event]) -> None:
        timeout_ms = max(1800, int(getattr(self.config, "pocket_visual_confirmation_ms", 1300)) + 400)
        for memory in self._memory.values():
            evidence = memory.evidence
            if (
                evidence is None
                or memory.resolved
                or memory.detected_emitted
                or not evidence.observer_active
                or self._elapsed_ms(frame.ts_cam_ns, evidence.candidate_since_ns) < timeout_ms
            ):
                continue
            self._reject(
                memory,
                frame,
                events,
                reason_codes=["automatic_observation_timeout"],
                candidate_reason="automatic_observation_timeout",
            )

    def _apply_visual_observations(
        self,
        frame: TracksFrame,
        observations_frame: PocketVisualObservationFrame,
        events: List[Event],
    ) -> None:
        by_pocket = {int(item.pocket_index): item for item in observations_frame.observations}
        for memory in self._memory.values():
            evidence = memory.evidence
            if evidence is None or memory.resolved:
                continue
            evidence.observer_active = True
            evidence.observer_latency_ms = float(observations_frame.latency_ms)
            observed = by_pocket.get(int(evidence.pocket_index))
            if observed is not None:
                self._merge_visual_metadata(evidence, observed)

        for observed in observations_frame.observations:
            if observed.inward_crossing:
                self._start_visual_candidate(frame, observed, observations_frame.latency_ms, events)

            matches = self._visual_candidate_matches(observed)
            if observed.outward_crossing:
                for memory in matches:
                    if memory.evidence is not None:
                        memory.evidence.visual_outward = True
                        memory.evidence.visual_status = "outward_crossing"
                    self._reject(
                        memory,
                        frame,
                        events,
                        reason_codes=["visual_outward_crossing"],
                        candidate_reason="visual_outward_crossing",
                    )
                continue

            for memory in matches:
                evidence = memory.evidence
                if evidence is None or memory.resolved:
                    continue
                self._merge_visual_metadata(evidence, observed)
                if observed.lip_occupied:
                    evidence.visual_status = "lip_occupied"
                    if memory.visual_lip_since_ns is None:
                        memory.visual_lip_since_ns = frame.ts_cam_ns
                    if self._elapsed_ms(frame.ts_cam_ns, memory.visual_lip_since_ns) >= self._lip_veto_ms():
                        self._reject(
                            memory,
                            frame,
                            events,
                            reason_codes=["persistent_lip_occupancy"],
                            candidate_reason="lip_occupied_veto",
                        )
                elif observed.clear:
                    evidence.visual_status = "clear"
                    if not memory.track_lip_veto:
                        memory.visual_lip_since_ns = None

        for memory in self._memory.values():
            evidence = memory.evidence
            if (
                evidence is not None
                and not memory.resolved
                and evidence.visual_status == "clear"
            ):
                self._emit_detected_if_ready(
                    memory,
                    frame,
                    events,
                    missing_ms=self._memory_missing_ms(memory, frame.ts_cam_ns),
                )

    def _start_visual_candidate(
        self,
        frame: TracksFrame,
        observed: PocketVisualObservation,
        latency_ms: float,
        events: List[Event],
    ) -> None:
        group = normalize_group(observed.group)
        if group is None or not observed.associated_track_ids:
            return
        memory = self._memory_for_visual_candidate(observed, group, frame.ts_cam_ns)
        if memory is None or memory.resolved:
            return
        if memory.evidence is None:
            has_ball_foreground = bool(
                {"foreground_motion", "ball_sized_motion"}.intersection(observed.evidence_sources)
            )
            credible_approach = memory.last_inward_speed_mm_s >= self._trajectory_limits().min_speed_mm_s
            currently_visible = any(
                int(track.track_id) == int(memory.track_id)
                and track.visibility == "visible"
                and float(track.quality) > 0.25
                for track in frame.tracks
            )
            recently_disappeared = bool(
                not currently_visible
                and self._elapsed_ms(frame.ts_cam_ns, memory.last_visible_ts_ns)
                <= self._trajectory_limits().handoff_ms
            )
            strong_crossing_shape = bool(
                float(observed.motion_score) >= 0.35
                and float(observed.foreground_score) >= 0.20
                and observed.foreground_depth_diameters is not None
                and float(observed.foreground_depth_diameters) >= 0.20
                and {"foreground_motion", "ball_sized_motion"}.issubset(observed.evidence_sources)
            )
            newborn_visible_crossing = bool(
                currently_visible
                and memory.visible_observation_count <= 2
                and self._elapsed_ms(frame.ts_cam_ns, memory.first_seen_ts_ns)
                <= self._trajectory_limits().handoff_ms
                and float(observed.motion_score) >= 0.75
                and float(observed.foreground_score) >= 0.60
            )
            strong_fast_crossing = bool(
                strong_crossing_shape and (recently_disappeared or newborn_visible_crossing)
            )
            if not has_ball_foreground or not (credible_approach or strong_fast_crossing):
                return
        pocket_index = int(observed.pocket_index)
        if memory.evidence is not None and memory.evidence.pocket_index != pocket_index:
            return
        if memory.evidence is None:
            memory.evidence = PocketEvidence(
                decision_id=self._new_decision_id(),
                pocket_index=pocket_index,
                candidate_since_ns=frame.ts_cam_ns,
                last_evidence_ts_ns=frame.ts_cam_ns,
                candidate_reason="pocket_visual_inward",
            )
        evidence = memory.evidence
        evidence.last_evidence_ts_ns = frame.ts_cam_ns
        evidence.crossed_mouth = True
        evidence.visual_inward = True
        evidence.visual_status = "inward_crossing"
        evidence.observer_active = True
        evidence.observer_latency_ms = float(latency_ms)
        if observed.associated_track_ids:
            evidence.entry_source_track_id = int(observed.associated_track_ids[0])
        self._merge_visual_metadata(evidence, observed)
        memory.visual_lip_since_ns = None
        visible_ids = {
            int(track.track_id)
            for track in frame.tracks
            if track.visibility == "visible" and float(track.quality) > 0.25
        }
        if memory.track_id not in visible_ids and memory.nonvisible_since_ns is None:
            memory.nonvisible_since_ns = frame.ts_cam_ns
            memory.absent_since_ns = frame.ts_cam_ns
        if memory.state != "on_table":
            return
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
                    "pocket_index": pocket_index,
                    "decision_id": evidence.decision_id,
                    "candidate_reason": evidence.candidate_reason,
                    "evidence": self._evidence_payload(memory, frame.ts_cam_ns),
                },
                confidence=max(0.72, float(observed.confidence)),
            )
        )

    def _memory_for_visual_candidate(
        self,
        observed: PocketVisualObservation,
        group: Group,
        now_ns: int,
    ) -> Optional[_PocketMemory]:
        associated = {int(value) for value in observed.associated_track_ids}
        exact = [
            memory
            for track_id, memory in self._memory.items()
            if track_id in associated and not memory.resolved and memory.group == group
        ]
        if exact:
            return max(exact, key=lambda item: item.last_visible_ts_ns)
        history_ns = self._entry_history_ms() * 1_000_000
        recent = [
            memory
            for memory in self._memory.values()
            if not memory.resolved
            and memory.group == group
            and int(now_ns) - int(memory.last_visible_ts_ns) <= history_ns
            and (
                memory.track_id in associated
                or (memory.evidence is None and memory.last_visibility != "visible")
            )
        ]
        return max(recent, key=lambda item: item.last_visible_ts_ns) if recent else None

    def _visual_candidate_matches(self, observed: PocketVisualObservation) -> list[_PocketMemory]:
        associated = {int(value) for value in observed.associated_track_ids}
        group = normalize_group(observed.group)
        matches: list[_PocketMemory] = []
        for memory in self._memory.values():
            evidence = memory.evidence
            if memory.resolved or evidence is None or evidence.pocket_index != int(observed.pocket_index):
                continue
            evidence_ids = set(evidence.associated_track_ids)
            if associated and memory.track_id not in associated and not (associated & evidence_ids):
                continue
            if group is not None and memory.group != group:
                continue
            matches.append(memory)
        return matches

    @staticmethod
    def _merge_visual_metadata(evidence: PocketEvidence, observed: PocketVisualObservation) -> None:
        evidence.associated_track_ids = sorted(
            set(evidence.associated_track_ids) | {int(value) for value in observed.associated_track_ids}
        )
        sources = list(evidence.evidence_sources)
        if observed.inward_crossing and "pocket_visual_inward" not in sources:
            sources.append("pocket_visual_inward")
        for source in observed.evidence_sources:
            normalized = str(source).strip()
            if normalized and normalized not in sources:
                sources.append(normalized)
        evidence.evidence_sources = sources

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
                    "projected_entry": bool(evidence and evidence.projected_entry),
                    "entry_source_track_id": evidence.entry_source_track_id if evidence is not None else None,
                    "entry_speed_mm_s": float(evidence.entry_speed_mm_s) if evidence is not None else 0.0,
                    "detected_emitted": bool(memory.detected_emitted),
                    "last_zone": memory.last_zone,
                    "zone": memory.last_zone,
                    "inward_speed_mm_s": float(memory.last_inward_speed_mm_s),
                    "last_bbox_aspect": float(memory.last_bbox_aspect),
                    "visible_observation_count": int(memory.visible_observation_count),
                    "last_approach": (
                        {
                            "pocket_index": memory.last_approach_probe.pocket_index,
                            "depth_mm": float(memory.last_approach_probe.depth_mm),
                            "lateral_mm": float(memory.last_approach_probe.signed_lateral_mm),
                            "mouth_half_width_mm": float(memory.last_approach_probe.mouth_half_width_mm),
                            "geometry_valid": bool(memory.last_approach_probe.geometry_valid),
                        }
                        if memory.last_approach_probe is not None
                        else None
                    ),
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

    def reject_pending(
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
            self._reject(
                memory,
                frame,
                events,
                reason_codes=[*reasons, "automatic_ambiguous_reject"],
                candidate_reason="automatic_ambiguous_reject",
            )
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
            confirmation_ms = self._confirmation_elapsed_ms(memory, now_ns)
            if memory.state == "commit_ready" and confirmation_ms >= self._final_confirmation_missing_ms(memory):
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
                    "projected_entry": bool(evidence.projected_entry),
                    "entry_source_track_id": evidence.entry_source_track_id,
                    "entry_speed_mm_s": float(evidence.entry_speed_mm_s),
                    "candidate_reason": evidence.candidate_reason,
                    "finalizable_at_missing_ms": self._final_confirmation_missing_ms(memory),
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

    def mark_rejected(self, decision_ids: list[str]) -> None:
        wanted = {str(value) for value in decision_ids if str(value).strip()}
        for memory in self._memory.values():
            evidence = memory.evidence
            if evidence is None or evidence.decision_id not in wanted or memory.state == "confirmed":
                continue
            memory.state = "rejected"
            memory.decision = "rejected"
            memory.resolved = True
            memory.reason_codes = ["automatic_turn_resolution_reject"]

    def _new_memory(self, track: TrackObservation, group: Group, frame: TracksFrame) -> _PocketMemory:
        return _PocketMemory(
            track_id=track.track_id,
            group=group,
            last_center_px=track.center_px,
            last_center_mm=self._center_mm(track),
            last_velocity=self._velocity(track),
            last_seen_frame=frame.frame_id,
            last_seen_ts_ns=frame.ts_cam_ns,
            last_visible_frame=frame.frame_id,
            last_visible_ts_ns=frame.ts_cam_ns,
            last_quality=float(track.quality),
            first_seen_frame=frame.frame_id,
            first_seen_ts_ns=frame.ts_cam_ns,
            last_bbox_aspect=self._bbox_aspect(track),
            last_bbox_horizontal=self._bbox_horizontal(track),
        )

    def _handle_visible(
        self,
        memory: _PocketMemory,
        track: TrackObservation,
        sample: PocketSample,
        approach: PocketApproachProbe,
        trajectory: Optional[PocketEntryAssessment],
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
                self._reject(
                    memory,
                    frame,
                    events,
                    reason_codes=[*reason_codes, "automatic_ambiguous_reject"],
                    candidate_reason="reappeared_group_changed",
                )
            else:
                self._reject(memory, frame, events, reason_codes=reason_codes, candidate_reason="reappeared_visible")
            reappeared_now = True

        if active and not reappeared_now:
            self._update_candidate_evidence(memory, track, sample, approach, trajectory, frame, events)

        if active and memory.evidence is not None and not memory.resolved:
            if memory.evidence.projected_entry and memory.state in {"candidate", "tentative", "commit_ready"}:
                limits = self._trajectory_limits()
                reversed_entry = projected_entry_has_reversed(
                    approach,
                    pocket_index=memory.evidence.pocket_index,
                    best_depth_mm=float(memory.evidence.best_entry_depth_mm or approach.depth_mm),
                    limits=limits,
                )
                if reversed_entry and memory.projected_reversal_since_ns is None:
                    memory.projected_reversal_since_ns = frame.ts_cam_ns
                immediate_outward = float(approach.pocketward_speed_mm_s) <= -float(limits.min_speed_mm_s)
                if reversed_entry and (
                    immediate_outward
                    or self._elapsed_ms(frame.ts_cam_ns, memory.projected_reversal_since_ns or frame.ts_cam_ns) >= 90
                ):
                    self._reject(
                        memory,
                        frame,
                        events,
                        reason_codes=["projected_entry_reversed"],
                        candidate_reason="projected_entry_reversed",
                    )
                elif not reversed_entry:
                    memory.projected_reversal_since_ns = None
                    memory.evidence.best_entry_depth_mm = max(
                        float(memory.evidence.best_entry_depth_mm or approach.depth_mm),
                        float(approach.depth_mm),
                    )
            elif sample.zone is None and memory.state in {"candidate", "tentative", "commit_ready"}:
                self._reject(memory, frame, events, reason_codes=["back_to_table"], candidate_reason="back_to_table")
            elif sample.zone is None and not memory.tentative_emitted:
                memory.evidence = None
                memory.projected_reversal_since_ns = None
                memory.resting_mouth_since_ns = None

        if active and sample.zone == "mouth" and self._speed(track) <= self._still_speed(track) * 1.25:
            if memory.resting_mouth_since_ns is None:
                memory.resting_mouth_since_ns = frame.ts_cam_ns
        elif sample.zone not in {"mouth", "throat", "interior"}:
            memory.resting_mouth_since_ns = None

        if memory.evidence is None:
            memory.group = normalize_group(track.group) or memory.group
        current_center_mm = self._center_mm(track)
        if (
            memory.trajectory_anchor_probe is None
            or self._distance(memory.last_center_mm, current_center_mm) >= max(1.0, self.ball_diameter_mm * 0.03)
        ):
            memory.trajectory_anchor_ts_ns = frame.ts_cam_ns
            memory.trajectory_anchor_probe = approach
        memory.last_center_px = track.center_px
        memory.last_center_mm = current_center_mm
        memory.last_velocity = self._velocity(track)
        memory.last_seen_frame = frame.frame_id
        memory.last_seen_ts_ns = frame.ts_cam_ns
        memory.last_visible_frame = frame.frame_id
        memory.last_visible_ts_ns = frame.ts_cam_ns
        memory.last_quality = float(track.quality)
        memory.last_bbox_aspect = self._bbox_aspect(track)
        memory.last_bbox_horizontal = self._bbox_horizontal(track)
        memory.visible_observation_count += 1
        memory.last_zone = sample.zone
        memory.last_visibility = "visible"
        memory.last_lost_frames = 0
        memory.last_distance_mm = sample.distance_mm if sample.zone is not None else approach.distance_mm
        memory.last_depth_mm = float(sample.depth_mm if sample.zone is not None else approach.depth_mm)
        memory.last_lateral_mm = float(
            sample.lateral_mm if sample.zone is not None else abs(approach.signed_lateral_mm)
        )
        memory.last_inward_speed_mm_s = float(
            sample.pocketward_speed_mm_s if sample.zone is not None else approach.pocketward_speed_mm_s
        )
        memory.last_approach_probe = approach
        self._append_trajectory_history(memory, frame.ts_cam_ns, approach)
        memory.nonvisible_since_ns = None
        memory.absent_since_ns = None

    def _handle_nonvisible(
        self,
        memory: _PocketMemory,
        track: TrackObservation,
        frame: TracksFrame,
        events: List[Event],
        *,
        visual_mode: bool,
    ) -> None:
        memory.last_seen_frame = frame.frame_id
        memory.last_seen_ts_ns = frame.ts_cam_ns
        memory.last_visibility = str(track.visibility or "occluded")
        memory.last_lost_frames = int(getattr(track, "lost_frames", 0))
        if not self._has_pocket_interest(memory):
            return
        if memory.nonvisible_since_ns is None:
            memory.nonvisible_since_ns = frame.ts_cam_ns
        evidence = memory.evidence
        if (
            evidence is not None
            and not memory.detected_emitted
            and memory.last_zone in {"mouth", "throat", "interior"}
            and self._elapsed_ms(memory.last_visible_ts_ns, evidence.candidate_since_ns) >= 900
        ):
            memory.track_lip_veto = True
            memory.visual_lip_since_ns = evidence.candidate_since_ns
            evidence.visual_status = "lip_occupied"
            if "prolonged_visible_at_lip" not in evidence.evidence_sources:
                evidence.evidence_sources.append("prolonged_visible_at_lip")
        allow_occluded = self._occluded_commit_allowed(memory)
        if visual_mode:
            allow_occluded = bool(memory.evidence and memory.evidence.visual_inward)
        self._advance_missing(memory, frame, events, allow_occluded_commit=allow_occluded)

    def _handle_absent(self, memory: _PocketMemory, frame: TracksFrame, events: List[Event]) -> None:
        if memory.evidence is None:
            self._start_blurred_disappearance_candidate(memory, frame, events)
        if not self._has_pocket_interest(memory):
            return
        if memory.absent_since_ns is None:
            memory.absent_since_ns = frame.ts_cam_ns
        if memory.nonvisible_since_ns is None:
            memory.nonvisible_since_ns = memory.absent_since_ns
        self._advance_missing(memory, frame, events, allow_occluded_commit=True)

    def _start_blurred_disappearance_candidate(
        self,
        memory: _PocketMemory,
        frame: TracksFrame,
        events: List[Event],
    ) -> None:
        probe = memory.last_approach_probe
        limits = self._trajectory_limits()
        missing_ms = self._elapsed_ms(frame.ts_cam_ns, memory.last_visible_ts_ns)
        if (
            probe is None
            or probe.pocket_index is None
            or not probe.geometry_valid
            or memory.visible_observation_count > 3
            or memory.last_bbox_aspect < 1.45
            or memory.last_bbox_aspect > self._blur_max_aspect_ratio()
            or memory.last_quality > 0.74
            or missing_ms > self._entry_history_ms() + 250
            or probe.depth_mm < -limits.history_depth_mm
        ):
            return
        pocket_index = int(probe.pocket_index)
        pocket_point = self.pockets_mm[pocket_index] if pocket_index < len(self.pockets_mm) else None
        if pocket_point is None:
            return
        direction = np.asarray(pocket_point, dtype=np.float32) - np.asarray(memory.last_center_mm, dtype=np.float32)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            return
        axis_alignment = abs(float(direction[0] if memory.last_bbox_horizontal else direction[1])) / direction_norm
        if axis_alignment < 0.80:
            return
        corridor_slope = 1.20 if pocket_index in {0, 2, 3, 5} else 0.65
        capture_half_width = float(probe.mouth_half_width_mm) + max(0.0, -float(probe.depth_mm)) * corridor_slope
        if abs(float(probe.signed_lateral_mm)) > capture_half_width:
            return
        memory.evidence = PocketEvidence(
            decision_id=self._new_decision_id(),
            pocket_index=pocket_index,
            candidate_since_ns=frame.ts_cam_ns,
            last_evidence_ts_ns=frame.ts_cam_ns,
            candidate_reason="blurred_single_frame_disappearance",
            projected_entry=True,
            entry_source_track_id=memory.track_id,
            entry_speed_mm_s=limits.min_speed_mm_s,
            best_entry_depth_mm=float(probe.depth_mm),
            entry_lateral_mm=float(probe.signed_lateral_mm),
            projected_lateral_mm=float(probe.signed_lateral_mm),
            evidence_sources=["blurred_single_frame", "full_track_disappearance"],
        )
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
                    "pocket_index": pocket_index,
                    "decision_id": memory.evidence.decision_id,
                    "candidate_reason": memory.evidence.candidate_reason,
                    "evidence": self._evidence_payload(memory, frame.ts_cam_ns),
                },
                confidence=0.72,
            )
        )

    def _update_candidate_evidence(
        self,
        memory: _PocketMemory,
        track: TrackObservation,
        sample: PocketSample,
        approach: PocketApproachProbe,
        trajectory: Optional[PocketEntryAssessment],
        frame: TracksFrame,
        events: List[Event],
    ) -> None:
        has_zone_evidence = sample.zone in {"mouth", "throat", "interior"} and sample.pocket_index is not None
        if not has_zone_evidence and trajectory is None:
            return
        pocket_index = int(sample.pocket_index) if has_zone_evidence else int(trajectory.pocket_index)
        reason = self._candidate_reason(track, sample) if has_zone_evidence else trajectory.reason
        if memory.evidence is None and reason is None:
            return
        if memory.evidence is not None and memory.evidence.pocket_index != pocket_index:
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
                pocket_index=pocket_index,
                candidate_since_ns=frame.ts_cam_ns,
                last_evidence_ts_ns=frame.ts_cam_ns,
            )
        evidence = memory.evidence
        evidence.last_evidence_ts_ns = frame.ts_cam_ns
        if has_zone_evidence:
            evidence.crossed_mouth = True
            if sample.zone in {"throat", "interior"}:
                evidence.crossed_throat = True
            if sample.zone == "interior":
                evidence.entered_interior = True
        if trajectory is not None:
            evidence.projected_entry = True
            evidence.entry_source_track_id = int(trajectory.source_track_id)
            evidence.entry_speed_mm_s = max(float(evidence.entry_speed_mm_s), float(trajectory.speed_mm_s))
            evidence.best_entry_depth_mm = max(
                float(evidence.best_entry_depth_mm) if evidence.best_entry_depth_mm is not None else float("-inf"),
                float(trajectory.depth_mm),
            )
            evidence.entry_lateral_mm = float(trajectory.lateral_mm)
            evidence.projected_lateral_mm = float(trajectory.projected_lateral_mm)

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
                        "distance_to_pocket_mm": (
                            sample.distance_mm if has_zone_evidence else approach.distance_mm
                        ),
                        "evidence": self._evidence_payload(memory, frame.ts_cam_ns),
                    },
                    confidence=(
                        0.72
                        if sample.zone in {"throat", "interior"}
                        else 0.68
                        if trajectory is not None
                        else 0.58
                    ),
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
        if memory.resolved:
            return
        if memory.state == "commit_ready":
            self._emit_detected_if_ready(memory, frame, events, missing_ms=missing_ms)
            return
        if missing_ms < self._commit_ready_missing_ms():
            return
        if self._strong_confirmation_evidence(memory) and not memory.reappear_veto:
            if allow_occluded_commit:
                self._commit_ready(memory, frame, events)
                self._emit_detected_if_ready(memory, frame, events, missing_ms=missing_ms)
            return
        if memory.evidence is not None and memory.evidence.observer_active:
            self._reject(
                memory,
                frame,
                events,
                reason_codes=[*self._ambiguity_reason_codes(memory), "automatic_insufficient_evidence"],
                candidate_reason="automatic_insufficient_evidence",
            )
        else:
            self._reject(
                memory,
                frame,
                events,
                reason_codes=[*self._ambiguity_reason_codes(memory), "automatic_ambiguous_reject"],
                candidate_reason="automatic_ambiguous_reject",
            )

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

    def _emit_detected_if_ready(
        self,
        memory: _PocketMemory,
        frame: TracksFrame,
        events: List[Event],
        *,
        missing_ms: int,
    ) -> None:
        if (
            memory.detected_emitted
            or memory.evidence is None
            or memory.nonvisible_since_ns is None
            or self._confirmation_elapsed_ms(memory, frame.ts_cam_ns) < self._final_confirmation_missing_ms(memory)
            or not self._strong_confirmation_evidence(memory)
            or memory.reappear_veto
            or memory.visual_lip_since_ns is not None
            or memory.evidence.visual_outward
        ):
            return
        memory.detected_emitted = True
        events.append(
            Event(
                name="POCKET_DETECTED",
                ts_cam_ns=frame.ts_cam_ns,
                frame_id=frame.frame_id,
                payload=self._decision_payload(
                    memory,
                    frame.ts_cam_ns,
                    decision="detected",
                    reason_codes=["automatic_confirmation_window_elapsed"],
                ),
                confidence=0.94,
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
                self._reject(
                    candidate,
                    frame,
                    events,
                    reason_codes=[
                        "reappeared_near_pocket",
                        "track_relinked",
                        "reappeared_group_changed",
                        "automatic_ambiguous_reject",
                    ],
                    candidate_reason="reappeared_group_changed",
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
            "reason_codes": list(reason_codes),
            "candidate_reason": evidence.candidate_reason,
            "last_center_mm": list(memory.last_center_mm),
            "last_center_px": list(memory.last_center_px),
            "missing_ms": self._memory_missing_ms(memory, now_ns),
            "decision_latency_ms": self._elapsed_ms(now_ns, evidence.candidate_since_ns),
            "confirmation": "two_stage_commit",
            "evidence_sources": list(evidence.evidence_sources),
            "associated_track_ids": list(evidence.associated_track_ids),
            "visual_status": evidence.visual_status,
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
            "projected_entry": bool(evidence and evidence.projected_entry),
            "entry_source_track_id": evidence.entry_source_track_id if evidence is not None else None,
            "entry_speed_mm_s": float(evidence.entry_speed_mm_s) if evidence is not None else 0.0,
            "best_entry_depth_mm": (
                float(evidence.best_entry_depth_mm)
                if evidence is not None and evidence.best_entry_depth_mm is not None
                else None
            ),
            "entry_lateral_mm": float(evidence.entry_lateral_mm) if evidence is not None else None,
            "projected_lateral_mm": float(evidence.projected_lateral_mm) if evidence is not None else None,
            "observer_active": bool(evidence and evidence.observer_active),
            "visual_inward": bool(evidence and evidence.visual_inward),
            "visual_outward": bool(evidence and evidence.visual_outward),
            "visual_status": evidence.visual_status if evidence is not None else "unobserved",
            "observer_latency_ms": float(evidence.observer_latency_ms) if evidence is not None else 0.0,
            "associated_track_ids": list(evidence.associated_track_ids) if evidence is not None else [],
            "evidence_sources": list(evidence.evidence_sources) if evidence is not None else [],
            "pocket_index": evidence.pocket_index if evidence is not None else None,
            "distance_mm": memory.last_distance_mm,
            "depth_mm": float(memory.last_depth_mm),
            "lateral_mm": float(memory.last_lateral_mm),
            "last_visibility": memory.last_visibility,
            "lost_frames": int(memory.last_lost_frames),
            "projected_reversal_ms": (
                self._elapsed_ms(now_ns, memory.projected_reversal_since_ns)
                if memory.projected_reversal_since_ns is not None
                else 0
            ),
            "decision": memory.decision,
            "detected_emitted": bool(memory.detected_emitted),
        }

    def _sample(self, track: TrackObservation) -> PocketSample:
        return self._geometry_model.sample(self._center_mm(track), self._velocity(track))

    def _approach_probe(self, track: TrackObservation) -> PocketApproachProbe:
        return self._geometry_model.approach_probe(self._center_mm(track), self._velocity(track))

    def _entry_trajectory(
        self,
        memory: _PocketMemory,
        track: TrackObservation,
        approach: PocketApproachProbe,
        frame: TracksFrame,
        *,
        memory_created_this_frame: bool,
        visible_ids: set[int],
    ) -> Optional[PocketEntryAssessment]:
        limits = self._trajectory_limits()
        if not memory_created_this_frame:
            previous = memory.last_approach_probe
            if (
                previous is not None
                and previous.pocket_index == approach.pocket_index
                and float(approach.depth_mm) >= float(previous.depth_mm) + max(2.0, self.ball_diameter_mm * 0.04)
            ):
                reported = assess_reported_entry(approach, track_id=memory.track_id, limits=limits)
                if reported is not None:
                    return reported
            historical = self._assess_same_track_history(memory, approach, frame.ts_cam_ns, limits)
            return historical

        matches: list[tuple[float, float, _PocketMemory, PocketEntryAssessment]] = []
        for source in self._memory.values():
            if (
                source.track_id == memory.track_id
                or source.resolved
                or source.group != memory.group
                or source.track_id in visible_ids
                or (source.trajectory_anchor_probe is None and source.last_approach_probe is None)
            ):
                continue
            source_probe = source.trajectory_anchor_probe or source.last_approach_probe
            source_ts_ns = source.trajectory_anchor_ts_ns or source.last_visible_ts_ns
            elapsed_ms = self._elapsed_ms(frame.ts_cam_ns, source_ts_ns)
            assessment = assess_track_handoff(
                source_probe,
                approach,
                source_track_id=source.track_id,
                elapsed_ms=float(elapsed_ms),
                limits=limits,
            )
            if assessment is None:
                continue
            matches.append(
                (
                    float(elapsed_ms),
                    abs(float(assessment.projected_lateral_mm)),
                    source,
                    assessment,
                )
            )
        if not matches:
            return None
        matches.sort(key=lambda item: (item[0], item[1], item[2].track_id))
        source = matches[0][2]
        assessment = matches[0][3]
        self._adopt_track_handoff(memory, source)
        return assessment

    def _assess_same_track_history(
        self,
        memory: _PocketMemory,
        approach: PocketApproachProbe,
        now_ns: int,
        limits: PocketTrajectoryLimits,
    ) -> Optional[PocketEntryAssessment]:
        history_limits = PocketTrajectoryLimits(
            candidate_depth_mm=limits.candidate_depth_mm,
            history_depth_mm=limits.history_depth_mm,
            handoff_ms=self._entry_history_ms(),
            min_speed_mm_s=limits.min_speed_mm_s,
            max_speed_mm_s=limits.max_speed_mm_s,
            ball_diameter_mm=limits.ball_diameter_mm,
        )
        for ts_ns, previous in reversed(memory.trajectory_history):
            elapsed_ms = self._elapsed_ms(now_ns, ts_ns)
            if elapsed_ms <= 0 or elapsed_ms > self._entry_history_ms():
                continue
            assessment = assess_track_handoff(
                previous,
                approach,
                source_track_id=memory.track_id,
                elapsed_ms=float(elapsed_ms),
                limits=history_limits,
            )
            if assessment is None:
                continue
            return PocketEntryAssessment(
                pocket_index=assessment.pocket_index,
                reason="projected_entry_pre_shot_history",
                source_track_id=assessment.source_track_id,
                speed_mm_s=assessment.speed_mm_s,
                depth_mm=assessment.depth_mm,
                lateral_mm=assessment.lateral_mm,
                projected_lateral_mm=assessment.projected_lateral_mm,
            )
        return None

    def _append_trajectory_history(
        self,
        memory: _PocketMemory,
        ts_ns: int,
        approach: PocketApproachProbe,
    ) -> None:
        memory.trajectory_history.append((int(ts_ns), approach))
        cutoff = int(ts_ns) - self._entry_history_ms() * 1_000_000
        while memory.trajectory_history and memory.trajectory_history[0][0] < cutoff:
            memory.trajectory_history.popleft()

    @staticmethod
    def _adopt_track_handoff(memory: _PocketMemory, source: _PocketMemory) -> None:
        if source.evidence is not None:
            memory.evidence = source.evidence
            memory.state = source.state
            memory.decision = source.decision
            memory.tentative_since_ns = source.tentative_since_ns
            memory.commit_ready_since_ns = source.commit_ready_since_ns
            memory.nonvisible_since_ns = source.nonvisible_since_ns
            memory.absent_since_ns = source.absent_since_ns
            memory.tentative_emitted = source.tentative_emitted
            memory.commit_ready_emitted = source.commit_ready_emitted
            memory.detected_emitted = source.detected_emitted
            memory.projected_reversal_since_ns = source.projected_reversal_since_ns
            memory.reason_codes = list(source.reason_codes)
        source.evidence = None
        source.state = "handed_off"
        source.decision = "none"
        source.resolved = True
        source.reason_codes = ["track_handoff"]

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
        if evidence is not None and evidence.projected_entry:
            return ["projected_entry_missing"]
        if evidence is not None and evidence.crossed_mouth:
            return ["mouth_missing"]
        return ["near_pocket_missing"]

    def _ambiguity_reason_codes(self, memory: _PocketMemory) -> list[str]:
        evidence = memory.evidence
        if memory.reappear_veto:
            reasons = ["reappeared_near_pocket"]
            if memory.reappear_group is not None and memory.reappear_group != memory.group:
                reasons.append("reappeared_group_changed")
            return reasons
        if memory.resting_mouth_since_ns is not None and not bool(evidence and evidence.crossed_throat):
            return ["mouth_rest_disappear_ambiguous"]
        if not self._strong_confirmation_evidence(memory):
            return ["insufficient_pocket_evidence"]
        return ["pocket_evidence_ambiguous"]

    @staticmethod
    def _strong_confirmation_evidence(memory: _PocketMemory) -> bool:
        evidence = memory.evidence
        return bool(
            evidence
            and not evidence.visual_outward
            and (evidence.entered_interior or evidence.crossed_throat or evidence.visual_inward)
        )

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

    def _confirmation_elapsed_ms(self, memory: _PocketMemory, now_ns: int) -> int:
        evidence = memory.evidence
        if evidence is not None and evidence.observer_active:
            return self._elapsed_ms(now_ns, evidence.candidate_since_ns)
        return self._memory_missing_ms(memory, now_ns)

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

    @staticmethod
    def _bbox_aspect(track: TrackObservation) -> float:
        x1, y1, x2, y2 = track.bbox
        width = max(1.0, float(x2) - float(x1))
        height = max(1.0, float(y2) - float(y1))
        return max(width, height) / min(width, height)

    @staticmethod
    def _bbox_horizontal(track: TrackObservation) -> bool:
        x1, y1, x2, y2 = track.bbox
        return float(x2) - float(x1) >= float(y2) - float(y1)

    def _tentative_missing_ms(self) -> int:
        return max(1, int(getattr(self.config, "pocket_tentative_missing_ms", 300) or 300))

    def _commit_ready_missing_ms(self) -> int:
        explicit = getattr(self.config, "pocket_commit_ready_missing_ms", None)
        legacy = getattr(self.config, "pocket_confirm_missing_ms", None)
        selected = explicit if explicit is not None else legacy if legacy is not None else 700
        return max(self._tentative_missing_ms(), int(selected))

    def _final_confirmation_missing_ms(self, memory: _PocketMemory | None = None) -> int:
        if memory is not None and memory.evidence is not None and memory.evidence.observer_active:
            visual = max(1, int(getattr(self.config, "pocket_visual_confirmation_ms", 1300) or 1300))
            return max(self._commit_ready_missing_ms(), visual)
        reappear = max(1, int(getattr(self.config, "pocket_reappear_window_ms", 800) or 800))
        return max(self._commit_ready_missing_ms(), reappear)

    def _lip_veto_ms(self) -> int:
        return max(1, int(getattr(self.config, "pocket_lip_veto_ms", 1100) or 1100))

    def _entry_history_ms(self) -> int:
        return max(100, int(getattr(self.config, "pocket_entry_history_ms", 1500) or 1500))

    def _blur_max_aspect_ratio(self) -> float:
        return max(1.45, float(getattr(self.config, "pocket_blur_max_aspect_ratio", 2.8) or 2.8))

    def _reappear_match_distance_mm(self) -> float:
        return max(
            self.ball_diameter_mm * 2.2,
            float(getattr(self.config, "pocket_mouth_radius_mm", 125.0)) * 0.75,
        )

    def _trajectory_limits(self) -> PocketTrajectoryLimits:
        candidate_config = getattr(self.config, "pocket_entry_candidate_depth_mm", None)
        history_config = getattr(self.config, "pocket_entry_history_depth_mm", None)
        candidate_depth = (
            float(candidate_config)
            if candidate_config is not None
            else max(125.0, self.ball_diameter_mm * 1.65, float(getattr(self.config, "pocket_funnel_radius_mm", 95.0)))
        )
        history_depth = (
            float(history_config)
            if history_config is not None
            else max(candidate_depth, self.ball_diameter_mm * 5.25)
        )
        min_speed = max(
            1.0,
            float(getattr(self.config, "pocket_entry_min_speed_mm_s", 100.0) or 100.0),
        )
        max_speed = max(
            min_speed,
            float(getattr(self.config, "pocket_entry_max_speed_mm_s", 4000.0) or 4000.0),
        )
        return PocketTrajectoryLimits(
            candidate_depth_mm=max(self.ball_diameter_mm, candidate_depth),
            history_depth_mm=max(candidate_depth, history_depth),
            handoff_ms=max(1, int(getattr(self.config, "pocket_entry_handoff_ms", 450) or 450)),
            min_speed_mm_s=min_speed,
            max_speed_mm_s=max_speed,
            ball_diameter_mm=max(1.0, float(self.ball_diameter_mm)),
        )

    @staticmethod
    def _occluded_lost_frames_threshold() -> int:
        return 4

    @staticmethod
    def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return float(np.linalg.norm(np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32)))

    @staticmethod
    def _canonical_pocket_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) != 6:
            return points
        ranked = sorted(points, key=lambda point: point[1])
        top = sorted(ranked[:3], key=lambda point: point[0])
        bottom = sorted(ranked[3:], key=lambda point: point[0], reverse=True)
        return [*top, *bottom]

    @staticmethod
    def _canonical_pocket_curves(
        curves: list[list[tuple[float, float]]],
    ) -> list[list[tuple[float, float]]]:
        if len(curves) != 6 or any(not curve for curve in curves):
            return curves
        ranked = sorted(curves, key=lambda curve: float(np.mean([point[1] for point in curve])))
        top = sorted(ranked[:3], key=lambda curve: float(np.mean([point[0] for point in curve])))
        bottom = sorted(ranked[3:], key=lambda curve: float(np.mean([point[0] for point in curve])), reverse=True)
        return [*top, *bottom]

    @staticmethod
    def _elapsed_ms(now_ns: int, since_ns: int) -> int:
        return max(0, int(round((int(now_ns) - int(since_ns)) / 1_000_000.0)))


__all__ = ["PerBallPocketFSM", "PocketEvidence"]
