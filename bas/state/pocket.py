from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..config import StateConfig
from ..schemas import Event, MatchPhase, TrackObservation, TracksFrame
from .models import Group, normalize_group


@dataclass(frozen=True)
class _PocketGeometry:
    index: int
    center_mm: tuple[float, float]
    mouth_a_mm: tuple[float, float]
    mouth_b_mm: tuple[float, float]
    mouth_mid_mm: tuple[float, float]
    tangent_unit: tuple[float, float]
    inward_normal: tuple[float, float]
    mouth_half_width_mm: float
    throat_half_width_mm: float
    throat_depth_mm: float
    interior_depth_mm: float


@dataclass(frozen=True)
class _PocketSample:
    zone: Optional[str]
    pocket_index: Optional[int]
    distance_mm: Optional[float]
    depth_mm: float
    lateral_mm: float
    inward_speed_mm_s: float
    inside_playable: bool


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
    candidate_since_ns: Optional[int] = None
    tentative_since_ns: Optional[int] = None
    nonvisible_since_ns: Optional[int] = None
    absent_since_ns: Optional[int] = None
    resting_mouth_since_ns: Optional[int] = None
    pocket_index: Optional[int] = None
    crossed_mouth: bool = False
    crossed_throat: bool = False
    entered_interior: bool = False
    tentative_emitted: bool = False
    last_zone: Optional[str] = None
    last_visibility: str = "visible"
    last_lost_frames: int = 0
    last_distance_mm: Optional[float] = None
    last_depth_mm: float = 0.0
    last_lateral_mm: float = 0.0
    last_inward_speed_mm_s: float = 0.0
    candidate_reason: Optional[str] = None
    decision_id: Optional[str] = None
    reason_codes: list[str] = field(default_factory=list)
    reappear_veto: bool = False
    reappear_track_id: Optional[int] = None
    reappear_group: Optional[Group] = None


class PerBallPocketFSM:
    """Two-stage pocket confirmation with evidence accumulation and vetoes."""

    def __init__(self, config: StateConfig):
        self.config = config
        self.inner_polygon_mm: list[tuple[float, float]] = []
        self.pockets_mm: list[tuple[float, float]] = []
        self.pocket_curves_mm: list[list[tuple[float, float]]] = []
        self.ball_diameter_mm = 57.15
        self._memory: Dict[int, _PocketMemory] = {}
        self._decision_seq = 1

    def reset(self) -> None:
        self._memory.clear()
        self._decision_seq = 1

    def set_table_context(
        self,
        *,
        inner_polygon_mm: Optional[list[tuple[float, float]]] = None,
        pockets_mm: Optional[list[tuple[float, float]]] = None,
        ball_diameter_mm: Optional[float] = None,
        pocket_curves_mm: Optional[list[list[tuple[float, float]]]] = None,
    ) -> None:
        if inner_polygon_mm is not None:
            self.inner_polygon_mm = [(float(x), float(y)) for x, y in inner_polygon_mm]
        if pockets_mm is not None:
            self.pockets_mm = [(float(x), float(y)) for x, y in pockets_mm]
        if ball_diameter_mm is not None:
            self.ball_diameter_mm = float(ball_diameter_mm)
        if pocket_curves_mm is not None:
            self.pocket_curves_mm = [
                [(float(point[0]), float(point[1])) for point in list(curve or []) if len(point) >= 2]
                for curve in pocket_curves_mm
            ]

    def update(self, frame: TracksFrame, phase: MatchPhase) -> List[Event]:
        events: List[Event] = []
        active = phase in {MatchPhase.SHOT_ACTIVE, MatchPhase.SETTLING, MatchPhase.TURN_RESOLVE}
        seen_ids: set[int] = set()

        for track in frame.tracks:
            group = normalize_group(track.group)
            if group is None or not self._track_relevant(track):
                continue
            memory = self._memory.get(track.track_id)
            if memory is None:
                if track.visibility != "visible" or float(track.quality) <= 0.25:
                    continue
                memory = self._new_memory(track, group, frame)
                self._memory[track.track_id] = memory
            elif memory.resolved and track.visibility == "visible":
                memory = self._new_memory(track, group, frame)
                self._memory[track.track_id] = memory

            seen_ids.add(track.track_id)
            if track.visibility == "visible" and float(track.quality) > 0.25:
                sample = self._sample(track)
                cross_track_vetoed = self._apply_cross_track_reappear_veto(memory, track, sample, frame, events)
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
            rows.append(
                {
                    "track_id": int(memory.track_id),
                    "group": memory.group,
                    "state": memory.state,
                    "decision": memory.decision,
                    "decision_id": memory.decision_id,
                    "pocket_index": memory.pocket_index,
                    "crossed_mouth": bool(memory.crossed_mouth),
                    "crossed_throat": bool(memory.crossed_throat),
                    "entered_interior": bool(memory.entered_interior),
                    "last_zone": memory.last_zone,
                    "zone": memory.last_zone,
                    "inward_speed_mm_s": float(memory.last_inward_speed_mm_s),
                    "candidate_reason": memory.candidate_reason,
                    "missing_ms": self._memory_missing_ms(memory, memory.last_seen_ts_ns),
                    "reappear_veto": bool(memory.reappear_veto),
                    "reappear_group": memory.reappear_group,
                    "reason_codes": list(memory.reason_codes),
                }
            )
        return rows

    def has_pending_resolution(self, now_ns: int) -> bool:
        return bool(self.pending_candidates(now_ns))

    def pending_candidates(self, now_ns: int) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for memory in self._memory.values():
            if memory.resolved:
                continue
            if memory.state not in {"candidate", "tentative"}:
                continue
            rows.append(
                {
                    "track_id": int(memory.track_id),
                    "group": memory.group,
                    "state": memory.state,
                    "decision_id": memory.decision_id,
                    "pocket_index": memory.pocket_index,
                    "missing_ms": int(self._memory_missing_ms(memory, now_ns)),
                    "crossed_mouth": bool(memory.crossed_mouth),
                    "crossed_throat": bool(memory.crossed_throat),
                    "entered_interior": bool(memory.entered_interior),
                    "candidate_reason": memory.candidate_reason,
                }
            )
        return rows

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
        sample: _PocketSample,
        frame: TracksFrame,
        active: bool,
        events: List[Event],
    ) -> None:
        reappeared_now = False
        if memory.nonvisible_since_ns is not None and not memory.resolved and memory.state in {"candidate", "tentative"}:
            missing_ms = self._elapsed_ms(frame.ts_cam_ns, memory.nonvisible_since_ns)
            if missing_ms <= self._reappear_window_ms():
                reappeared_group = normalize_group(track.group)
                reason_codes = ["reappeared_same_track"]
                if reappeared_group is not None and reappeared_group != memory.group:
                    memory.reappear_group = reappeared_group
                    reason_codes.append("reappeared_group_changed")
                self._emit_reappeared(memory, track.track_id, frame, missing_ms, events)
                self._reject(memory, frame, events, reason_codes=reason_codes, candidate_reason="reappeared_visible")
                reappeared_now = True

        if active and not reappeared_now:
            self._update_candidate_evidence(memory, track, sample, frame, events)

        if active and sample.inside_playable and sample.zone is None and memory.state in {"candidate", "tentative"} and not memory.resolved:
            self._reject(memory, frame, events, reason_codes=["back_to_table"], candidate_reason="back_to_table")

        if active and sample.zone == "mouth" and self._speed(track) <= self._still_speed(track) * 1.25:
            if memory.resting_mouth_since_ns is None:
                memory.resting_mouth_since_ns = frame.ts_cam_ns
            if memory.pocket_index is None:
                memory.pocket_index = sample.pocket_index
        elif sample.zone not in {"mouth", "throat", "interior"}:
            memory.resting_mouth_since_ns = None

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
        memory.last_inward_speed_mm_s = float(sample.inward_speed_mm_s)
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
        if self._should_emit_tentative(memory):
            self._tentative(memory, frame, events, reason_codes=self._tentative_reason_codes(memory))
        self._finalize_if_ready(memory, frame, events, allow_occluded_commit=self._occluded_commit_allowed(memory))

    def _handle_absent(self, memory: _PocketMemory, frame: TracksFrame, events: List[Event]) -> None:
        if not self._has_pocket_interest(memory):
            return
        if memory.absent_since_ns is None:
            memory.absent_since_ns = frame.ts_cam_ns
        if memory.nonvisible_since_ns is None:
            memory.nonvisible_since_ns = memory.absent_since_ns
        if self._should_emit_tentative(memory):
            self._tentative(memory, frame, events, reason_codes=self._tentative_reason_codes(memory))
        self._finalize_if_ready(memory, frame, events, allow_occluded_commit=True)

    def _update_candidate_evidence(
        self,
        memory: _PocketMemory,
        track: TrackObservation,
        sample: _PocketSample,
        frame: TracksFrame,
        events: List[Event],
    ) -> None:
        zone = sample.zone
        if zone not in {"mouth", "throat", "interior"}:
            return

        memory.pocket_index = sample.pocket_index
        memory.crossed_mouth = True
        if zone in {"throat", "interior"}:
            memory.crossed_throat = True
        if zone == "interior":
            memory.entered_interior = True

        reason = self._candidate_reason(track, sample)
        if reason is None:
            return
        if memory.state == "on_table":
            memory.state = "candidate"
            memory.candidate_since_ns = frame.ts_cam_ns
            memory.candidate_reason = reason
            memory.decision_id = self._ensure_decision_id(memory)
            events.append(
                Event(
                    name="POCKET_CANDIDATE",
                    ts_cam_ns=frame.ts_cam_ns,
                    frame_id=frame.frame_id,
                    payload={
                        "track_id": memory.track_id,
                        "logical_id": f"track:{memory.track_id}",
                        "group": memory.group,
                        "pocket_index": memory.pocket_index,
                        "decision_id": memory.decision_id,
                        "candidate_reason": reason,
                        "distance_to_pocket_mm": sample.distance_mm,
                        "evidence": self._evidence(memory, frame.ts_cam_ns),
                    },
                    confidence=0.72 if zone in {"throat", "interior"} else 0.58,
                )
            )
        elif memory.candidate_reason is None:
            memory.candidate_reason = reason

    def _tentative(
        self,
        memory: _PocketMemory,
        frame: TracksFrame,
        events: List[Event],
        *,
        reason_codes: list[str],
    ) -> None:
        if memory.tentative_emitted:
            return
        memory.state = "tentative"
        memory.decision = "tentative"
        memory.tentative_since_ns = frame.ts_cam_ns
        memory.tentative_emitted = True
        memory.reason_codes = list(reason_codes)
        memory.decision_id = self._ensure_decision_id(memory)
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

    def _finalize_if_ready(
        self,
        memory: _PocketMemory,
        frame: TracksFrame,
        events: List[Event],
        *,
        allow_occluded_commit: bool,
    ) -> None:
        missing_ms = self._memory_missing_ms(memory, frame.ts_cam_ns)
        if missing_ms < self._confirm_missing_ms():
            return
        if self._strong_confirmation_evidence(memory) and not memory.reappear_veto:
            if not allow_occluded_commit:
                return
            self._confirm(memory, frame, events)
            return

        reason_codes = self._review_reason_codes(memory)
        self._review(memory, frame, events, reason_codes=reason_codes)

    def _confirm(self, memory: _PocketMemory, frame: TracksFrame, events: List[Event]) -> None:
        memory.state = "commit_ready"
        memory.decision = "commit_ready"
        memory.resolved = True
        payload = self._decision_payload(
            memory,
            frame.ts_cam_ns,
            decision="commit_ready",
            review_required=False,
            reason_codes=["ready_for_turn_commit"],
        )
        events.append(Event(name="POCKET_COMMIT_READY", ts_cam_ns=frame.ts_cam_ns, frame_id=frame.frame_id, payload=payload, confidence=0.92))

    def _review(self, memory: _PocketMemory, frame: TracksFrame, events: List[Event], *, reason_codes: list[str]) -> None:
        if memory.resolved:
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
        if memory.resolved:
            return
        memory.state = "rejected"
        memory.decision = "rejected"
        memory.resolved = True
        memory.candidate_reason = candidate_reason
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
        sample: _PocketSample,
        frame: TracksFrame,
        events: List[Event],
    ) -> bool:
        vetoed = False
        if sample.pocket_index is None:
            return False
        center_mm = self._center_mm(track)
        for candidate in self._memory.values():
            if candidate.track_id == memory.track_id or candidate.resolved:
                continue
            if "cue" in {candidate.group, memory.group}:
                continue
            if candidate.pocket_index != sample.pocket_index:
                continue
            if candidate.nonvisible_since_ns is None:
                continue
            if self._elapsed_ms(frame.ts_cam_ns, candidate.nonvisible_since_ns) > self._reappear_window_ms():
                continue
            if self._distance(candidate.last_center_mm, center_mm) > self._reappear_match_distance_mm():
                continue
            candidate.reappear_veto = True
            candidate.reappear_track_id = int(track.track_id)
            candidate.reappear_group = memory.group
            self._emit_reappeared(candidate, track.track_id, frame, self._memory_missing_ms(candidate, frame.ts_cam_ns), events)
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
                    "pocket_index": memory.pocket_index,
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
        return {
            "track_id": memory.track_id,
            "logical_id": f"track:{memory.track_id}",
            "group": memory.group,
            "pocket_index": memory.pocket_index,
            "decision_id": self._ensure_decision_id(memory),
            "decision": decision,
            "review_required": bool(review_required),
            "reason_codes": list(reason_codes),
            "candidate_reason": memory.candidate_reason,
            "last_center_mm": list(memory.last_center_mm),
            "last_center_px": list(memory.last_center_px),
            "missing_ms": self._memory_missing_ms(memory, now_ns),
            "confirmation": "two_stage_commit",
            "evidence": self._evidence(memory, now_ns),
        }

    def _evidence(self, memory: _PocketMemory, now_ns: int) -> dict[str, object]:
        return {
            "zone": memory.last_zone,
            "inward_speed_mm_s": float(memory.last_inward_speed_mm_s),
            "candidate_reason": memory.candidate_reason,
            "missing_ms": self._memory_missing_ms(memory, now_ns),
            "reappear_veto": bool(memory.reappear_veto),
            "reappear_track_id": memory.reappear_track_id,
            "reappear_group": memory.reappear_group,
            "crossed_mouth": bool(memory.crossed_mouth),
            "crossed_throat": bool(memory.crossed_throat),
            "entered_interior": bool(memory.entered_interior),
            "distance_mm": memory.last_distance_mm,
            "depth_mm": float(memory.last_depth_mm),
            "lateral_mm": float(memory.last_lateral_mm),
            "last_visibility": memory.last_visibility,
            "lost_frames": int(memory.last_lost_frames),
            "decision": memory.decision,
        }

    def _sample(self, track: TrackObservation) -> _PocketSample:
        center_mm = self._center_mm(track)
        velocity = self._velocity(track)
        inside_playable = self._inside_playable(center_mm)
        geometries = self._pocket_geometries()
        if not geometries:
            return _PocketSample(None, None, None, 0.0, 0.0, 0.0, inside_playable)

        pos = np.asarray(center_mm, dtype=np.float32)
        vel = np.asarray(velocity, dtype=np.float32)
        best_geometry = min(geometries, key=lambda item: float(np.linalg.norm(np.asarray(item.center_mm, dtype=np.float32) - pos)))
        center = np.asarray(best_geometry.center_mm, dtype=np.float32)
        mouth_mid = np.asarray(best_geometry.mouth_mid_mm, dtype=np.float32)
        tangent = np.asarray(best_geometry.tangent_unit, dtype=np.float32)
        inward = np.asarray(best_geometry.inward_normal, dtype=np.float32)
        delta = pos - mouth_mid
        distance = float(np.linalg.norm(center - pos))
        depth = float(np.dot(delta, inward))
        lateral = float(abs(np.dot(delta, tangent)))
        toward_center = center - pos
        toward_center_norm = float(np.linalg.norm(toward_center))
        if toward_center_norm > 1e-6:
            toward_center = toward_center / toward_center_norm
        else:
            toward_center = inward
        inward_speed = float(np.dot(vel, toward_center))

        mouth_radius = max(float(getattr(self.config, "pocket_mouth_radius_mm", 125.0)), self.ball_diameter_mm * 2.05)
        throat_radius = max(float(getattr(self.config, "pocket_throat_radius_mm", 75.0)), self.ball_diameter_mm * 1.20)
        interior_radius = max(float(getattr(self.config, "pocket_interior_radius_mm", 44.0)), self.ball_diameter_mm * 0.72)
        lateral_buffer = self.ball_diameter_mm * 0.55
        mouth_depth_floor = -self.ball_diameter_mm * 0.55

        zone: Optional[str] = None
        if depth >= best_geometry.interior_depth_mm or distance <= interior_radius:
            zone = "interior"
        elif (depth >= best_geometry.throat_depth_mm and lateral <= best_geometry.throat_half_width_mm + lateral_buffer) or distance <= throat_radius:
            zone = "throat"
        elif (depth >= mouth_depth_floor and lateral <= best_geometry.mouth_half_width_mm + lateral_buffer) or distance <= mouth_radius:
            zone = "mouth"

        return _PocketSample(
            zone=zone,
            pocket_index=best_geometry.index,
            distance_mm=distance,
            depth_mm=depth,
            lateral_mm=lateral,
            inward_speed_mm_s=inward_speed,
            inside_playable=inside_playable,
        )

    def _pocket_geometries(self) -> list[_PocketGeometry]:
        geometries: list[_PocketGeometry] = []
        if self.pockets_mm:
            for index, center in enumerate(self.pockets_mm):
                curve = self.pocket_curves_mm[index] if index < len(self.pocket_curves_mm) else []
                geometries.append(self._build_geometry(index, center, curve))
        elif self.pocket_curves_mm:
            for index, curve in enumerate(self.pocket_curves_mm):
                center = self._curve_center(curve)
                geometries.append(self._build_geometry(index, center, curve))
        return geometries

    def _build_geometry(
        self,
        index: int,
        center: tuple[float, float],
        curve: Sequence[tuple[float, float]],
    ) -> _PocketGeometry:
        center_vec = np.asarray(center, dtype=np.float32)
        if len(curve) >= 2:
            curve_points = np.asarray(curve, dtype=np.float32).reshape((-1, 2))
            mouth_a = curve_points[0]
            mouth_b = curve_points[-1]
            mouth_mid = np.mean(curve_points, axis=0)
            tangent = mouth_b - mouth_a
            tangent = self._unit(tangent, fallback=np.asarray([1.0, 0.0], dtype=np.float32))
            inward = center_vec - mouth_mid
            inward = self._unit(inward, fallback=np.asarray([0.0, 1.0], dtype=np.float32))
            half_width = max(float(np.linalg.norm(mouth_b - mouth_a)) * 0.5, self.ball_diameter_mm * 0.95)
        else:
            table_center = self._table_center()
            inward = center_vec - table_center
            inward = self._unit(inward, fallback=np.asarray([0.0, 1.0], dtype=np.float32))
            tangent = np.asarray([-inward[1], inward[0]], dtype=np.float32)
            mouth_mid = center_vec - inward * float(max(self.ball_diameter_mm * 1.15, self.config.pocket_mouth_radius_mm * 0.55))
            half_width = max(float(getattr(self.config, "pocket_mouth_radius_mm", 125.0)) * 0.68, self.ball_diameter_mm * 1.05)
            mouth_a = mouth_mid - tangent * half_width
            mouth_b = mouth_mid + tangent * half_width

        throat_half_width = max(self.ball_diameter_mm * 0.62, half_width * 0.56)
        throat_depth = max(self.ball_diameter_mm * 0.55, float(getattr(self.config, "pocket_throat_radius_mm", 75.0)) * 0.55)
        interior_depth = max(throat_depth + self.ball_diameter_mm * 0.45, float(getattr(self.config, "pocket_interior_radius_mm", 44.0)) * 1.25)
        return _PocketGeometry(
            index=index,
            center_mm=(float(center_vec[0]), float(center_vec[1])),
            mouth_a_mm=(float(mouth_a[0]), float(mouth_a[1])),
            mouth_b_mm=(float(mouth_b[0]), float(mouth_b[1])),
            mouth_mid_mm=(float(mouth_mid[0]), float(mouth_mid[1])),
            tangent_unit=(float(tangent[0]), float(tangent[1])),
            inward_normal=(float(inward[0]), float(inward[1])),
            mouth_half_width_mm=float(half_width),
            throat_half_width_mm=float(throat_half_width),
            throat_depth_mm=float(throat_depth),
            interior_depth_mm=float(interior_depth),
        )

    def _candidate_reason(self, track: TrackObservation, sample: _PocketSample) -> Optional[str]:
        if sample.zone == "interior":
            return "entered_interior"
        if sample.zone == "throat":
            return "crossed_throat"
        if sample.zone == "mouth" and sample.inward_speed_mm_s >= self._inward_speed_threshold(track):
            return "mouth_inward_trend"
        return None

    def _tentative_reason_codes(self, memory: _PocketMemory) -> list[str]:
        if memory.entered_interior:
            return ["interior_missing"]
        if memory.crossed_throat:
            return ["throat_missing"]
        if memory.crossed_mouth:
            return ["mouth_missing"]
        return ["near_pocket_missing"]

    def _review_reason_codes(self, memory: _PocketMemory) -> list[str]:
        if memory.reappear_veto:
            reasons = ["reappeared_near_pocket"]
            if memory.reappear_group is not None and memory.reappear_group != memory.group:
                reasons.append("reappeared_group_changed")
            return reasons
        if memory.resting_mouth_since_ns is not None and not memory.crossed_throat:
            return ["mouth_rest_disappear_requires_review"]
        if not self._strong_confirmation_evidence(memory):
            return ["insufficient_pocket_evidence"]
        return ["pocket_review_required"]

    def _should_emit_tentative(self, memory: _PocketMemory) -> bool:
        return (not memory.tentative_emitted) and self._has_pocket_interest(memory)

    def _strong_confirmation_evidence(self, memory: _PocketMemory) -> bool:
        return bool(memory.entered_interior or memory.crossed_throat)

    def _occluded_commit_allowed(self, memory: _PocketMemory) -> bool:
        if memory.absent_since_ns is not None:
            return True
        return int(memory.last_lost_frames) >= self._occluded_lost_frames_threshold()

    def _has_pocket_interest(self, memory: _PocketMemory) -> bool:
        return bool(
            memory.pocket_index is not None
            or memory.crossed_mouth
            or memory.crossed_throat
            or memory.entered_interior
            or memory.last_zone in {"mouth", "throat", "interior"}
            or memory.resting_mouth_since_ns is not None
            or memory.state in {"candidate", "tentative"}
        )

    def _memory_missing_ms(self, memory: _PocketMemory, now_ns: int) -> int:
        since_ns = memory.nonvisible_since_ns or memory.absent_since_ns
        if since_ns is None:
            return 0
        return self._elapsed_ms(now_ns, since_ns)

    def _ensure_decision_id(self, memory: _PocketMemory) -> str:
        if memory.decision_id is None:
            memory.decision_id = f"pocket:{self._decision_seq}"
            self._decision_seq += 1
        return memory.decision_id

    def _curve_center(self, curve: Sequence[tuple[float, float]]) -> tuple[float, float]:
        curve_points = np.asarray(list(curve or []), dtype=np.float32).reshape((-1, 2))
        if curve_points.shape[0] == 0:
            return (0.0, 0.0)
        mean = np.mean(curve_points, axis=0)
        return (float(mean[0]), float(mean[1]))

    def _table_center(self) -> np.ndarray:
        if len(self.inner_polygon_mm) >= 3:
            pts = np.asarray(self.inner_polygon_mm, dtype=np.float32).reshape((-1, 2))
            return np.mean(pts, axis=0)
        if self.pockets_mm:
            pts = np.asarray(self.pockets_mm, dtype=np.float32).reshape((-1, 2))
            return np.mean(pts, axis=0)
        return np.asarray([0.0, 0.0], dtype=np.float32)

    def _inside_playable(self, center_mm: tuple[float, float]) -> bool:
        if len(self.inner_polygon_mm) < 3:
            return True
        polygon = np.asarray(self.inner_polygon_mm, dtype=np.float32).reshape((-1, 1, 2))
        dist = cv2.pointPolygonTest(polygon, (float(center_mm[0]), float(center_mm[1])), False)
        return dist >= 0

    def _center_mm(self, track: TrackObservation) -> tuple[float, float]:
        point = track.center_mm if track.center_mm is not None else track.center_px
        return (float(point[0]), float(point[1]))

    def _velocity(self, track: TrackObservation) -> tuple[float, float]:
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

    def _track_relevant(self, track: TrackObservation) -> bool:
        if track.visibility == "visible":
            return float(track.quality) > 0.25
        return True

    def _confirm_missing_ms(self) -> int:
        return max(1, int(getattr(self.config, "pocket_confirm_missing_ms", 350)))

    def _reappear_window_ms(self) -> int:
        return max(1, int(getattr(self.config, "pocket_reappear_window_ms", 800)))

    def _reappear_match_distance_mm(self) -> float:
        return max(self.ball_diameter_mm * 2.2, float(getattr(self.config, "pocket_mouth_radius_mm", 125.0)) * 0.75)

    def _occluded_lost_frames_threshold(self) -> int:
        return 4

    @staticmethod
    def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return float(np.linalg.norm(np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32)))

    @staticmethod
    def _unit(vector: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return np.asarray(fallback, dtype=np.float32)
        return np.asarray(vector, dtype=np.float32) / norm

    @staticmethod
    def _elapsed_ms(now_ns: int, since_ns: int) -> int:
        return max(0, int(round((int(now_ns) - int(since_ns)) / 1_000_000.0)))
