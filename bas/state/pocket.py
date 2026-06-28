from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..config import StateConfig
from ..schemas import Event, MatchPhase, TrackObservation, TracksFrame
from .models import Group, normalize_group


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
    candidate_since_ns: Optional[int] = None
    missing_since_ns: Optional[int] = None
    resting_mouth_since_ns: Optional[int] = None
    pocket_index: Optional[int] = None
    crossed_throat: bool = False
    confirmed: bool = False
    last_zone: Optional[str] = None


class PerBallPocketFSM:
    """Multi-frame pocket confirmation with reappearance rejection."""

    def __init__(self, config: StateConfig):
        self.config = config
        self.inner_polygon_mm: list[tuple[float, float]] = []
        self.pockets_mm: list[tuple[float, float]] = []
        self.ball_diameter_mm = 57.15
        self._memory: Dict[int, _PocketMemory] = {}

    def reset(self) -> None:
        self._memory.clear()

    def set_table_context(
        self,
        *,
        inner_polygon_mm: Optional[list[tuple[float, float]]] = None,
        pockets_mm: Optional[list[tuple[float, float]]] = None,
        ball_diameter_mm: Optional[float] = None,
    ) -> None:
        if inner_polygon_mm is not None:
            self.inner_polygon_mm = [(float(x), float(y)) for x, y in inner_polygon_mm]
        if pockets_mm is not None:
            self.pockets_mm = [(float(x), float(y)) for x, y in pockets_mm]
        if ball_diameter_mm is not None:
            self.ball_diameter_mm = float(ball_diameter_mm)

    def update(self, frame: TracksFrame, phase: MatchPhase) -> List[Event]:
        events: List[Event] = []
        active = phase in {MatchPhase.SHOT_ACTIVE, MatchPhase.SETTLING, MatchPhase.TURN_RESOLVE}
        visible = {
            track.track_id: track
            for track in frame.tracks
            if track.visibility == "visible" and track.quality > 0.25 and normalize_group(track.group) is not None
        }

        for track in visible.values():
            group = normalize_group(track.group)
            if group is None:
                continue
            center_mm = self._center_mm(track)
            speed = self._speed(track)
            zone, pocket_index, dist = self._pocket_zone(center_mm)
            memory = self._memory.get(track.track_id)
            if memory is not None and memory.missing_since_ns is not None and not memory.confirmed:
                missing_ms = self._elapsed_ms(frame.ts_cam_ns, memory.missing_since_ns)
                if memory.state == "pocket_candidate" and missing_ms <= self._reappear_window_ms():
                    events.append(
                        Event(
                            name="POCKET_REAPPEARED",
                            ts_cam_ns=frame.ts_cam_ns,
                            frame_id=frame.frame_id,
                            payload={"track_id": track.track_id, "group": group, "missing_ms": missing_ms},
                            confidence=0.85,
                        )
                    )
                memory.state = "on_table"
                memory.missing_since_ns = None
                memory.crossed_throat = False

            if memory is None or memory.confirmed:
                memory = _PocketMemory(
                    track_id=track.track_id,
                    group=group,
                    last_center_px=track.center_px,
                    last_center_mm=center_mm,
                    last_velocity=self._velocity(track),
                    last_seen_frame=frame.frame_id,
                    last_seen_ts_ns=frame.ts_cam_ns,
                    last_quality=float(track.quality),
                )
                self._memory[track.track_id] = memory

            if active and memory.state == "on_table" and zone in {"throat", "interior"}:
                memory.state = "pocket_candidate"
                memory.candidate_since_ns = frame.ts_cam_ns
                memory.pocket_index = pocket_index
                memory.crossed_throat = True
                events.append(
                    Event(
                        name="POCKET_CANDIDATE",
                        ts_cam_ns=frame.ts_cam_ns,
                        frame_id=frame.frame_id,
                        payload={
                            "track_id": track.track_id,
                            "group": group,
                            "pocket_index": pocket_index,
                            "zone": zone,
                            "distance_to_pocket_mm": dist,
                        },
                        confidence=0.66 if zone == "throat" else 0.78,
                    )
                )
            elif memory.state == "pocket_candidate":
                inside_playable = self._inside_playable(center_mm)
                if inside_playable and zone is None:
                    memory.state = "on_table"
                    memory.candidate_since_ns = None
                    memory.pocket_index = None
                    memory.crossed_throat = False
                    events.append(
                        Event(
                            name="POCKET_REJECTED_BACK_TO_TABLE",
                            ts_cam_ns=frame.ts_cam_ns,
                            frame_id=frame.frame_id,
                            payload={"track_id": track.track_id, "group": group},
                            confidence=0.78,
                        )
                    )

            if active and zone == "mouth" and speed <= self._still_speed(track) * 1.25:
                if memory.resting_mouth_since_ns is None:
                    memory.resting_mouth_since_ns = frame.ts_cam_ns
                memory.pocket_index = pocket_index
            elif zone not in {"mouth", "throat", "interior"}:
                memory.resting_mouth_since_ns = None

            memory.group = group
            memory.last_center_px = track.center_px
            memory.last_center_mm = center_mm
            memory.last_velocity = self._velocity(track)
            memory.last_seen_frame = frame.frame_id
            memory.last_seen_ts_ns = frame.ts_cam_ns
            memory.last_quality = float(track.quality)
            memory.last_zone = zone

        if active:
            visible_ids = set(visible.keys())
            for memory in list(self._memory.values()):
                if memory.track_id in visible_ids or memory.confirmed:
                    continue
                if memory.group == "cue_stick":
                    continue
                if memory.missing_since_ns is None:
                    memory.missing_since_ns = frame.ts_cam_ns
                missing_ms = self._elapsed_ms(frame.ts_cam_ns, memory.missing_since_ns)
                if missing_ms < self._confirm_missing_ms():
                    continue
                if self._can_confirm(memory, frame.ts_cam_ns):
                    memory.confirmed = True
                    memory.state = "pocket_confirmed"
                    payload = self._confirmed_payload(memory, frame.ts_cam_ns)
                    events.append(Event(name="POCKET_CONFIRMED", ts_cam_ns=frame.ts_cam_ns, frame_id=frame.frame_id, payload=payload, confidence=0.90))
                    events.append(Event(name="POT_PROBABLE", ts_cam_ns=frame.ts_cam_ns, frame_id=frame.frame_id, payload=payload, confidence=0.90))
                else:
                    memory.state = "lost"
                    events.append(
                        Event(
                            name="BALL_LOST_UNCONFIRMED",
                            ts_cam_ns=frame.ts_cam_ns,
                            frame_id=frame.frame_id,
                            payload={
                                "track_id": memory.track_id,
                                "group": memory.group,
                                "missing_ms": missing_ms,
                                "last_center_mm": list(memory.last_center_mm),
                            },
                            confidence=0.45,
                        )
                    )
                    memory.confirmed = True

        return events

    def debug_snapshot(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for memory in self._memory.values():
            rows.append(
                {
                    "track_id": int(memory.track_id),
                    "group": memory.group,
                    "state": memory.state,
                    "pocket_index": memory.pocket_index,
                    "crossed_throat": bool(memory.crossed_throat),
                    "confirmed": bool(memory.confirmed),
                    "last_zone": memory.last_zone,
                }
            )
        return rows

    def has_pending_resolution(self, now_ns: int) -> bool:
        return bool(self.pending_candidates(now_ns))

    def pending_candidates(self, now_ns: int) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for memory in self._memory.values():
            if memory.confirmed:
                continue
            if memory.state != "pocket_candidate" and memory.missing_since_ns is None:
                continue
            if memory.missing_since_ns is not None:
                missing_ms = self._elapsed_ms(now_ns, memory.missing_since_ns)
                if missing_ms > self._confirm_missing_ms():
                    continue
            else:
                missing_ms = 0
            rows.append(
                {
                    "track_id": int(memory.track_id),
                    "group": memory.group,
                    "state": memory.state,
                    "pocket_index": memory.pocket_index,
                    "missing_ms": int(missing_ms),
                    "crossed_throat": bool(memory.crossed_throat),
                }
            )
        return rows

    def _can_confirm(self, memory: _PocketMemory, now_ns: int) -> bool:
        if memory.state == "pocket_candidate" and memory.crossed_throat:
            return True
        if memory.resting_mouth_since_ns is not None:
            return self._elapsed_ms(now_ns, memory.resting_mouth_since_ns) <= self._mouth_settle_ms()
        return False

    def _confirmed_payload(self, memory: _PocketMemory, now_ns: int) -> dict[str, object]:
        missing_since = memory.missing_since_ns or memory.last_seen_ts_ns
        return {
            "track_id": memory.track_id,
            "logical_id": f"track:{memory.track_id}",
            "group": memory.group,
            "pocket_index": memory.pocket_index,
            "last_center_mm": list(memory.last_center_mm),
            "last_center_px": list(memory.last_center_px),
            "missing_ms": self._elapsed_ms(now_ns, missing_since),
            "confirmation": "multi_frame",
        }

    def _pocket_zone(self, center_mm: tuple[float, float]) -> tuple[Optional[str], Optional[int], Optional[float]]:
        if not self.pockets_mm:
            return None, None, None
        pos = np.asarray(center_mm, dtype=np.float32)
        pockets = np.asarray(self.pockets_mm, dtype=np.float32).reshape((-1, 2))
        distances = np.linalg.norm(pockets - pos.reshape((1, 2)), axis=1)
        idx = int(np.argmin(distances))
        dist = float(distances[idx])
        mouth = max(float(getattr(self.config, "pocket_mouth_radius_mm", 125.0)), self.ball_diameter_mm * 2.05)
        throat = max(float(getattr(self.config, "pocket_throat_radius_mm", 75.0)), self.ball_diameter_mm * 1.20)
        interior = max(float(getattr(self.config, "pocket_interior_radius_mm", 44.0)), self.ball_diameter_mm * 0.72)
        if dist <= interior:
            return "interior", idx, dist
        if dist <= throat:
            return "throat", idx, dist
        if dist <= mouth:
            return "mouth", idx, dist
        return None, idx, dist

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

    def _confirm_missing_ms(self) -> int:
        return max(1, int(getattr(self.config, "pocket_confirm_missing_ms", 350)))

    def _reappear_window_ms(self) -> int:
        return max(1, int(getattr(self.config, "pocket_reappear_window_ms", 800)))

    def _mouth_settle_ms(self) -> int:
        return max(1, int(getattr(self.config, "pocket_mouth_settle_ms", 3000)))

    @staticmethod
    def _elapsed_ms(now_ns: int, since_ns: int) -> int:
        return max(0, int(round((int(now_ns) - int(since_ns)) / 1_000_000.0)))
