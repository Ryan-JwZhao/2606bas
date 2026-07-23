from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..schemas import TrackObservation, TracksFrame
from .models import CueBallControlGoal
from .numbered_tracker import ball_number_from_track


@dataclass(frozen=True)
class CueBallStop:
    position_mm: tuple[float, float]
    stable_since_ns: int
    observed_at_ns: int


@dataclass(frozen=True)
class CueBallTargetRegion:
    after_ball: int
    kind: str
    center_mm: tuple[float, float]
    radius_mm: float
    polygon_mm: tuple[tuple[float, float], ...]

    def contains(self, position_mm: tuple[float, float]) -> bool:
        point = np.asarray(position_mm, dtype=np.float32)
        center = np.asarray(self.center_mm, dtype=np.float32)
        return float(np.linalg.norm(point - center)) <= float(self.radius_mm)


class CueBallStopObserver:
    """Returns a stop position only after the visible cue ball stays still."""

    def __init__(
        self,
        *,
        still_speed_mm_s: float = 8.0,
        stable_hold_ms: int = 650,
        jitter_tolerance_mm: float = 12.0,
    ) -> None:
        self.still_speed_mm_s = max(0.1, float(still_speed_mm_s))
        self.stable_hold_ns = max(100, int(stable_hold_ms)) * 1_000_000
        self.jitter_tolerance_mm = max(1.0, float(jitter_tolerance_mm))
        self._candidate_since_ns = 0
        self._candidate_positions: list[tuple[float, float]] = []
        self._last_stop: CueBallStop | None = None
        self.status = "missing"

    @property
    def last_stop(self) -> CueBallStop | None:
        return self._last_stop

    def reset(self) -> None:
        self._candidate_since_ns = 0
        self._candidate_positions.clear()
        self._last_stop = None
        self.status = "missing"

    def update(self, tracks_frame: TracksFrame) -> CueBallStop | None:
        cue = _visible_cue_ball(tracks_frame.tracks)
        if cue is None or cue.center_mm is None:
            self._clear_candidate("missing")
            return None
        position = (float(cue.center_mm[0]), float(cue.center_mm[1]))
        speed = _speed_mm_s(cue)
        if speed is not None and speed > self.still_speed_mm_s:
            self._clear_candidate("moving")
            return None

        ts_ns = int(tracks_frame.ts_cam_ns)
        if not self._candidate_positions:
            self._candidate_since_ns = ts_ns
            self._candidate_positions = [position]
            self._last_stop = None
            self.status = "settling"
            return None

        anchor = np.mean(np.asarray(self._candidate_positions, dtype=np.float32), axis=0)
        if float(np.linalg.norm(np.asarray(position, dtype=np.float32) - anchor)) > self.jitter_tolerance_mm:
            self._candidate_since_ns = ts_ns
            self._candidate_positions = [position]
            self._last_stop = None
            self.status = "settling"
            return None

        self._candidate_positions.append(position)
        if len(self._candidate_positions) > 12:
            self._candidate_positions.pop(0)
        if ts_ns - self._candidate_since_ns < self.stable_hold_ns:
            self.status = "settling"
            return None

        averaged = np.mean(np.asarray(self._candidate_positions, dtype=np.float32), axis=0)
        self._last_stop = CueBallStop(
            position_mm=(float(averaged[0]), float(averaged[1])),
            stable_since_ns=int(self._candidate_since_ns),
            observed_at_ns=ts_ns,
        )
        self.status = "stopped"
        return self._last_stop

    def _clear_candidate(self, status: str) -> None:
        self._candidate_since_ns = 0
        self._candidate_positions.clear()
        self._last_stop = None
        self.status = status


def resolve_cue_ball_target_region(
    goal: CueBallControlGoal,
    *,
    start_positions_mm: Mapping[int, tuple[float, float]],
    table_polygon_mm: Sequence[tuple[float, float]],
    ball_diameter_mm: float,
) -> CueBallTargetRegion | None:
    radius_mm = max(float(ball_diameter_mm), float(goal.radius_ball_diameters) * float(ball_diameter_mm))
    if goal.kind == "table_zone":
        if goal.center is None or len(table_polygon_mm) < 3:
            return None
        polygon = np.asarray(table_polygon_mm, dtype=np.float32).reshape((-1, 2))
        low = np.min(polygon, axis=0)
        span = np.maximum(np.ptp(polygon, axis=0), 1.0)
        center = low + np.asarray(goal.center, dtype=np.float32) * span
    elif goal.kind == "stop":
        cue = start_positions_mm.get(0)
        if cue is None:
            return None
        center = np.asarray(cue, dtype=np.float32)
    elif goal.kind in {"follow", "draw"}:
        cue = start_positions_mm.get(0)
        target = start_positions_mm.get(int(goal.after_ball))
        if cue is None or target is None:
            return None
        cue_point = np.asarray(cue, dtype=np.float32)
        target_point = np.asarray(target, dtype=np.float32)
        direction = target_point - cue_point
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            return None
        direction /= length
        amount = float(goal.distance_ball_diameters) * float(ball_diameter_mm)
        center = target_point + direction * amount * (1.0 if goal.kind == "follow" else -1.0)
    else:
        return None

    if len(table_polygon_mm) >= 3:
        table_polygon = np.asarray(table_polygon_mm, dtype=np.float32).reshape((-1, 2))
        low = np.min(table_polygon, axis=0)
        high = np.max(table_polygon, axis=0)
        inset_low = low + radius_mm
        inset_high = high - radius_mm
        for axis in range(2):
            center[axis] = (
                0.5 * (low[axis] + high[axis])
                if inset_low[axis] > inset_high[axis]
                else np.clip(center[axis], inset_low[axis], inset_high[axis])
            )

    center_mm = (float(center[0]), float(center[1]))
    points = tuple(
        (
            center_mm[0] + radius_mm * math.cos(2.0 * math.pi * index / 64.0),
            center_mm[1] + radius_mm * math.sin(2.0 * math.pi * index / 64.0),
        )
        for index in range(65)
    )
    return CueBallTargetRegion(
        after_ball=int(goal.after_ball),
        kind=str(goal.kind),
        center_mm=center_mm,
        radius_mm=radius_mm,
        polygon_mm=points,
    )


def _visible_cue_ball(tracks: Sequence[TrackObservation]) -> TrackObservation | None:
    return next(
        (
            track
            for track in tracks
            if track.visibility == "visible" and ball_number_from_track(track) == 0
        ),
        None,
    )


def _speed_mm_s(track: TrackObservation) -> float | None:
    if track.velocity_mm_s is None:
        return None
    return float(np.linalg.norm(np.asarray(track.velocity_mm_s, dtype=np.float32)))


__all__ = [
    "CueBallStop",
    "CueBallStopObserver",
    "CueBallTargetRegion",
    "resolve_cue_ball_target_region",
]
