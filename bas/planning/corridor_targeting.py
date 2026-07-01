from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..utils import unit


OBJECT_GROUPS = {"solid", "stripe", "black"}


@dataclass(frozen=True)
class CorridorTargetCandidate:
    track_id: int
    group: str
    center_px: tuple[float, float]
    lateral_px: float
    forward_px: float
    distance_px: float
    quality: float


def rank_object_balls_in_corridor(
    *,
    cue_ball: object,
    balls: Sequence[object],
    direction_px: np.ndarray,
    half_width_px: float,
    min_quality: float = 0.25,
) -> list[CorridorTargetCandidate]:
    direction = unit(np.asarray(direction_px, dtype=np.float32).reshape((2,)))
    if float(np.linalg.norm(direction)) < 1.0e-6:
        return []
    cue_track_id = int(getattr(cue_ball, "track_id", -1))
    cue_center = np.asarray(getattr(cue_ball, "center_px"), dtype=np.float32).reshape((2,))
    normal = np.asarray([-float(direction[1]), float(direction[0])], dtype=np.float32)
    half_width = max(0.5, float(half_width_px))
    ranked: list[tuple[tuple[float, float, float, float], CorridorTargetCandidate]] = []
    for ball in balls:
        group = str(getattr(ball, "group", "")).strip().lower()
        if group not in OBJECT_GROUPS:
            continue
        if int(getattr(ball, "track_id", -1)) == cue_track_id:
            continue
        quality = float(getattr(ball, "quality", 0.0))
        if quality <= float(min_quality):
            continue
        center = np.asarray(getattr(ball, "center_px"), dtype=np.float32).reshape((2,))
        vec = center - cue_center
        forward = float(np.dot(vec, direction))
        if forward <= 0.0:
            continue
        lateral = float(np.dot(vec, normal))
        if abs(lateral) > half_width:
            continue
        distance = float(np.linalg.norm(vec))
        ranked.append(
            (
                (abs(lateral), forward, distance, -quality),
                CorridorTargetCandidate(
                    track_id=int(getattr(ball, "track_id")),
                    group=group,
                    center_px=(float(center[0]), float(center[1])),
                    lateral_px=float(lateral),
                    forward_px=float(forward),
                    distance_px=float(distance),
                    quality=float(quality),
                ),
            )
        )
    ranked.sort(key=lambda item: item[0])
    return [candidate for _key, candidate in ranked]
