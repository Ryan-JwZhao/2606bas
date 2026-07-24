from __future__ import annotations

from math import cos, pi, sin
from typing import Sequence

import numpy as np

from .models import BallTargetZone, SetupTargetZoneGuide, TrainingScenario


def target_zone_contains_point(
    zone: BallTargetZone,
    point_mm: tuple[float, float],
    reachable_polygon_mm: Sequence[tuple[float, float]],
) -> bool:
    """Apply the canonical normalized ellipse test used by setup validation and guidance."""

    frame = _target_zone_frame(reachable_polygon_mm)
    if frame is None:
        return False
    low, span = frame
    normalized = (np.asarray(point_mm, dtype=np.float32) - low) / span
    delta = np.abs(normalized - np.asarray(zone.center, dtype=np.float32))
    tolerance = np.maximum(np.asarray(zone.tolerance, dtype=np.float32), 1e-4)
    return float(np.sum((delta / tolerance) ** 2)) <= 1.0


def build_setup_target_zone_guides(
    scenario: TrainingScenario,
    reachable_polygon_mm: Sequence[tuple[float, float]],
    *,
    samples: int = 48,
) -> list[SetupTargetZoneGuide]:
    """Resolve normalized scenario zones into the same table-mm ellipses used by setup validation."""

    if not scenario.zones:
        return []
    frame = _target_zone_frame(reachable_polygon_mm)
    if frame is None:
        return []
    low, span = frame
    point_count = max(12, int(samples))
    guides: list[SetupTargetZoneGuide] = []
    for zone in scenario.zones:
        center = np.asarray(zone.center, dtype=np.float32)
        tolerance = np.maximum(np.asarray(zone.tolerance, dtype=np.float32), 1e-4)
        points: list[tuple[float, float]] = []
        for index in range(point_count):
            angle = 2.0 * pi * float(index) / float(point_count)
            normalized = center + tolerance * np.asarray([cos(angle), sin(angle)], dtype=np.float32)
            point_mm = low + normalized * span
            points.append((float(point_mm[0]), float(point_mm[1])))
        points.append(points[0])
        guides.append(SetupTargetZoneGuide(ball=int(zone.ball), polygon_mm=tuple(points)))
    return guides


def _target_zone_frame(
    reachable_polygon_mm: Sequence[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(reachable_polygon_mm) < 3:
        return None
    polygon = np.asarray(reachable_polygon_mm, dtype=np.float32).reshape((-1, 2))
    return np.min(polygon, axis=0), np.maximum(np.ptp(polygon, axis=0), 1.0)
