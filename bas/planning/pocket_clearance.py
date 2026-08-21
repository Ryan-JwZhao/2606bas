from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..utils import angle_deg, unit


PointLike = Sequence[float] | np.ndarray
PocketMouthLike = Sequence[PointLike]


@dataclass(frozen=True)
class PocketEntryAssessment:
    """Physical feasibility of the final object-ball segment through a pocket mouth."""

    feasible: bool
    reason: str
    entrance_angle_deg: float
    jaw_clearance_mm: float
    required_clearance_mm: float
    clearance_margin_mm: float
    mouth_crossing_mm: tuple[float, float] | None = None


@dataclass(frozen=True)
class PocketEntryPath:
    """A full-ball-safe final leg through a finite pocket mouth."""

    path_end_mm: tuple[float, float]
    assessment: PocketEntryAssessment
    standard: str = "full_ball_corridor_v2"


def assess_pocket_entry(
    final_leg_start_mm: PointLike,
    pocket_center_mm: PointLike,
    pocket_mouth_mm: PocketMouthLike | None,
    *,
    ball_radius_mm: float,
    safety_margin_mm: float,
) -> PocketEntryAssessment:
    """Reject an entry whose full ball cannot pass between the two pocket jaws.

    The moving ball is treated as a disc.  Its centre line must cross the finite
    jaw-to-jaw mouth segment and remain at least ``radius + safety`` away from
    both jaw tips.  Missing geometry fails closed so an unverified route cannot
    reach the overlay.
    """

    required = max(0.0, float(ball_radius_mm)) + max(0.0, float(safety_margin_mm))
    if pocket_mouth_mm is None or len(pocket_mouth_mm) != 2:
        return _rejected("missing_mouth", required)

    start = _point(final_leg_start_mm)
    pocket = _point(pocket_center_mm)
    jaw_a = _point(pocket_mouth_mm[0])
    jaw_b = _point(pocket_mouth_mm[1])
    path = pocket - start
    mouth = jaw_b - jaw_a
    path_length = float(np.linalg.norm(path))
    mouth_span = float(np.linalg.norm(mouth))
    if path_length <= 1.0e-6:
        return _rejected("degenerate_path", required)
    if mouth_span <= 1.0e-6:
        return _rejected("degenerate_mouth", required)

    mouth_midpoint = 0.5 * (jaw_a + jaw_b)
    outward_axis = pocket - mouth_midpoint
    if float(np.linalg.norm(outward_axis)) <= 1.0e-6:
        return _rejected("degenerate_mouth", required)
    entrance_angle = float(angle_deg(unit(path), unit(outward_axis)))

    crossing = _segment_intersection(start, pocket, jaw_a, jaw_b)
    jaw_clearance = min(
        _point_segment_distance(jaw_a, start, pocket),
        _point_segment_distance(jaw_b, start, pocket),
    )
    clearance_margin = float(jaw_clearance - required)
    if crossing is None:
        return PocketEntryAssessment(
            feasible=False,
            reason="misses_mouth",
            entrance_angle_deg=entrance_angle,
            jaw_clearance_mm=float(jaw_clearance),
            required_clearance_mm=required,
            clearance_margin_mm=clearance_margin,
        )
    if clearance_margin < -1.0e-6:
        return PocketEntryAssessment(
            feasible=False,
            reason="jaw_clearance",
            entrance_angle_deg=entrance_angle,
            jaw_clearance_mm=float(jaw_clearance),
            required_clearance_mm=required,
            clearance_margin_mm=clearance_margin,
            mouth_crossing_mm=(float(crossing[0]), float(crossing[1])),
        )
    return PocketEntryAssessment(
        feasible=True,
        reason="ok",
        entrance_angle_deg=entrance_angle,
        jaw_clearance_mm=float(jaw_clearance),
        required_clearance_mm=required,
        clearance_margin_mm=clearance_margin,
        mouth_crossing_mm=(float(crossing[0]), float(crossing[1])),
    )


def find_pocket_entry_path(
    final_leg_start_mm: PointLike,
    pocket_center_mm: PointLike,
    pocket_mouth_mm: PocketMouthLike | None,
    *,
    ball_radius_mm: float,
    safety_margin_mm: float,
    max_entrance_angle_deg: float,
    sample_count: int = 25,
) -> PocketEntryPath | None:
    """Search the usable mouth corridor instead of aiming at one fitted centre.

    The mouth segment is sampled only where a complete ball disc can fit.  For
    every sample, the incoming ray is extended beyond the mouth and checked
    against both physical jaw tips.  The explicit entrance-angle ceiling keeps
    a mathematically clear but practically tangential route off the overlay.
    Missing or undersized mouth geometry fails closed.
    """

    if pocket_mouth_mm is None or len(pocket_mouth_mm) != 2:
        return None
    start = _point(final_leg_start_mm)
    pocket = _point(pocket_center_mm)
    jaw_a = _point(pocket_mouth_mm[0])
    jaw_b = _point(pocket_mouth_mm[1])
    mouth = jaw_b - jaw_a
    mouth_span = float(np.linalg.norm(mouth))
    required = max(0.0, float(ball_radius_mm)) + max(0.0, float(safety_margin_mm))
    if mouth_span <= 2.0 * required + 1.0e-6:
        return None

    mouth_midpoint = 0.5 * (jaw_a + jaw_b)
    outward_axis = pocket - mouth_midpoint
    if float(np.linalg.norm(outward_axis)) <= 1.0e-6:
        return None
    outward = unit(outward_axis)
    maximum_angle = max(0.0, min(89.0, float(max_entrance_angle_deg)))
    lower = required / mouth_span
    upper = 1.0 - lower
    samples = max(5, int(sample_count))
    best: tuple[float, PocketEntryPath] | None = None
    for position in np.linspace(lower, upper, samples):
        crossing = jaw_a + mouth * float(position)
        direction = unit(crossing - start)
        if float(np.linalg.norm(direction)) <= 1.0e-6:
            continue
        # The object ball must move from the playing surface through the mouth,
        # not skim it backwards from the pocket side.
        if float(np.dot(direction, outward)) <= 0.05:
            continue
        entrance_angle = float(angle_deg(direction, outward))
        if entrance_angle > maximum_angle + 1.0e-6:
            continue
        extension = max(float(np.linalg.norm(pocket - crossing)), 2.0 * required)
        path_end = crossing + direction * extension
        jaw_clearance = min(
            _point_segment_distance(jaw_a, start, path_end),
            _point_segment_distance(jaw_b, start, path_end),
        )
        clearance_margin = float(jaw_clearance - required)
        if clearance_margin < -1.0e-6:
            continue
        assessment = PocketEntryAssessment(
            feasible=True,
            reason="ok",
            entrance_angle_deg=entrance_angle,
            jaw_clearance_mm=float(jaw_clearance),
            required_clearance_mm=required,
            clearance_margin_mm=clearance_margin,
            mouth_crossing_mm=(float(crossing[0]), float(crossing[1])),
        )
        path = PocketEntryPath(
            path_end_mm=(float(path_end[0]), float(path_end[1])),
            assessment=assessment,
        )
        center_offset = abs(float(position) - 0.5) * mouth_span
        quality = float(jaw_clearance - 0.06 * entrance_angle - 0.02 * center_offset)
        if best is None or quality > best[0]:
            best = (quality, path)
    return best[1] if best is not None else None


def _rejected(reason: str, required_clearance_mm: float) -> PocketEntryAssessment:
    return PocketEntryAssessment(
        feasible=False,
        reason=reason,
        entrance_angle_deg=float("inf"),
        jaw_clearance_mm=0.0,
        required_clearance_mm=float(required_clearance_mm),
        clearance_margin_mm=-float(required_clearance_mm),
    )


def _point(value: PointLike) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape((2,))


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _segment_intersection(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
) -> np.ndarray | None:
    direction_a = end_a - start_a
    direction_b = end_b - start_b
    denominator = _cross_2d(direction_a, direction_b)
    if abs(denominator) <= 1.0e-9:
        return None
    offset = start_b - start_a
    along_a = _cross_2d(offset, direction_b) / denominator
    along_b = _cross_2d(offset, direction_a) / denominator
    tolerance = 1.0e-7
    if not (-tolerance <= along_a <= 1.0 + tolerance):
        return None
    if not (-tolerance <= along_b <= 1.0 + tolerance):
        return None
    return start_a + direction_a * float(np.clip(along_a, 0.0, 1.0))


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    squared_length = float(np.dot(segment, segment))
    if squared_length <= 1.0e-12:
        return float(np.linalg.norm(point - start))
    position = float(np.dot(point - start, segment) / squared_length)
    closest = start + float(np.clip(position, 0.0, 1.0)) * segment
    return float(np.linalg.norm(point - closest))


__all__ = ["PocketEntryAssessment", "PocketEntryPath", "assess_pocket_entry", "find_pocket_entry_path"]
