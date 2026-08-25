from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pocket_geometry import PocketApproachProbe


@dataclass(frozen=True)
class PocketTrajectoryLimits:
    candidate_depth_mm: float
    history_depth_mm: float
    handoff_ms: int
    min_speed_mm_s: float
    max_speed_mm_s: float
    ball_diameter_mm: float


@dataclass(frozen=True)
class PocketEntryAssessment:
    pocket_index: int
    reason: str
    source_track_id: int
    speed_mm_s: float
    depth_mm: float
    lateral_mm: float
    projected_lateral_mm: float


def assess_reported_entry(
    probe: PocketApproachProbe,
    *,
    track_id: int,
    limits: PocketTrajectoryLimits,
) -> Optional[PocketEntryAssessment]:
    """Accept a same-track approach whose reported velocity crosses the mouth."""

    if not _current_position_is_eligible(probe, limits):
        return None
    speed = float(probe.pocketward_speed_mm_s)
    if speed < limits.min_speed_mm_s or speed > limits.max_speed_mm_s:
        return None
    projected_lateral = _project_lateral_at_mouth(
        depth_mm=probe.depth_mm,
        lateral_mm=probe.signed_lateral_mm,
        pocketward_speed_mm_s=speed,
        tangential_speed_mm_s=probe.tangential_speed_mm_s,
    )
    if abs(projected_lateral) > capture_half_width_mm(probe, limits):
        return None
    return PocketEntryAssessment(
        pocket_index=int(probe.pocket_index),
        reason="projected_entry_same_track",
        source_track_id=int(track_id),
        speed_mm_s=speed,
        depth_mm=float(probe.depth_mm),
        lateral_mm=float(probe.signed_lateral_mm),
        projected_lateral_mm=float(projected_lateral),
    )


def assess_track_handoff(
    source: PocketApproachProbe,
    current: PocketApproachProbe,
    *,
    source_track_id: int,
    elapsed_ms: float,
    limits: PocketTrajectoryLimits,
) -> Optional[PocketEntryAssessment]:
    """Recover pocketward velocity when a detector changes IDs near a pocket."""

    if (
        source.pocket_index is None
        or current.pocket_index is None
        or source.pocket_index != current.pocket_index
        or not source.geometry_valid
        or elapsed_ms <= 0.0
        or elapsed_ms > limits.handoff_ms
        or source.depth_mm < -limits.history_depth_mm
    ):
        return None

    source_gate = capture_half_width_mm(source, limits) + max(0.0, -float(source.depth_mm)) * 0.45
    if abs(float(source.signed_lateral_mm)) > source_gate:
        return None

    elapsed_s = elapsed_ms / 1000.0
    depth_advance = float(current.depth_mm - source.depth_mm)
    lateral_advance = float(current.signed_lateral_mm - source.signed_lateral_mm)
    pocketward_speed = depth_advance / elapsed_s
    path_speed = float(np.hypot(depth_advance, lateral_advance) / elapsed_s)
    if (
        depth_advance < max(8.0, limits.ball_diameter_mm * 0.18)
        or pocketward_speed < limits.min_speed_mm_s
        or path_speed > limits.max_speed_mm_s
    ):
        return None

    if not _current_position_is_eligible(current, limits, pocketward_speed=pocketward_speed):
        return None

    tangential_speed = lateral_advance / elapsed_s
    projected_lateral = _project_lateral_at_mouth(
        depth_mm=current.depth_mm,
        lateral_mm=current.signed_lateral_mm,
        pocketward_speed_mm_s=pocketward_speed,
        tangential_speed_mm_s=tangential_speed,
    )
    if abs(projected_lateral) > capture_half_width_mm(current, limits):
        return None
    return PocketEntryAssessment(
        pocket_index=int(current.pocket_index),
        reason="projected_entry_track_handoff",
        source_track_id=int(source_track_id),
        speed_mm_s=float(path_speed),
        depth_mm=float(current.depth_mm),
        lateral_mm=float(current.signed_lateral_mm),
        projected_lateral_mm=float(projected_lateral),
    )


def projected_entry_has_reversed(
    probe: PocketApproachProbe,
    *,
    pocket_index: int,
    best_depth_mm: float,
    limits: PocketTrajectoryLimits,
) -> bool:
    """Return true once a visible candidate demonstrably comes back to the table."""

    if not probe.geometry_valid or probe.pocket_index != pocket_index:
        return True
    reversal_distance = max(12.0, limits.ball_diameter_mm * 0.25)
    if float(probe.depth_mm) < float(best_depth_mm) - reversal_distance:
        return True
    if float(probe.pocketward_speed_mm_s) <= -max(limits.min_speed_mm_s, 2.0 * reversal_distance):
        return True
    hysteresis = max(12.0, limits.ball_diameter_mm * 0.25)
    predictive_lead = min(
        limits.ball_diameter_mm * 3.0,
        max(0.0, float(probe.pocketward_speed_mm_s)) * 0.18,
    )
    if float(probe.depth_mm) < -limits.candidate_depth_mm - predictive_lead - hysteresis:
        return True
    if abs(float(probe.signed_lateral_mm)) > capture_half_width_mm(probe, limits) + hysteresis:
        return True
    return False


def _current_position_is_eligible(
    probe: PocketApproachProbe,
    limits: PocketTrajectoryLimits,
    *,
    pocketward_speed: float | None = None,
) -> bool:
    if not probe.geometry_valid or probe.pocket_index is None:
        return False
    speed = max(0.0, float(probe.pocketward_speed_mm_s if pocketward_speed is None else pocketward_speed))
    predictive_lead = min(limits.ball_diameter_mm * 3.0, speed * 0.18)
    if float(probe.depth_mm) < -limits.candidate_depth_mm - predictive_lead:
        return False
    lateral_allowance = capture_half_width_mm(probe, limits) + max(0.0, -float(probe.depth_mm)) * 0.18
    return abs(float(probe.signed_lateral_mm)) <= lateral_allowance


def capture_half_width_mm(probe: PocketApproachProbe, limits: PocketTrajectoryLimits) -> float:
    return max(1.0, float(probe.mouth_half_width_mm) + limits.ball_diameter_mm * 0.5)


def _project_lateral_at_mouth(
    *,
    depth_mm: float,
    lateral_mm: float,
    pocketward_speed_mm_s: float,
    tangential_speed_mm_s: float,
) -> float:
    if depth_mm >= 0.0 or pocketward_speed_mm_s <= 1e-6:
        return float(lateral_mm)
    time_to_mouth_s = min(0.5, max(0.0, -float(depth_mm) / float(pocketward_speed_mm_s)))
    return float(lateral_mm + tangential_speed_mm_s * time_to_mouth_s)


__all__ = [
    "PocketEntryAssessment",
    "PocketTrajectoryLimits",
    "assess_reported_entry",
    "assess_track_handoff",
    "capture_half_width_mm",
    "projected_entry_has_reversed",
]
