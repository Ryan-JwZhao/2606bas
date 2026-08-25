from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


# Empirical floor: a 79-degree cut may still carry a target about 300 mm,
# while medium and long routes at the same angle remain below the score gate.
DIFFICULTY_VERSION = "cut_distance_transfer_v2"
MINIMUM_TRANSFER_FACTOR = 0.06


class RouteScoringProfile(str, Enum):
    RULE = "rule"
    TARGET = "target"


@dataclass(frozen=True)
class RouteDifficultyMetrics:
    cut_angle_deg: float
    max_cut_angle_deg: float
    cue_distance_mm: float
    object_distance_mm: float
    table_diagonal_mm: float
    clearance_norm: float
    pocket_entry_angle_deg: float
    rebounds: int = 0


@dataclass(frozen=True)
class RouteDifficulty:
    cut_penalty: float
    cut_distance_transfer_factor: float
    effective_object_distance_mm: float
    distance_penalty: float
    pocket_angle_penalty: float
    rebound_penalty: float
    score: float
    risk: float
    minimum_score: float
    accepted: bool


@dataclass(frozen=True)
class _ProfileWeights:
    base_score: float
    object_distance_weight: float
    cut_score_weight: float
    distance_score_weight: float
    pocket_angle_score_weight: float
    clearance_score_weight: float
    rebound_penalty_per_hit: float
    cut_risk_weight: float
    distance_risk_weight: float
    pocket_angle_risk_weight: float
    clearance_risk_weight: float


_PROFILE_WEIGHTS = {
    RouteScoringProfile.RULE: _ProfileWeights(
        base_score=2.0,
        object_distance_weight=0.55,
        cut_score_weight=1.30,
        distance_score_weight=0.90,
        pocket_angle_score_weight=0.35,
        clearance_score_weight=0.25,
        rebound_penalty_per_hit=0.0,
        cut_risk_weight=0.45,
        distance_risk_weight=0.35,
        pocket_angle_risk_weight=0.0,
        clearance_risk_weight=0.20,
    ),
    RouteScoringProfile.TARGET: _ProfileWeights(
        base_score=2.6,
        object_distance_weight=0.65,
        cut_score_weight=1.25,
        distance_score_weight=0.85,
        pocket_angle_score_weight=0.30,
        clearance_score_weight=0.25,
        rebound_penalty_per_hit=0.18,
        cut_risk_weight=0.45,
        distance_risk_weight=0.35,
        pocket_angle_risk_weight=0.15,
        clearance_risk_weight=0.20,
    ),
}


def evaluate_route_difficulty(
    metrics: RouteDifficultyMetrics,
    *,
    profile: RouteScoringProfile,
    minimum_score: float,
) -> RouteDifficulty:
    """Score geometric routes with collision-energy-aware object travel.

    For an equal-mass ball collision, the target-ball speed along its intended
    path scales approximately with cos(cut_angle).  Stopping distance scales
    with speed squared, so the transferable travel budget scales with cos².
    Dividing object travel by that factor couples long distance with thin cuts
    while preserving the previous score weights for straight shots.
    """

    weights = _PROFILE_WEIGHTS[RouteScoringProfile(profile)]
    cut_angle = _clamp(abs(float(metrics.cut_angle_deg)), 0.0, 90.0)
    max_cut_angle = max(1.0, float(metrics.max_cut_angle_deg))
    cut_penalty = (cut_angle / max_cut_angle) ** 1.8

    raw_transfer = math.cos(math.radians(cut_angle)) ** 2
    transfer_factor = _clamp(raw_transfer, MINIMUM_TRANSFER_FACTOR, 1.0)
    object_distance = max(0.0, float(metrics.object_distance_mm))
    effective_object_distance = object_distance / transfer_factor
    cue_distance = max(0.0, float(metrics.cue_distance_mm))
    table_diagonal = max(1.0, float(metrics.table_diagonal_mm))
    distance_penalty = (
        cue_distance + weights.object_distance_weight * effective_object_distance
    ) / table_diagonal

    clearance_norm = _clamp(float(metrics.clearance_norm), 0.0, 1.0)
    pocket_angle_penalty = _pocket_angle_penalty(metrics.pocket_entry_angle_deg)
    rebound_penalty = weights.rebound_penalty_per_hit * max(0, int(metrics.rebounds))
    score = float(
        weights.base_score
        - weights.cut_score_weight * cut_penalty
        - weights.distance_score_weight * distance_penalty
        - rebound_penalty
        - weights.pocket_angle_score_weight * pocket_angle_penalty
        + weights.clearance_score_weight * clearance_norm
    )
    risk = _clamp(
        weights.cut_risk_weight * cut_penalty
        + weights.distance_risk_weight * distance_penalty
        + rebound_penalty
        + weights.pocket_angle_risk_weight * pocket_angle_penalty
        + weights.clearance_risk_weight * (1.0 - clearance_norm),
        0.0,
        1.0,
    )
    threshold = float(minimum_score)
    return RouteDifficulty(
        cut_penalty=float(cut_penalty),
        cut_distance_transfer_factor=float(transfer_factor),
        effective_object_distance_mm=float(effective_object_distance),
        distance_penalty=float(distance_penalty),
        pocket_angle_penalty=float(pocket_angle_penalty),
        rebound_penalty=float(rebound_penalty),
        score=score,
        risk=float(risk),
        minimum_score=threshold,
        accepted=bool(math.isfinite(score) and score >= threshold),
    )


def _pocket_angle_penalty(entrance_angle_deg: float) -> float:
    return float(max(0.0, (float(entrance_angle_deg) - 20.0) / 70.0) ** 1.5)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


__all__ = [
    "DIFFICULTY_VERSION",
    "RouteDifficulty",
    "RouteDifficultyMetrics",
    "RouteScoringProfile",
    "evaluate_route_difficulty",
]
