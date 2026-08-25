from __future__ import annotations

import pytest

from bas.planning.shot_difficulty import (
    RouteDifficultyMetrics,
    RouteScoringProfile,
    evaluate_route_difficulty,
)


TABLE_DIAGONAL_MM = 2840.0


def _evaluate(*, cut_angle_deg: float, object_distance_mm: float, profile: RouteScoringProfile = RouteScoringProfile.RULE):
    return evaluate_route_difficulty(
        RouteDifficultyMetrics(
            cut_angle_deg=cut_angle_deg,
            max_cut_angle_deg=80.0,
            cue_distance_mm=50.0,
            object_distance_mm=object_distance_mm,
            table_diagonal_mm=TABLE_DIAGONAL_MM,
            clearance_norm=1.0,
            pocket_entry_angle_deg=0.0,
            rebounds=0,
        ),
        profile=profile,
        minimum_score=0.0,
    )


def test_straight_route_preserves_original_object_distance() -> None:
    result = _evaluate(cut_angle_deg=0.0, object_distance_mm=1121.15)

    assert result.cut_distance_transfer_factor == pytest.approx(1.0)
    assert result.effective_object_distance_mm == pytest.approx(1121.15)
    assert result.accepted


def test_distance_cost_grows_more_quickly_for_thinner_cuts() -> None:
    near_straight = _evaluate(cut_angle_deg=0.0, object_distance_mm=291.15)
    far_straight = _evaluate(cut_angle_deg=0.0, object_distance_mm=1121.15)
    near_thin = _evaluate(cut_angle_deg=75.0, object_distance_mm=291.15)
    far_thin = _evaluate(cut_angle_deg=75.0, object_distance_mm=1121.15)

    straight_drop = near_straight.score - far_straight.score
    thin_drop = near_thin.score - far_thin.score

    assert thin_drop > 10.0 * straight_drop
    assert near_thin.accepted
    assert not far_thin.accepted


def test_short_grazing_route_is_barely_accepted_while_medium_route_is_rejected() -> None:
    short = _evaluate(cut_angle_deg=79.0, object_distance_mm=300.0)
    medium = _evaluate(cut_angle_deg=79.0, object_distance_mm=600.0)

    assert 0.0 <= short.score <= 0.20
    assert short.accepted
    assert medium.score < 0.0
    assert not medium.accepted


@pytest.mark.parametrize("profile", [RouteScoringProfile.RULE, RouteScoringProfile.TARGET])
def test_far_grazing_route_fails_minimum_score(profile: RouteScoringProfile) -> None:
    result = _evaluate(cut_angle_deg=79.0, object_distance_mm=1121.15, profile=profile)

    assert result.cut_distance_transfer_factor == pytest.approx(0.06)
    assert result.effective_object_distance_mm > 18_000.0
    assert result.score < result.minimum_score
    assert not result.accepted
