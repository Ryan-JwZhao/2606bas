from __future__ import annotations

import math

import numpy as np
import pytest

from bas.planning.pocket_clearance import assess_pocket_entry


BALL_RADIUS_MM = 28.575
TABLE_CENTER_MM = np.asarray([1270.0, 635.0], dtype=np.float64)

# The six fitted pocket centres and physical jaw endpoints from
# no_line_video_20260820_153411.mp4.  These values are deliberately table-mm
# geometry rather than image pixels so the regression is camera-independent.
RECORDED_POCKETS = [
    ((1.504668116569519, -85.4677734375), ((-0.16607864201068878, 1.1135486364364624), (82.14236450195312, -51.948265075683594))),
    ((1313.610107421875, -70.65151977539062), ((1255.9141845703125, -54.37510681152344), (1369.5958251953125, -43.73866271972656))),
    ((2611.342041015625, -12.19668197631836), ((2537.654541015625, 1.3274861574172974), (2608.055419921875, 61.4681282043457))),
    ((2526.6064453125, 1339.6461181640625), ((2538.126953125, 1267.4754638671875), (2456.595947265625, 1320.7725830078125))),
    ((1230.3128662109375, 1336.8466796875), ((1286.3345947265625, 1320.020751953125), (1174.764404296875, 1319.115966796875))),
    ((-64.17188262939453, 1277.83544921875), ((1.3713555335998535, 1269.2698974609375), (-68.50920104980469, 1209.0924072265625))),
]


def _start_for_entrance_angle(
    pocket: tuple[float, float],
    mouth: tuple[tuple[float, float], tuple[float, float]],
    angle_deg: float,
    distance_mm: float = 600.0,
) -> np.ndarray:
    pocket_arr = np.asarray(pocket, dtype=np.float64)
    outward = pocket_arr - TABLE_CENTER_MM
    outward /= np.linalg.norm(outward)
    radians = math.radians(angle_deg)
    rotated = np.asarray(
        [
            outward[0] * math.cos(radians) - outward[1] * math.sin(radians),
            outward[0] * math.sin(radians) + outward[1] * math.cos(radians),
        ],
        dtype=np.float64,
    )
    return pocket_arr - rotated * distance_mm


@pytest.mark.parametrize(
    ("pocket_index", "safe_angle_deg"),
    [(0, 27.0), (1, 0.0), (2, -23.0), (3, 28.0), (4, 0.0), (5, -25.0)],
)
def test_recorded_six_pockets_accept_a_route_with_real_ball_clearance(
    pocket_index: int,
    safe_angle_deg: float,
) -> None:
    pocket, mouth = RECORDED_POCKETS[pocket_index]
    assessment = assess_pocket_entry(
        _start_for_entrance_angle(pocket, mouth, safe_angle_deg),
        pocket,
        mouth,
        ball_radius_mm=BALL_RADIUS_MM,
        safety_margin_mm=6.0,
    )

    assert assessment.feasible, assessment.reason
    assert assessment.clearance_margin_mm >= 0.0
    assert assessment.required_clearance_mm == pytest.approx(34.575)


@pytest.mark.parametrize(
    ("pocket_index", "blocked_angle_deg"),
    [(0, 0.0), (1, 55.0), (2, 0.0), (3, 0.0), (4, 55.0), (5, 0.0)],
)
def test_recorded_six_pockets_reject_routes_that_only_a_point_ball_could_take(
    pocket_index: int,
    blocked_angle_deg: float,
) -> None:
    pocket, mouth = RECORDED_POCKETS[pocket_index]
    assessment = assess_pocket_entry(
        _start_for_entrance_angle(pocket, mouth, blocked_angle_deg),
        pocket,
        mouth,
        ball_radius_mm=BALL_RADIUS_MM,
        safety_margin_mm=6.0,
    )

    assert not assessment.feasible
    assert assessment.reason in {"jaw_clearance", "misses_mouth"}


def test_latest_no_line_green_ball_route_is_rejected_by_near_jaw_clearance() -> None:
    pocket, mouth = RECORDED_POCKETS[1]
    assessment = assess_pocket_entry(
        (797.4368896484375, 216.83296966552734),
        pocket,
        mouth,
        ball_radius_mm=BALL_RADIUS_MM,
        safety_margin_mm=0.0,
    )

    assert not assessment.feasible
    assert assessment.reason == "jaw_clearance"
    # Entry scoring uses the measured mouth normal, not the old reversed
    # pocket-to-table-centre direction.
    assert assessment.entrance_angle_deg == pytest.approx(58.6166, abs=0.01)
    assert assessment.clearance_margin_mm == pytest.approx(-14.7213, abs=0.02)


def test_missing_or_degenerate_mouth_geometry_fails_closed() -> None:
    missing = assess_pocket_entry(
        (500.0, 250.0),
        (500.0, -50.0),
        None,
        ball_radius_mm=BALL_RADIUS_MM,
        safety_margin_mm=0.0,
    )
    degenerate = assess_pocket_entry(
        (500.0, 250.0),
        (500.0, -50.0),
        ((500.0, 0.0), (500.0, 0.0)),
        ball_radius_mm=BALL_RADIUS_MM,
        safety_margin_mm=0.0,
    )

    assert not missing.feasible
    assert missing.reason == "missing_mouth"
    assert not degenerate.feasible
    assert degenerate.reason == "degenerate_mouth"
