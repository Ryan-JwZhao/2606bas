from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from bas.planning.route_stability import (
    BallPositionMeasurement,
    PlanningPositionStabilizer,
    RouteTopologyContinuity,
)
from bas.route_geometry import cue_alignment_start
from bas.schemas import ShotCandidate


# Frames 27-62 from no_line_video_20260820_153411.mp4.  The balls are physically
# stationary; the object-ball geometry fallback creates the large route swing.
_RECORDED_STATIONARY_TRACE = (
    ((657.25702, 188.63057), (0.03357, 0.00732), (796.73138, 217.18895), (-1.58386, 0.58395)),
    ((657.40167, 188.56683), (0.76172, -0.32104), (796.56409, 217.23950), (-1.88477, 0.63782)),
    ((657.51831, 188.49840), (1.09070, -0.55862), (796.13635, 217.28923), (-3.41187, 0.66910)),
    ((657.65186, 188.46574), (1.39160, -0.52994), (796.04004, 217.29584), (-2.70935, 0.46875)),
    ((657.70837, 188.40817), (1.19385, -0.63873), (796.00836, 217.32576), (-1.92322, 0.45731)),
    ((657.75232, 188.37471), (1.00098, -0.58624), (795.96539, 217.33971), (-1.47034, 0.36865)),
    ((657.78308, 188.35132), (0.80750, -0.50079), (796.00195, 217.34558), (-0.76843, 0.26962)),
    ((657.81970, 188.34033), (0.71167, -0.38147), (795.83899, 217.37372), (-1.33362, 0.31921)),
    ((657.83563, 188.36148), (0.54443, -0.13992), (796.77924, 217.11841), (3.94043, -1.09756)),
    ((657.65521, 188.46495), (-0.56885, 0.43793), (796.97723, 217.17989), (3.57300, -0.39917)),
    ((657.56238, 188.53731), (-0.84412, 0.65460), (796.92767, 217.23563), (2.06970, 0.02548)),
    ((657.49738, 188.58795), (-0.88135, 0.68436), (797.36816, 217.19917), (3.59741, -0.16968)),
    ((657.42267, 188.62860), (-0.95398, 0.65262), (797.69531, 217.17537), (4.01001, -0.23193)),
    ((657.32849, 188.62520), (-1.10168, 0.40695), (797.75140, 217.10558), (2.89307, -0.50766)),
    ((657.29010, 188.62471), (-0.91248, 0.26215), (797.71863, 217.04166), (1.71265, -0.65674)),
    ((657.26318, 188.62437), (-0.73059, 0.16846), (797.72498, 216.99600), (1.14624, -0.66010)),
    ((657.23755, 188.62480), (-0.60547, 0.11185), (797.48065, 217.01118), (-0.50415, -0.35141)),
    ((657.22552, 188.61482), (-0.45593, 0.02151), (797.11926, 217.08096), (-2.17468, 0.12833)),
    ((657.21771, 188.60751), (-0.33630, -0.02319), (797.11353, 217.07509), (-1.44287, 0.05325)),
    ((657.21155, 188.60272), (-0.24963, -0.03983), (797.21155, 217.05406), (-0.43701, -0.07278)),
    ((657.20728, 188.59938), (-0.18372, -0.04303), (797.24274, 217.04816), (-0.12390, -0.07751)),
    ((657.20428, 188.59702), (-0.13489, -0.03983), (797.01776, 217.05853), (-1.23047, 0.00275)),
    ((657.19537, 188.61134), (-0.13367, 0.04730), (796.89270, 217.09337), (-1.43982, 0.17990)),
    ((657.19012, 188.61569), (-0.11353, 0.05310), (796.80884, 217.10703), (-1.36414, 0.18661)),
    ((657.18646, 188.61873), (-0.09277, 0.05005), (796.80096, 217.09326), (-0.92712, 0.05096)),
    ((657.19678, 188.64586), (-0.00793, 0.17120), (797.16718, 217.12941), (1.26953, 0.21805)),
    ((657.19501, 188.64972), (-0.01404, 0.13107), (797.11200, 217.10928), (0.54382, 0.03860)),
    ((657.19373, 188.65242), (-0.01526, 0.09903), (796.95471, 217.16309), (-0.45105, 0.30029)),
    ((657.19287, 188.65431), (-0.01465, 0.07401), (796.83124, 217.13040), (-0.92468, 0.02808)),
    ((657.20123, 188.67076), (0.03357, 0.13214), (796.90094, 217.08539), (-0.24414, -0.21179)),
    ((657.19812, 188.66716), (0.00549, 0.06744), (796.85699, 217.03949), (-0.38391, -0.37247)),
    ((657.20496, 188.67976), (0.03845, 0.10818), (796.93085, 217.00500), (0.12817, -0.41840)),
    ((657.20972, 188.68857), (0.04944, 0.11520), (797.34393, 216.97913), (2.19482, -0.40421)),
    ((657.21301, 188.69472), (0.04883, 0.10651), (797.28040, 216.95749), (1.10229, -0.37323)),
    ((657.21539, 188.69904), (0.04333, 0.09125), (798.76184, 216.84149), (8.28918, -0.83572)),
    ((657.21698, 188.70206), (0.03662, 0.07462), (799.07599, 216.90744), (6.99341, -0.20584)),
)


def _config(**overrides):
    values = {
        "route_stability_enabled": True,
        "route_stability_stationary_tau_ms": 3000.0,
        "route_stability_motion_tau_ms": 60.0,
        "route_stability_quiet_speed_mm_s": 12.0,
        "route_stability_fast_speed_mm_s": 160.0,
        "route_stability_deadband_mm": 2.5,
        "route_stability_micro_gain": 0.08,
        "route_stability_response_distance_mm": 7.0,
        "route_stability_reset_distance_mm": 80.0,
        "route_stability_reset_gap_ms": 500.0,
        "route_stability_prediction_ms": 70.0,
        "route_stability_prediction_max_mm": 12.0,
        "route_topology_continuity_enabled": True,
        "route_topology_switch_confirm_ms": 220.0,
        "route_topology_switch_score_delta": 0.18,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _max_pairwise(points) -> float:
    values = np.asarray(points, dtype=np.float64)
    return float(np.max(np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)))


def _guide_start(cue, target) -> np.ndarray:
    pocket = np.asarray((1313.610107421875, -70.65151977539062), dtype=np.float32)
    obj = np.asarray(target, dtype=np.float32)
    obj_dir = (pocket - obj) / np.linalg.norm(pocket - obj)
    ghost = obj - obj_dir * 57.15
    table = np.asarray([(0.0, 0.0), (2540.0, 0.0), (2540.0, 1270.0), (0.0, 1270.0)], dtype=np.float32)
    return cue_alignment_start(
        np.asarray(cue, dtype=np.float32),
        ghost,
        table,
        table_width_mm=2540.0,
        table_height_mm=1270.0,
    )


def test_recorded_stationary_trace_suppresses_amplified_guide_jitter() -> None:
    stabilizer = PlanningPositionStabilizer(_config())
    raw_ends = []
    stable_ends = []
    period_ns = 68_469_702

    for index, (cue, cue_velocity, target, target_velocity) in enumerate(_RECORDED_STATIONARY_TRACE):
        raw_ends.append(_guide_start(cue, target))
        shown = stabilizer.update(
            index * period_ns,
            [
                BallPositionMeasurement(1, cue, cue_velocity, quality=0.89, uncertainty_mm=0.5),
                BallPositionMeasurement(2, target, target_velocity, quality=0.79, uncertainty_mm=1.5),
            ],
            phase="STABLE_IDLE",
        )
        stable_ends.append(_guide_start(shown[1], shown[2]))

    assert _max_pairwise(raw_ends) > 20.0
    assert _max_pairwise(stable_ends) < 2.0


def test_slow_and_fast_motion_keep_generating_fresh_positions_without_freeze() -> None:
    period_ns = 68_469_702
    for step_mm, speed_mm_s, max_lag_mm in ((1.0, 14.605, 5.0), (10.0, 146.05, 12.0)):
        stabilizer = PlanningPositionStabilizer(_config())
        outputs = []
        for index in range(40):
            measured = (100.0 + index * step_mm, 200.0)
            shown = stabilizer.update(
                index * period_ns,
                [BallPositionMeasurement(1, measured, (speed_mm_s, 0.0))],
                phase="SHOT_ACTIVE",
            )[1]
            outputs.append(shown)

        x_values = np.asarray(outputs)[:, 0]
        assert np.all(np.diff(x_values) > 0.0)
        assert (100.0 + 39 * step_mm) - float(x_values[-1]) < max_lag_mm


def test_large_motion_resets_immediately_and_stop_does_not_overshoot() -> None:
    stabilizer = PlanningPositionStabilizer(_config())
    period_ns = 68_469_702
    stabilizer.update(0, [BallPositionMeasurement(1, (100.0, 200.0), (0.0, 0.0))], phase="STABLE_IDLE")
    jumped = stabilizer.update(
        period_ns,
        [BallPositionMeasurement(1, (220.0, 200.0), (800.0, 0.0))],
        phase="SHOT_ACTIVE",
    )[1]
    np.testing.assert_allclose(jumped, (220.0, 200.0), atol=1e-6)

    stopped = []
    for index in range(2, 12):
        shown = stabilizer.update(
            index * period_ns,
            [BallPositionMeasurement(1, (220.0, 200.0), (0.0, 0.0))],
            phase="STABLE_IDLE",
        )[1]
        stopped.append(shown)
    assert max(float(point[0]) for point in stopped) <= 220.0 + 1e-6


def _candidate(target: int, pocket: int, score: float, cue_x: float, *, rails=()) -> ShotCandidate:
    explanation = {"target_shot_rebounds": len(rails), "target_shot_rails": list(rails)}
    return ShotCandidate(
        candidate_id=f"t{target}_p{pocket}_{cue_x}",
        cue_track_id=1,
        target_track_id=target,
        target_group="solid",
        pocket_index=pocket,
        cue_ball=(cue_x, 100.0),
        object_ball=(200.0, 100.0),
        ghost_ball=(180.0, 100.0),
        pocket_point=(300.0, 0.0),
        aim_line=[(cue_x, 100.0), (180.0, 100.0)],
        object_line=[(200.0, 100.0), (300.0, 0.0)],
        cut_angle_deg=20.0,
        cue_distance_mm=80.0,
        object_distance_mm=100.0,
        score=score,
        risk=0.2,
        explanation=explanation,
    )


def test_topology_hold_uses_current_frame_geometry_and_never_stale_snapshot() -> None:
    continuity = RouteTopologyContinuity(_config())
    first_a = _candidate(2, 1, 1.00, 100.0)
    first_b = _candidate(3, 4, 0.95, 100.0)
    assert continuity.select([first_a, first_b], ts_cam_ns=0, shot_mode="rule")[0] is first_a

    current_b = _candidate(3, 4, 1.05, 108.0)
    current_a = _candidate(2, 1, 1.00, 108.0)
    held = continuity.select([current_b, current_a], ts_cam_ns=100_000_000, shot_mode="rule")
    assert held[0] is current_a
    assert held[0].cue_ball == (108.0, 100.0)

    # The previous topology is no longer physically valid because it is absent
    # from this frame's candidates.  Switch immediately; never reuse old geometry.
    replacement = _candidate(3, 4, 1.05, 116.0)
    switched = continuity.select([replacement], ts_cam_ns=150_000_000, shot_mode="rule")
    assert switched[0] is replacement


def test_topology_switches_after_sustained_material_score_gain() -> None:
    continuity = RouteTopologyContinuity(_config(route_topology_switch_confirm_ms=200.0))
    original = _candidate(2, 1, 1.00, 100.0)
    continuity.select([original], ts_cam_ns=0, shot_mode="rule")

    better_1 = _candidate(3, 4, 1.30, 102.0)
    current_1 = _candidate(2, 1, 1.00, 102.0)
    pending = continuity.select([better_1, current_1], ts_cam_ns=100_000_000, shot_mode="rule")
    assert pending[0] is current_1

    better_2 = _candidate(3, 4, 1.31, 104.0)
    current_2 = _candidate(2, 1, 1.00, 104.0)
    committed = continuity.select([better_2, current_2], ts_cam_ns=310_000_000, shot_mode="rule")
    assert committed[0] is better_2


def test_timestamp_rewind_resets_history() -> None:
    stabilizer = PlanningPositionStabilizer(_config())
    stabilizer.update(1_000_000_000, [BallPositionMeasurement(1, (100.0, 100.0), (0.0, 0.0))])
    rewound = stabilizer.update(10, [BallPositionMeasurement(1, (140.0, 130.0), (0.0, 0.0))])[1]
    np.testing.assert_allclose(rewound, (140.0, 130.0), atol=1e-6)


def test_filter_cost_is_linear_in_visible_ball_count() -> None:
    stabilizer = PlanningPositionStabilizer(_config())
    measurements = [
        BallPositionMeasurement(index, (float(index), float(index)), (0.0, 0.0))
        for index in range(16)
    ]
    shown = stabilizer.update(0, measurements)
    assert len(shown) == len(measurements)
    assert all(math.isfinite(float(value)) for point in shown.values() for value in point)
