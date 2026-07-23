from __future__ import annotations

import pytest

from bas.config import TrainingConfig
from bas.schemas import PocketVisualObservation, PocketVisualObservationFrame, TrackObservation, TracksFrame
from bas.training import CueBallStopObserver, TrainingSession, get_training_scenario, list_training_scenarios


TABLE_POLYGON = [(0.0, 0.0), (2540.0, 0.0), (2540.0, 1270.0), (0.0, 1270.0)]
POCKET = (2540.0, 635.0)


def _track(
    number: int,
    position: tuple[float, float],
    *,
    velocity_mm_s: tuple[float, float] = (0.0, 0.0),
) -> TrackObservation:
    x, y = position
    return TrackObservation(
        track_id=number,
        bbox=(x - 20.0, y - 20.0, x + 20.0, y + 20.0),
        center_px=(x, y),
        center_mm=(x, y),
        radius_px=20.0,
        radius_mm=28.575,
        cls_name=str(number),
        group="cue" if number == 0 else "solid",
        confidence=0.99,
        velocity_mm_s=velocity_mm_s,
        visibility="visible",
    )


def _frame(
    frame_id: int,
    ts_ns: int,
    positions: dict[int, tuple[float, float]],
    *,
    cue_velocity_mm_s: tuple[float, float] = (0.0, 0.0),
) -> TracksFrame:
    return TracksFrame(
        frame_id=frame_id,
        ts_cam_ns=ts_ns,
        tracks=[
            _track(
                number,
                position,
                velocity_mm_s=(cue_velocity_mm_s if number == 0 else (0.0, 0.0)),
            )
            for number, position in positions.items()
        ],
    )


def _session(scenario_id: str, positions: dict[int, tuple[float, float]]) -> TrainingSession:
    session = TrainingSession(TrainingConfig(scenario_id=scenario_id))
    session.set_table_context(
        inner_polygon_mm=TABLE_POLYGON,
        ball_center_reachable_polygon_mm=TABLE_POLYGON,
        pockets_mm=[POCKET],
        ball_diameter_mm=57.15,
    )
    session.update(_frame(1, 100_000_000, positions))
    started, state = session.start()
    assert started is True, state.message
    return session


def _confirm_pot_after_cue_motion(
    session: TrainingSession,
    *,
    ball: int,
    first_frame_id: int,
    start_ts_ns: int,
    moving_positions: dict[int, tuple[float, float]],
    final_positions: dict[int, tuple[float, float]],
):
    session.update(
        _frame(
            first_frame_id,
            start_ts_ns,
            moving_positions,
            cue_velocity_mm_s=(220.0, 0.0),
        )
    )
    crossing_frame = first_frame_id + 1
    crossing_ts = start_ts_ns + 200_000_000
    session.update(
        _frame(crossing_frame, crossing_ts, final_positions),
        PocketVisualObservationFrame(
            frame_id=crossing_frame,
            ts_cam_ns=crossing_ts,
            observations=[
                PocketVisualObservation(
                    pocket_index=0,
                    inward_crossing=True,
                    group="solid",
                    confidence=0.99,
                    associated_track_ids=[ball],
                    evidence_sources=["foreground_motion", "ball_sized_motion"],
                    motion_score=0.9,
                    foreground_score=0.9,
                    foreground_depth_diameters=0.5,
                )
            ],
        ),
    )
    resolved_frame = first_frame_id + 2
    resolved_ts = crossing_ts + 1_300_000_000
    session.update(
        _frame(resolved_frame, resolved_ts, final_positions),
        PocketVisualObservationFrame(
            frame_id=resolved_frame,
            ts_cam_ns=resolved_ts,
            observations=[PocketVisualObservation(pocket_index=0, clear=True)],
        ),
    )
    stopped_frame = resolved_frame + 1
    return session.update(
        _frame(stopped_frame, resolved_ts + 700_000_000, final_positions),
    )


def test_cue_ball_stop_observer_requires_continuous_visible_stability() -> None:
    observer = CueBallStopObserver(still_speed_mm_s=8.0, stable_hold_ms=600)

    assert observer.update(_frame(1, 0, {0: (100.0, 100.0)}, cue_velocity_mm_s=(50.0, 0.0))) is None
    assert observer.status == "moving"
    assert observer.update(_frame(2, 100_000_000, {0: (150.0, 100.0)})) is None
    assert observer.update(_frame(3, 500_000_000, {0: (153.0, 101.0)})) is None
    stopped = observer.update(_frame(4, 750_000_000, {0: (152.0, 99.0)}))

    assert stopped is not None
    assert stopped.position_mm == pytest.approx((151.67, 100.0), abs=0.1)
    assert observer.status == "stopped"

    assert observer.update(_frame(5, 800_000_000, {})) is None
    assert observer.status == "missing"


def test_cue_control_catalog_is_third_and_contains_four_projects() -> None:
    scenarios = list_training_scenarios()
    groups = list(dict.fromkeys(scenario.group for scenario in scenarios))
    cue_control = [scenario for scenario in scenarios if scenario.group == "cue_control"]

    assert groups == ["beginner", "entry", "cue_control", "advanced"]
    assert [scenario.scenario_id for scenario in cue_control] == [
        "CUE_CONTROL_TWO_BALL_LINK",
        "CUE_CONTROL_STOP_ZONE",
        "CUE_CONTROL_FOLLOW_ZONE",
        "CUE_CONTROL_DRAW_ZONE",
    ]
    assert all(scenario.require_cue_ball_settle_after_pot for scenario in cue_control)
    assert all(scenario.group_title == "白球控制训练模式" for scenario in cue_control)


@pytest.mark.parametrize(
    ("scenario_id", "final_cue_position"),
    [
        ("CUE_CONTROL_STOP_ZONE", (600.0, 635.0)),
        ("CUE_CONTROL_FOLLOW_ZONE", (1230.0, 635.0)),
        ("CUE_CONTROL_DRAW_ZONE", (770.0, 635.0)),
    ],
)
def test_single_ball_cue_control_passes_only_after_cue_stops_in_goal(
    scenario_id: str,
    final_cue_position: tuple[float, float],
) -> None:
    initial = {0: (600.0, 635.0), 1: (1000.0, 635.0)}
    session = _session(scenario_id, initial)

    state = _confirm_pot_after_cue_motion(
        session,
        ball=1,
        first_frame_id=2,
        start_ts_ns=200_000_000,
        moving_positions={0: (800.0, 635.0), 1: initial[1]},
        final_positions={0: final_cue_position},
    )

    assert state.phase == "passed"
    assert state.cue_ball_status == "stopped"
    assert state.cue_ball_goal_result == "passed"
    assert state.cue_ball_position_mm == pytest.approx(final_cue_position, abs=0.1)
    assert any(event.name == "TRAINING_CUE_BALL_STOPPED" for event in state.events)
    assert any(event.name == "TRAINING_CUE_BALL_CONTROL_PASSED" for event in state.events)


def test_cue_control_fails_when_stopped_cue_ball_is_outside_goal() -> None:
    initial = {0: (600.0, 635.0), 1: (1000.0, 635.0)}
    session = _session("CUE_CONTROL_STOP_ZONE", initial)

    state = _confirm_pot_after_cue_motion(
        session,
        ball=1,
        first_frame_id=2,
        start_ts_ns=200_000_000,
        moving_positions={0: (800.0, 635.0), 1: initial[1]},
        final_positions={0: (1300.0, 635.0)},
    )

    assert state.phase == "failed"
    assert state.failure_reason == "cue_ball_outside_target"
    assert state.cue_ball_goal_result == "failed"
    assert "未进入目标区域" in state.message


def test_two_ball_link_waits_for_each_stop_and_advances_to_two() -> None:
    initial = {
        0: (508.0, 864.0),
        1: (1067.0, 533.0),
        2: (1981.0, 381.0),
    }
    session = _session("CUE_CONTROL_TWO_BALL_LINK", initial)
    goal_center = session.state.cue_ball_goal_center_mm
    assert goal_center is not None

    after_one = _confirm_pot_after_cue_motion(
        session,
        ball=1,
        first_frame_id=2,
        start_ts_ns=200_000_000,
        moving_positions={0: (900.0, 650.0), 1: initial[1], 2: initial[2]},
        final_positions={0: goal_center, 2: initial[2]},
    )
    assert after_one.phase == "running"
    assert after_one.expected_numbers == [2]
    assert after_one.cue_ball_goal_result == "passed"

    after_two = _confirm_pot_after_cue_motion(
        session,
        ball=2,
        first_frame_id=6,
        start_ts_ns=2_600_000_000,
        moving_positions={0: (1500.0, 500.0), 2: initial[2]},
        final_positions={0: (1600.0, 500.0)},
    )
    assert after_two.phase == "passed"
    assert after_two.potted_numbers == [1, 2]
    assert after_two.cue_ball_status == "stopped"


def test_cue_control_scenarios_keep_strict_ball_order_rules() -> None:
    scenario = get_training_scenario("CUE_CONTROL_TWO_BALL_LINK")
    assert scenario.stages == ((1,), (2,))
    assert scenario.rule_set_id == "strict"
