from __future__ import annotations

import pytest

from bas.config import TrainingConfig
from bas.schemas import PocketVisualObservation, PocketVisualObservationFrame, TrackObservation, TracksFrame
from bas.training import (
    TrainingSession,
    get_training_rule_set,
    get_training_scenario,
    list_training_scenarios,
    validate_scenario_setup,
)


TABLE_POLYGON = [(50.0, 50.0), (950.0, 50.0), (950.0, 450.0), (50.0, 450.0)]
POCKETS = [(0.0, 0.0), (500.0, 0.0), (1_000.0, 0.0), (0.0, 500.0), (500.0, 500.0), (1_000.0, 500.0), (1_000.0, 250.0)]


def _track(number: int, x: float, y: float) -> TrackObservation:
    return TrackObservation(
        track_id=number,
        bbox=(x - 10.0, y - 10.0, x + 10.0, y + 10.0),
        center_px=(x, y),
        center_mm=(x, y),
        radius_px=10.0,
        radius_mm=28.575,
        cls_name=str(number),
        group="cue" if number == 0 else "solid" if number <= 7 else "black" if number == 8 else "stripe",
        confidence=0.98,
        visibility="visible",
    )


def _frame(frame_id: int, positions: dict[int, tuple[float, float]]) -> TracksFrame:
    return TracksFrame(
        frame_id=frame_id,
        ts_cam_ns=frame_id * 100_000_000,
        tracks=[_track(number, *point) for number, point in positions.items()],
    )


def _validate(
    scenario_id: str,
    positions: dict[int, tuple[float, float]],
) -> tuple[bool, str]:
    return validate_scenario_setup(
        get_training_scenario(scenario_id),
        _frame(1, positions).tracks,
        ball_center_reachable_polygon_mm=TABLE_POLYGON,
        pockets_mm=POCKETS,
    )


def _session(scenario_id: str, positions: dict[int, tuple[float, float]]) -> TrainingSession:
    session = TrainingSession(TrainingConfig(scenario_id=scenario_id))
    judge_pockets = [(1_000.0, 250.0)] if scenario_id == "BEGINNER_SINGLE_STRAIGHT" else [(0.0, 0.0)]
    session.set_table_context(
        inner_polygon_mm=TABLE_POLYGON,
        ball_center_reachable_polygon_mm=TABLE_POLYGON,
        pockets_mm=judge_pockets,
    )
    session.update(_frame(1, positions))
    assert session.start()[0] is True
    return session


def _confirm_pot(
    session: TrainingSession,
    ball: int,
    frame_id: int,
    remaining: dict[int, tuple[float, float]],
):
    ts_ns = frame_id * 100_000_000
    group = _track(ball, 100.0, 100.0).group
    session.update(
        _frame(frame_id, remaining),
        PocketVisualObservationFrame(
            frame_id=frame_id,
            ts_cam_ns=ts_ns,
            observations=[
                PocketVisualObservation(
                    pocket_index=0,
                    inward_crossing=True,
                    group=group,
                    confidence=0.98,
                    associated_track_ids=[ball],
                    evidence_sources=["foreground_motion", "ball_sized_motion"],
                    motion_score=0.9,
                    foreground_score=0.8,
                    foreground_depth_diameters=0.5,
                )
            ],
        ),
    )
    resolved_frame = frame_id + 1
    resolved_ts_ns = ts_ns + 1_300_000_000
    return session.update(
        TracksFrame(
            frame_id=resolved_frame,
            ts_cam_ns=resolved_ts_ns,
            tracks=_frame(resolved_frame, remaining).tracks,
        ),
        PocketVisualObservationFrame(
            frame_id=resolved_frame,
            ts_cam_ns=resolved_ts_ns,
            observations=[PocketVisualObservation(pocket_index=0, clear=True)],
        ),
    )


def test_catalog_groups_six_beginner_and_seven_advanced_modes_without_changing_ids() -> None:
    scenarios = list_training_scenarios()
    beginner = [scenario for scenario in scenarios if scenario.group == "beginner"]
    advanced = [scenario for scenario in scenarios if scenario.group == "advanced"]

    assert [scenario.display_number for scenario in beginner] == list(range(1, 7))
    assert [scenario.display_number for scenario in advanced] == list(range(1, 8))
    assert [scenario.scenario_id for scenario in advanced] == [
        "ordered_line_1_7",
        "snake_1_15",
        "rotation_1_9",
        "odd_even_double_row",
        "solids_then_black",
        "stripes_then_black",
        "finish_6_7_8",
    ]
    assert {scenario.rule_set_id for scenario in beginner} == {"guided"}
    assert {scenario.rule_set_id for scenario in advanced} == {"strict"}


def test_shared_rule_seam_contains_all_guided_and_strict_behavior_differences() -> None:
    guided = get_training_rule_set("guided")
    strict = get_training_rule_set("strict")

    assert guided.evaluate_pocket_result([0, 2], [1]).action == "respot_continue"
    assert guided.evaluate_pocket_result([2], [1]).action == "respot_continue"
    assert guided.evaluate_pocket_result([1, 2], [1]).action == "respot_continue"
    assert strict.evaluate_pocket_result([0, 2], [1]).action == "fail"
    assert strict.evaluate_pocket_result([2], [1]).action == "fail"
    assert strict.evaluate_pocket_result([1, 2], [1]).action == "fail"
    assert strict.evaluate_pocket_result([1], [1]).action == "accept"


def test_single_free_layout_requires_exactly_cue_and_one_ball() -> None:
    assert _validate("BEGINNER_SINGLE_FREE", {0: (250, 350), 1: (600, 250)})[0] is True
    ok, message = _validate("BEGINNER_SINGLE_FREE", {0: (250, 350), 1: (600, 250), 2: (700, 300)})
    assert ok is False
    assert "移走" in message and "2" in message


def test_single_free_completes_on_one_but_not_on_cue_scratch() -> None:
    positions = {0: (250, 350), 1: (600, 250)}
    session = _session("BEGINNER_SINGLE_FREE", positions)
    scratched = _confirm_pot(session, 0, 2, {1: positions[1]})
    assert scratched.phase == "running"
    assert scratched.potted_numbers == []
    assert scratched.respot_required_numbers == [0]

    completed = _confirm_pot(session, 1, 4, {0: positions[0]})
    assert completed.phase == "passed"
    assert completed.potted_numbers == [1]


@pytest.mark.parametrize(
    ("cue", "target", "expected"),
    [
        ((200.0, 250.0), (400.0, 250.0), True),
        ((200.0, 285.0), (400.0, 250.0), True),
        ((200.0, 400.0), (400.0, 250.0), False),
    ],
)
def test_single_straight_layout_uses_wide_collinearity_tolerance(cue, target, expected) -> None:
    ok, message = _validate("BEGINNER_SINGLE_STRAIGHT", {0: cue, 1: target})
    assert ok is expected
    if not expected:
        assert "不共线" in message


def test_single_straight_completes_when_one_is_potted() -> None:
    positions = {0: (200, 250), 1: (400, 250)}
    session = _session("BEGINNER_SINGLE_STRAIGHT", positions)
    assert _confirm_pot(session, 1, 2, {0: positions[0]}).phase == "passed"


def test_beginner_missed_shot_keeps_current_target_and_progress() -> None:
    positions = {0: (250, 350), 1: (600, 250)}
    session = _session("BEGINNER_SINGLE_FREE", positions)
    state = session.update(_frame(2, positions))

    assert state.phase == "running"
    assert state.expected_numbers == [1]
    assert state.potted_numbers == []
    assert state.progress_current == 0


def test_one_to_three_short_line_checks_order_and_allows_small_deviation() -> None:
    valid = {0: (200, 380), 1: (350, 220), 2: (500, 250), 3: (650, 220)}
    assert _validate("BEGINNER_1_TO_3_LINE", valid)[0] is True
    ok, message = _validate(
        "BEGINNER_1_TO_3_LINE",
        {0: (200, 380), 1: (350, 220), 2: (650, 220), 3: (500, 220)},
    )
    assert ok is False
    assert "顺序" in message


def test_one_to_three_wrong_ball_warns_without_advancing_target_then_can_retry() -> None:
    positions = {0: (200, 380), 1: (350, 220), 2: (500, 220), 3: (650, 220)}
    session = _session("BEGINNER_1_TO_3_LINE", positions)
    warning = _confirm_pot(session, 2, 2, {0: positions[0], 1: positions[1], 3: positions[3]})
    assert warning.phase == "running"
    assert warning.error_count == 1
    assert warning.expected_numbers == [1]
    assert warning.potted_numbers == []

    after_one = _confirm_pot(session, 1, 4, {0: positions[0], 2: positions[2], 3: positions[3]})
    assert after_one.expected_numbers == [2]
    after_two = _confirm_pot(session, 2, 6, {0: positions[0], 3: positions[3]})
    assert after_two.expected_numbers == [3]
    assert _confirm_pot(session, 3, 8, {0: positions[0]}).phase == "passed"


def test_one_to_three_points_uses_normalized_zones_and_reports_swapped_ball() -> None:
    valid = {0: (180, 400), 1: (302, 170), 2: (698, 190), 3: (518, 330)}
    assert _validate("BEGINNER_1_TO_3_POINTS", valid)[0] is True
    swapped = dict(valid)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    ok, message = _validate("BEGINNER_1_TO_3_POINTS", swapped)
    assert ok is False
    assert "1 号球" in message and "推荐区域" in message


def test_one_to_three_points_keeps_fixed_sequence() -> None:
    positions = {0: (180, 400), 1: (302, 170), 2: (698, 190), 3: (518, 330)}
    session = _session("BEGINNER_1_TO_3_POINTS", positions)
    assert session.state.expected_numbers == [1]
    assert _confirm_pot(session, 1, 2, {0: positions[0], 2: positions[2], 3: positions[3]}).expected_numbers == [2]


@pytest.mark.parametrize(
    ("scenario_id", "numbers"),
    [
        ("BEGINNER_3_BALL_FREE", [1, 2, 3]),
        ("BEGINNER_5_BALL_FREE", [1, 2, 3, 4, 5]),
    ],
)
def test_free_clear_modes_accept_arbitrary_order_and_show_remaining(scenario_id: str, numbers: list[int]) -> None:
    positions = {0: (150, 400)}
    if scenario_id == "BEGINNER_3_BALL_FREE":
        positions.update({1: (302, 170), 2: (698, 190), 3: (518, 330)})
    else:
        positions.update({number: (220 + number * 120, 180 + (number % 2) * 110) for number in numbers})
    session = _session(scenario_id, positions)
    order = list(reversed(numbers))
    state = session.state
    for index, ball in enumerate(order):
        remaining_numbers = [number for number in numbers if number not in order[: index + 1]]
        remaining = {0: positions[0], **{number: positions[number] for number in remaining_numbers}}
        state = _confirm_pot(session, ball, 2 + index * 2, remaining)
        assert state.potted_numbers == order[: index + 1]
        assert state.remaining_numbers == [number for number in numbers if number in remaining_numbers]
        assert state.mode_hint == "自由选择模式"
    assert state.phase == "passed"


def test_five_ball_layout_reports_missing_and_extra_balls() -> None:
    positions = {0: (150, 400), 1: (300, 180), 2: (450, 300), 3: (600, 180), 4: (750, 300), 5: (850, 180)}
    assert _validate("BEGINNER_5_BALL_FREE", positions)[0] is True
    missing = dict(positions)
    missing.pop(5)
    assert "缺少目标球：5" in _validate("BEGINNER_5_BALL_FREE", missing)[1]
    extra = {**positions, 6: (300, 350)}
    assert "本项目外" in _validate("BEGINNER_5_BALL_FREE", extra)[1]


def test_five_ball_scratch_preserves_completed_progress() -> None:
    positions = {0: (150, 400), 1: (300, 180), 2: (450, 300), 3: (600, 180), 4: (750, 300), 5: (850, 180)}
    session = _session("BEGINNER_5_BALL_FREE", positions)
    after_three = _confirm_pot(
        session,
        3,
        2,
        {number: point for number, point in positions.items() if number != 3},
    )
    assert after_three.potted_numbers == [3]
    scratched = _confirm_pot(
        session,
        0,
        4,
        {number: point for number, point in positions.items() if number not in {0, 3}},
    )
    assert scratched.phase == "running"
    assert scratched.potted_numbers == [3]
    assert scratched.progress_current == 1


def test_advanced_stage_rules_are_unchanged() -> None:
    assert get_training_scenario("ordered_line_1_7").stages == tuple((number,) for number in range(1, 8))
    assert get_training_scenario("rotation_1_9").stages == tuple((number,) for number in range(1, 10))
    assert get_training_scenario("solids_then_black").stages == (tuple(range(1, 8)), (8,))
    assert get_training_scenario("stripes_then_black").stages == (tuple(range(9, 16)), (8,))
    assert get_training_scenario("finish_6_7_8").stages == ((6,), (7,), (8,))


@pytest.mark.parametrize(
    ("scenario_id", "numbers", "wrong_ball"),
    [
        ("rotation_1_9", list(range(1, 10)), 2),
        ("solids_then_black", list(range(1, 9)), 8),
        ("stripes_then_black", list(range(8, 16)), 8),
    ],
)
def test_advanced_rotation_and_black_rules_remain_strict(
    scenario_id: str,
    numbers: list[int],
    wrong_ball: int,
) -> None:
    positions = {0: (120, 420)}
    positions.update({number: (180 + number * 45, 180 + (number % 3) * 70) for number in numbers})
    session = _session(scenario_id, positions)
    remaining = {number: point for number, point in positions.items() if number != wrong_ball}
    state = _confirm_pot(session, wrong_ball, 2, remaining)

    assert state.phase == "failed"
    assert state.failure_reason == "wrong_ball"


def test_advanced_finish_six_seven_eight_completion_is_unchanged() -> None:
    positions = {0: (180, 400), 6: (420, 180), 7: (600, 260), 8: (780, 180)}
    session = _session("finish_6_7_8", positions)
    state = session.state
    for index, ball in enumerate((6, 7, 8)):
        remaining = {
            number: point
            for number, point in positions.items()
            if number == 0 or number in (6, 7, 8)[index + 1 :]
        }
        state = _confirm_pot(session, ball, 2 + index * 2, remaining)
    assert state.phase == "passed"
    assert state.potted_numbers == [6, 7, 8]
