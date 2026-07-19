from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..schemas import TrackObservation
from .models import TrainingScenario
from .numbered_tracker import ball_number_from_track


SCENARIOS: tuple[TrainingScenario, ...] = (
    TrainingScenario(
        scenario_id="ordered_line_1_7",
        title="1–7 顺序一字线",
        description="练习连续进球、短距离走位和下一颗球的落位角度。",
        setup_instructions="将 1–7 号球按号码顺序摆成一条直线，白球自由摆放；按 1→7 依次进球。",
        required_balls=tuple(range(1, 8)),
        stages=tuple((number,) for number in range(1, 8)),
        layout="line",
    ),
    TrainingScenario(
        scenario_id="snake_1_15",
        title="15 球中线蛇彩",
        description="综合训练准度、轻重杆、长距离走位和连续清台稳定性。",
        setup_instructions="将 1–15 号球沿球台长轴按号码排成一线，白球自由摆放；按 1→15 清台。",
        required_balls=tuple(range(1, 16)),
        stages=tuple((number,) for number in range(1, 16)),
        layout="line",
    ),
    TrainingScenario(
        scenario_id="rotation_1_9",
        title="1–9 轮转清台",
        description="训练开放球形中的路线规划，以及始终走到下一颗最低号球。",
        setup_instructions="将 1–9 号球分散摆放且彼此不贴球，白球自由摆放；按 1→9 依次进球。",
        required_balls=tuple(range(1, 10)),
        stages=tuple((number,) for number in range(1, 10)),
        layout="free",
    ),
    TrainingScenario(
        scenario_id="odd_even_double_row",
        title="单双号双排阶梯",
        description="通过跨区域切换目标，训练横台走位、母球速度和线路转换。",
        setup_instructions="奇数球 1/3/5/7/9 摆一排，偶数球 2/4/6/8 平行摆另一排；按奇数升序后偶数升序进球。",
        required_balls=tuple(range(1, 10)),
        stages=tuple((number,) for number in (1, 3, 5, 7, 9, 2, 4, 6, 8)),
        layout="double_row",
    ),
    TrainingScenario(
        scenario_id="solids_then_black",
        title="实色清台 + 黑八",
        description="模拟中式八球实色组收官：组内可自行规划，黑八必须最后进入。",
        setup_instructions="分散摆放 1–7 号球和 8 号球；先以任意顺序清掉 1–7，最后进 8 号球。",
        required_balls=tuple(range(1, 9)),
        stages=(tuple(range(1, 8)), (8,)),
        layout="free",
    ),
    TrainingScenario(
        scenario_id="stripes_then_black",
        title="花色清台 + 黑八",
        description="模拟中式八球花色组收官：组内可自行规划，黑八必须最后进入。",
        setup_instructions="分散摆放 9–15 号球和 8 号球；先以任意顺序清掉 9–15，最后进 8 号球。",
        required_balls=tuple(range(8, 16)),
        stages=(tuple(range(9, 16)), (8,)),
        layout="free",
    ),
    TrainingScenario(
        scenario_id="finish_6_7_8",
        title="6–7–8 三球收官",
        description="用短局反复训练关键球、过渡球和黑八收尾。",
        setup_instructions="分散摆放 6、7、8 号球，白球自由摆放；按 6→7→8 完成。",
        required_balls=(6, 7, 8),
        stages=((6,), (7,), (8,)),
        layout="free",
    ),
)

_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}


def list_training_scenarios() -> tuple[TrainingScenario, ...]:
    return SCENARIOS


def get_training_scenario(scenario_id: str | None) -> TrainingScenario:
    return _BY_ID.get(str(scenario_id or "").strip(), SCENARIOS[0])


def validate_scenario_setup(
    scenario: TrainingScenario,
    tracks: Sequence[TrackObservation],
    *,
    ball_diameter_mm: float = 57.15,
) -> tuple[bool, str]:
    visible = [track for track in tracks if track.visibility == "visible"]
    numbered = {number: track for track in visible if (number := ball_number_from_track(track)) is not None}
    visible_objects = {number for number in numbered if number != 0}
    required = set(scenario.required_balls)

    if scenario.require_cue_ball and 0 not in numbered:
        return False, "未识别到白球（0 号）"
    missing = sorted(required - visible_objects)
    if missing:
        return False, "缺少目标球：" + "、".join(str(number) for number in missing)
    if not scenario.allow_extra_object_balls:
        extras = sorted(visible_objects - required)
        if extras:
            return False, "请移走本项目外的球：" + "、".join(str(number) for number in extras)

    layout_tracks = [numbered[number] for number in scenario.required_balls]
    if scenario.layout == "line":
        return _validate_line(layout_tracks, scenario.required_balls, ball_diameter_mm)
    if scenario.layout == "double_row":
        return _validate_double_row(numbered, ball_diameter_mm)
    return True, "摆球已通过，可开始训练"


def _track_points(tracks: Iterable[TrackObservation]) -> tuple[np.ndarray, float]:
    items = list(tracks)
    use_mm = bool(items) and all(track.center_mm is not None for track in items)
    if use_mm:
        points = np.asarray([track.center_mm for track in items], dtype=np.float32)
        scale = 57.15
    else:
        points = np.asarray([track.center_px for track in items], dtype=np.float32)
        radii = [max(2.0, float(track.radius_px)) for track in items]
        scale = 2.0 * float(np.median(radii)) if radii else 24.0
    return points.reshape((-1, 2)), scale


def _validate_line(
    tracks: Sequence[TrackObservation],
    expected_order: Sequence[int],
    ball_diameter_mm: float,
) -> tuple[bool, str]:
    if len(tracks) < 3:
        return True, "摆球已通过，可开始训练"
    points, observed_scale = _track_points(tracks)
    if all(track.center_mm is not None for track in tracks):
        observed_scale = max(1.0, float(ball_diameter_mm))
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    main_axis = axes[0]
    projections = centered @ main_axis
    residual = centered - np.outer(projections, main_axis)
    max_residual = float(np.max(np.linalg.norm(residual, axis=1)))
    if max_residual > observed_scale * 1.35:
        return False, "目标球未摆成一条直线"
    sorted_indexes = np.argsort(projections)
    actual = tuple(int(expected_order[index]) for index in sorted_indexes)
    expected = tuple(int(number) for number in expected_order)
    if actual not in {expected, tuple(reversed(expected))}:
        return False, "直线中的球号顺序不正确"
    return True, "摆球已通过，可开始训练"


def _validate_double_row(
    numbered: dict[int, TrackObservation],
    ball_diameter_mm: float,
) -> tuple[bool, str]:
    odd_numbers = (1, 3, 5, 7, 9)
    even_numbers = (2, 4, 6, 8)
    for numbers in (odd_numbers, even_numbers):
        ok, _ = _validate_line([numbered[number] for number in numbers], numbers, ball_diameter_mm)
        if not ok:
            return False, "单双号球需要分别摆成两条直线"
    odd_points, odd_scale = _track_points([numbered[number] for number in odd_numbers])
    even_points, even_scale = _track_points([numbered[number] for number in even_numbers])
    scale = max(odd_scale, even_scale, float(ball_diameter_mm) if numbered[1].center_mm is not None else 1.0)
    if float(np.linalg.norm(np.mean(odd_points, axis=0) - np.mean(even_points, axis=0))) < scale * 1.5:
        return False, "单双号两排距离过近"
    return True, "摆球已通过，可开始训练"
