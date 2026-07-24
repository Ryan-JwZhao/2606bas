from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..schemas import TrackObservation
from .layout_validation import validate_scenario_setup as _validate_scenario_setup
from .models import BallTargetZone, CueBallControlGoal, LayoutConstraints, TrainingScenario


_BEGINNER_SAFETY = LayoutConstraints(
    min_spacing_ball_diameters=1.35,
    edge_margin_ball_diameters=0.35,
    pocket_clearance_ball_diameters=1.35,
)

BEGINNER_SCENARIOS: tuple[TrainingScenario, ...] = (
    TrainingScenario(
        scenario_id="BEGINNER_SINGLE_FREE",
        title="单球自由入袋",
        description="只练习把 1 号球打入任意球袋，白球落袋后重摆继续。",
        setup_instructions="仅摆放白球和 1 号球，两颗球放在有效击球区域内并避开库边和袋口。",
        required_balls=(1,),
        stages=((1,),),
        group="beginner",
        display_number=1,
        constraints=_BEGINNER_SAFETY,
        rule_set_id="guided",
    ),
    TrainingScenario(
        scenario_id="BEGINNER_SINGLE_STRAIGHT",
        title="单球直线入袋",
        description="练习白球、目标球和袋口方向大致共线的基础直球。",
        setup_instructions="仅摆放白球和 1 号球，使白球、1 号球与任一可用袋口方向大致成直线。",
        required_balls=(1,),
        stages=((1,),),
        layout="straight_shot",
        group="beginner",
        display_number=2,
        constraints=LayoutConstraints(
            line_tolerance_ball_diameters=2.0,
            max_span_ball_diameters=8.0,
            min_spacing_ball_diameters=1.35,
            edge_margin_ball_diameters=0.25,
            pocket_clearance_ball_diameters=1.1,
        ),
        rule_set_id="guided",
    ),
    TrainingScenario(
        scenario_id="BEGINNER_1_TO_3_LINE",
        title="1–3 顺序短线",
        description="以较短的一字线练习 1→2→3 连续进球。",
        setup_instructions="将 1、2、3 号球按顺序摆成较短直线，白球自由摆放；按 1→2→3 进球。",
        required_balls=(1, 2, 3),
        stages=((1,), (2,), (3,)),
        layout="line",
        group="beginner",
        display_number=3,
        constraints=LayoutConstraints(
            line_tolerance_ball_diameters=2.0,
            max_span_ball_diameters=9.0,
            min_spacing_ball_diameters=1.25,
            edge_margin_ball_diameters=0.25,
            pocket_clearance_ball_diameters=1.1,
        ),
        rule_set_id="guided",
    ),
    TrainingScenario(
        scenario_id="BEGINNER_1_TO_3_POINTS",
        title="1–3 三点顺序",
        description="练习在三个不同方向之间切换，并按 1→2→3 完成。",
        setup_instructions="将 1、2、3 分别放入三个宽松推荐区域；三球无需共线，按 1→2→3 进球。",
        required_balls=(1, 2, 3),
        stages=((1,), (2,), (3,)),
        layout="zones",
        group="beginner",
        display_number=4,
        zones=(
            BallTargetZone(1, (0.28, 0.30), (0.20, 0.22)),
            BallTargetZone(2, (0.72, 0.35), (0.20, 0.22)),
            BallTargetZone(3, (0.52, 0.70), (0.22, 0.20)),
        ),
        constraints=_BEGINNER_SAFETY,
        rule_set_id="guided",
    ),
    TrainingScenario(
        scenario_id="BEGINNER_3_BALL_FREE",
        title="三球自由清台",
        description="1、2、3 可按任意顺序入袋，练习自主选择简单路线。",
        setup_instructions="仅摆放白球和 1、2、3 号球，分散在容易处理的位置；进球顺序自由。",
        required_balls=(1, 2, 3),
        stages=((1, 2, 3),),
        layout="zones",
        group="beginner",
        display_number=5,
        zones=(
            BallTargetZone(1, (0.28, 0.30), (0.22, 0.23)),
            BallTargetZone(2, (0.72, 0.35), (0.22, 0.23)),
            BallTargetZone(3, (0.52, 0.70), (0.24, 0.22)),
        ),
        constraints=_BEGINNER_SAFETY,
        rule_set_id="guided",
    ),
    TrainingScenario(
        scenario_id="BEGINNER_5_BALL_FREE",
        title="五球自由清台",
        description="1–5 可按任意顺序入袋，完成基础自由清台。",
        setup_instructions="仅摆放白球和 1–5 号球，在有效区域内自由分散并避开贴库、贴球和袋口。",
        required_balls=(1, 2, 3, 4, 5),
        stages=((1, 2, 3, 4, 5),),
        group="beginner",
        display_number=6,
        constraints=_BEGINNER_SAFETY,
        rule_set_id="guided",
    ),
)


_BEGINNER_BY_ID = {scenario.scenario_id: scenario for scenario in BEGINNER_SCENARIOS}


def _entry_variant(
    source_id: str,
    *,
    scenario_id: str,
    display_number: int,
    description: str,
) -> TrainingScenario:
    """Reuse a beginner drill definition while exposing it in the entry catalog."""

    return replace(
        _BEGINNER_BY_ID[source_id],
        scenario_id=scenario_id,
        group="entry",
        display_number=display_number,
        description=description,
    )


ENTRY_SCENARIOS: tuple[TrainingScenario, ...] = (
    _entry_variant(
        "BEGINNER_1_TO_3_LINE",
        scenario_id="ENTRY_1_1_TO_3_LINE",
        display_number=1,
        description="认识目标球并练习 1→2→3 的简单连续击球。",
    ),
    _entry_variant(
        "BEGINNER_1_TO_3_POINTS",
        scenario_id="ENTRY_2_1_TO_3_POINTS",
        display_number=2,
        description="练习在三个容易进球的位置之间切换击球方向。",
    ),
    _entry_variant(
        "BEGINNER_3_BALL_FREE",
        scenario_id="ENTRY_3_3_BALL_FREE",
        display_number=3,
        description="观察台面并自主选择 1、2、3 号球的处理顺序。",
    ),
    TrainingScenario(
        scenario_id="ENTRY_4_LEFT_RIGHT_3_BALL",
        title="左右换边三球",
        description="练习大方向走位，以及目标球在球台左右两侧之间的切换。",
        setup_instructions="将 1 号球放在左侧、2 号球放在右侧、3 号球放在中部推荐区域；按 1→2→3 进球。",
        required_balls=(1, 2, 3),
        stages=((1,), (2,), (3,)),
        layout="zones",
        group="entry",
        display_number=4,
        zones=(
            BallTargetZone(1, (0.25, 0.45), (0.14, 0.28)),
            BallTargetZone(2, (0.75, 0.45), (0.14, 0.28)),
            BallTargetZone(3, (0.50, 0.65), (0.11, 0.22)),
        ),
        constraints=_BEGINNER_SAFETY,
        rule_set_id="guided",
    ),
    _entry_variant(
        "BEGINNER_5_BALL_FREE",
        scenario_id="ENTRY_5_5_BALL_FREE",
        display_number=5,
        description="第一次体验由 1–5 号球组成的小型自由清台。",
    ),
)


def _cue_control_single_ball(
    *,
    scenario_id: str,
    title: str,
    description: str,
    setup_instructions: str,
    display_number: int,
    goal: CueBallControlGoal,
) -> TrainingScenario:
    return replace(
        _BEGINNER_BY_ID["BEGINNER_SINGLE_STRAIGHT"],
        scenario_id=scenario_id,
        title=title,
        description=description,
        setup_instructions=setup_instructions,
        group="cue_control",
        display_number=display_number,
        rule_set_id="strict",
        cue_ball_goals=(goal,),
        require_cue_ball_settle_after_pot=True,
        success_message="白球控制达标，训练完成",
    )


CUE_CONTROL_SCENARIOS: tuple[TrainingScenario, ...] = (
    TrainingScenario(
        scenario_id="CUE_CONTROL_TWO_BALL_LINK",
        title="两球基础衔接",
        description="先打 1 号球，将白球留到较大的合格区域，再继续处理 2 号球。",
        setup_instructions="按推荐区域摆放白球、1 号球和 2 号球；必须先打 1 号，再打 2 号。",
        required_balls=(1, 2),
        stages=((1,), (2,)),
        layout="zones",
        group="cue_control",
        display_number=1,
        zones=(
            BallTargetZone(0, (0.20, 0.68), (0.14, 0.18)),
            BallTargetZone(1, (0.42, 0.42), (0.16, 0.20)),
            BallTargetZone(2, (0.78, 0.30), (0.15, 0.20)),
        ),
        constraints=_BEGINNER_SAFETY,
        rule_set_id="strict",
        cue_ball_goals=(
            CueBallControlGoal(
                after_ball=1,
                kind="table_zone",
                center=(0.56, 0.62),
                radius_ball_diameters=4.5,
            ),
        ),
        require_cue_ball_settle_after_pot=True,
        success_message="两球衔接完成，白球控制达标",
    ),
    _cue_control_single_ball(
        scenario_id="CUE_CONTROL_STOP_ZONE",
        title="白球停球区",
        description="1 号球进袋后，让白球停在起始位置附近的大圆形区域内。",
        setup_instructions="按基础直球方式摆放白球和 1 号球；1 号球进袋后，白球应停在原位附近。",
        display_number=2,
        goal=CueBallControlGoal(
            after_ball=1,
            kind="stop",
            radius_ball_diameters=3.0,
        ),
    ),
    _cue_control_single_ball(
        scenario_id="CUE_CONTROL_FOLLOW_ZONE",
        title="白球前进区",
        description="1 号球进袋后，让白球进入目标球前方的宽泛区域。",
        setup_instructions="按基础直球方式摆放白球和 1 号球；使用跟杆让白球穿过目标球原位置并进入前方区域。",
        display_number=3,
        goal=CueBallControlGoal(
            after_ball=1,
            kind="follow",
            radius_ball_diameters=3.0,
            distance_ball_diameters=4.0,
        ),
    ),
    _cue_control_single_ball(
        scenario_id="CUE_CONTROL_DRAW_ZONE",
        title="白球回位区",
        description="1 号球进袋后，让白球回到目标球后方的宽泛区域。",
        setup_instructions="按基础直球方式摆放白球和 1 号球；使用低杆让白球回到目标球后方区域。",
        display_number=4,
        goal=CueBallControlGoal(
            after_ball=1,
            kind="draw",
            radius_ball_diameters=3.0,
            distance_ball_diameters=4.0,
        ),
    ),
)


ADVANCED_SCENARIOS: tuple[TrainingScenario, ...] = (
    TrainingScenario(
        scenario_id="ordered_line_1_7",
        title="1–7 顺序一字线",
        description="练习连续进球、短距离走位和下一颗球的落位角度。",
        setup_instructions="将 1–7 号球按号码顺序摆成一条直线，白球自由摆放；按 1→7 依次进球。",
        required_balls=tuple(range(1, 8)),
        stages=tuple((number,) for number in range(1, 8)),
        layout="line",
        display_number=1,
    ),
    TrainingScenario(
        scenario_id="snake_1_15",
        title="15 球中线蛇彩",
        description="综合训练准度、轻重杆、长距离走位和连续清台稳定性。",
        setup_instructions="将 1–15 号球沿球台长轴按号码排成一线，白球自由摆放；按 1→15 清台。",
        required_balls=tuple(range(1, 16)),
        stages=tuple((number,) for number in range(1, 16)),
        layout="line",
        display_number=2,
    ),
    TrainingScenario(
        scenario_id="rotation_1_9",
        title="1–9 轮转清台",
        description="训练开放球形中的路线规划，以及始终走到下一颗最低号球。",
        setup_instructions="将 1–9 号球分散摆放且彼此不贴球，白球自由摆放；按 1→9 依次进球。",
        required_balls=tuple(range(1, 10)),
        stages=tuple((number,) for number in range(1, 10)),
        display_number=3,
    ),
    TrainingScenario(
        scenario_id="odd_even_double_row",
        title="单双号双排阶梯",
        description="通过跨区域切换目标，训练横台走位、母球速度和线路转换。",
        setup_instructions="奇数球 1/3/5/7/9 摆一排，偶数球 2/4/6/8 平行摆另一排；按奇数升序后偶数升序进球。",
        required_balls=tuple(range(1, 10)),
        stages=tuple((number,) for number in (1, 3, 5, 7, 9, 2, 4, 6, 8)),
        layout="double_row",
        display_number=4,
    ),
    TrainingScenario(
        scenario_id="solids_then_black",
        title="实色清台 + 黑八",
        description="模拟中式八球实色组收官：组内可自行规划，黑八必须最后进入。",
        setup_instructions="分散摆放 1–7 号球和 8 号球；先以任意顺序清掉 1–7，最后进 8 号球。",
        required_balls=tuple(range(1, 9)),
        stages=(tuple(range(1, 8)), (8,)),
        display_number=5,
    ),
    TrainingScenario(
        scenario_id="stripes_then_black",
        title="花色清台 + 黑八",
        description="模拟中式八球花色组收官：组内可自行规划，黑八必须最后进入。",
        setup_instructions="分散摆放 9–15 号球和 8 号球；先以任意顺序清掉 9–15，最后进 8 号球。",
        required_balls=tuple(range(8, 16)),
        stages=(tuple(range(9, 16)), (8,)),
        display_number=6,
    ),
    TrainingScenario(
        scenario_id="finish_6_7_8",
        title="6–7–8 三球收官",
        description="用短局反复训练关键球、过渡球和黑八收尾。",
        setup_instructions="分散摆放 6、7、8 号球，白球自由摆放；按 6→7→8 完成。",
        required_balls=(6, 7, 8),
        stages=((6,), (7,), (8,)),
        display_number=7,
    ),
)

SCENARIOS = BEGINNER_SCENARIOS + ENTRY_SCENARIOS + CUE_CONTROL_SCENARIOS + ADVANCED_SCENARIOS
_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}
_DEFAULT_SCENARIO = _BY_ID["ordered_line_1_7"]


def list_training_scenarios() -> tuple[TrainingScenario, ...]:
    return SCENARIOS


def get_training_scenario(scenario_id: str | None) -> TrainingScenario:
    return _BY_ID.get(str(scenario_id or "").strip(), _DEFAULT_SCENARIO)


def validate_scenario_setup(
    scenario: TrainingScenario,
    tracks: Sequence[TrackObservation],
    *,
    ball_diameter_mm: float = 57.15,
    ball_center_reachable_polygon_mm: Sequence[tuple[float, float]] | None = None,
    pockets_mm: Sequence[tuple[float, float]] | None = None,
) -> tuple[bool, str]:
    return _validate_scenario_setup(
        scenario,
        tracks,
        ball_diameter_mm=ball_diameter_mm,
        ball_center_reachable_polygon_mm=ball_center_reachable_polygon_mm,
        pockets_mm=pockets_mm,
    )
