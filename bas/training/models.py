from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ..schemas import Event


RULES_MODE = "rules"
TRAINING_MODE = "training"
TRAINING_GROUP_TITLES = {
    "beginner": ("新手训练模式", "新手训练"),
    "entry": ("入门训练模式", "入门训练"),
    "cue_control": ("白球控制训练模式", "白球控制训练"),
    "advanced": ("进阶训练模式", "进阶训练"),
}


def normalize_operating_mode(value: str | None) -> str:
    mode = str(value or RULES_MODE).strip().lower()
    return TRAINING_MODE if mode in {"training", "train", "practice", "drill"} else RULES_MODE


@dataclass(frozen=True)
class BallTargetZone:
    ball: int
    center: tuple[float, float]
    tolerance: tuple[float, float] = (0.18, 0.18)


@dataclass(frozen=True)
class LayoutConstraints:
    line_tolerance_ball_diameters: float = 1.35
    max_span_ball_diameters: float | None = None
    min_spacing_ball_diameters: float = 0.0
    edge_margin_ball_diameters: float = 0.0
    pocket_clearance_ball_diameters: float = 0.0


@dataclass(frozen=True)
class CueBallControlGoal:
    after_ball: int
    kind: str
    radius_ball_diameters: float
    distance_ball_diameters: float = 0.0
    center: tuple[float, float] | None = None


@dataclass(frozen=True)
class TrainingScenario:
    scenario_id: str
    title: str
    description: str
    setup_instructions: str
    required_balls: Tuple[int, ...]
    stages: Tuple[Tuple[int, ...], ...]
    layout: str = "free"
    require_cue_ball: bool = True
    allow_extra_object_balls: bool = False
    group: str = "advanced"
    display_number: int = 0
    zones: Tuple[BallTargetZone, ...] = ()
    constraints: LayoutConstraints = LayoutConstraints()
    rule_set_id: str = "strict"
    cue_ball_goals: Tuple[CueBallControlGoal, ...] = ()
    require_cue_ball_settle_after_pot: bool = False
    success_message: str | None = None

    @property
    def ordered_numbers(self) -> Tuple[int, ...]:
        return tuple(number for stage in self.stages for number in stage)

    @property
    def group_title(self) -> str:
        return TRAINING_GROUP_TITLES.get(self.group, (self.group, self.group))[0]

    @property
    def display_title(self) -> str:
        prefix = TRAINING_GROUP_TITLES.get(self.group, (self.group, self.group))[1]
        return f"{prefix}{self.display_number}：{self.title}" if self.display_number > 0 else self.title


@dataclass
class TrainingStateFrame:
    frame_id: int
    ts_cam_ns: int
    scenario_id: str
    scenario_title: str
    phase: str = "setup"
    message: str = "请按说明摆球"
    setup_ready: bool = False
    required_numbers: list[int] = field(default_factory=list)
    expected_numbers: list[int] = field(default_factory=list)
    visible_numbers: list[int] = field(default_factory=list)
    potted_numbers: list[int] = field(default_factory=list)
    progress_current: int = 0
    progress_total: int = 0
    attempt: int = 0
    elapsed_s: float = 0.0
    failure_reason: str | None = None
    error_count: int = 0
    remaining_numbers: list[int] = field(default_factory=list)
    mode_hint: str = ""
    respot_required_numbers: list[int] = field(default_factory=list)
    cue_ball_status: str = "idle"
    cue_ball_position_mm: tuple[float, float] | None = None
    cue_ball_goal_center_mm: tuple[float, float] | None = None
    cue_ball_goal_radius_mm: float | None = None
    cue_ball_goal_polygon_mm: list[tuple[float, float]] = field(default_factory=list)
    cue_ball_goal_result: str | None = None
    events: list[Event] = field(default_factory=list)

    @property
    def progress_ratio(self) -> float:
        if self.progress_total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.progress_current / self.progress_total))
