from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ..schemas import Event


RULES_MODE = "rules"
TRAINING_MODE = "training"


def normalize_operating_mode(value: str | None) -> str:
    mode = str(value or RULES_MODE).strip().lower()
    return TRAINING_MODE if mode in {"training", "train", "practice", "drill"} else RULES_MODE


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

    @property
    def ordered_numbers(self) -> Tuple[int, ...]:
        return tuple(number for stage in self.stages for number in stage)


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
    events: list[Event] = field(default_factory=list)

    @property
    def progress_ratio(self) -> float:
        if self.progress_total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.progress_current / self.progress_total))
