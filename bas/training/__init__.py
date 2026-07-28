from __future__ import annotations

from .cue_ball_control import CueBallStop, CueBallStopObserver, CueBallTargetRegion
from .models import (
    RULES_MODE,
    TRAINING_MODE,
    BallTargetZone,
    CueBallControlGoal,
    LayoutConstraints,
    SetupTargetZoneGuide,
    TrainingScenario,
    TrainingStateFrame,
    normalize_operating_mode,
)
from .numbered_tracker import NumberedBallTracker, ball_number_from_track, numbered_ball_group, parse_ball_number
from .overlay import TrainingOverlayBuilder
from .pocket_judge import TrainingPocketJudge, TrainingPocketJudgment
from .projection_drills import build_projection_drill_overlay
from .rules import (
    GUIDED_RULES,
    STRICT_RULES,
    TrainingRuleDecision,
    TrainingRuleSet,
    get_training_rule_set,
)
from .scenarios import get_training_scenario, list_training_scenarios, validate_scenario_setup
from .session import TrainingSession
from .stroke_check import (
    STROKE_CHECK_RENDERER_ID,
    STROKE_CHECK_SCENARIO_ID,
    StrokeCheckDimensions,
    StrokeCheckOverlayBuilder,
    build_stroke_check_overlay,
)

__all__ = [
    "RULES_MODE",
    "TRAINING_MODE",
    "BallTargetZone",
    "CueBallControlGoal",
    "CueBallStop",
    "CueBallStopObserver",
    "CueBallTargetRegion",
    "LayoutConstraints",
    "SetupTargetZoneGuide",
    "TrainingScenario",
    "TrainingStateFrame",
    "normalize_operating_mode",
    "NumberedBallTracker",
    "ball_number_from_track",
    "numbered_ball_group",
    "parse_ball_number",
    "TrainingOverlayBuilder",
    "TrainingPocketJudge",
    "TrainingPocketJudgment",
    "TrainingRuleDecision",
    "TrainingRuleSet",
    "GUIDED_RULES",
    "STRICT_RULES",
    "get_training_rule_set",
    "get_training_scenario",
    "list_training_scenarios",
    "validate_scenario_setup",
    "TrainingSession",
    "STROKE_CHECK_RENDERER_ID",
    "STROKE_CHECK_SCENARIO_ID",
    "StrokeCheckDimensions",
    "StrokeCheckOverlayBuilder",
    "build_stroke_check_overlay",
    "build_projection_drill_overlay",
]
