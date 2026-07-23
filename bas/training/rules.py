from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


ACTION_NONE = "none"
ACTION_ACCEPT = "accept"
ACTION_RESPOT_CONTINUE = "respot_continue"
ACTION_FAIL = "fail"


@dataclass(frozen=True)
class TrainingRuleDecision:
    """A pure rule result consumed by the shared training session engine."""

    action: str
    reason: str | None = None
    ball: int | None = None
    balls: tuple[int, ...] = ()
    expected: tuple[int, ...] = ()
    cue_ball_pocketed: bool = False


@dataclass(frozen=True)
class TrainingRuleSet:
    """The only behavior that varies between training styles."""

    rule_set_id: str
    cue_ball_pocketed_action: str
    wrong_ball_action: str
    multiple_object_balls_action: str

    def evaluate_pocket_result(
        self,
        confirmed_numbers: Iterable[int],
        expected_numbers: Iterable[int],
    ) -> TrainingRuleDecision:
        confirmed = tuple(dict.fromkeys(int(number) for number in confirmed_numbers))
        expected = tuple(sorted({int(number) for number in expected_numbers}))
        cue_ball_pocketed = 0 in confirmed
        object_balls = tuple(number for number in confirmed if number != 0)

        if cue_ball_pocketed:
            return TrainingRuleDecision(
                action=self.cue_ball_pocketed_action,
                reason="cue_ball_pocketed",
                balls=object_balls,
                expected=expected,
                cue_ball_pocketed=True,
            )
        if len(object_balls) > 1:
            return TrainingRuleDecision(
                action=self.multiple_object_balls_action,
                reason="ambiguous_multiple_pots",
                balls=object_balls,
                expected=expected,
            )
        if not object_balls:
            return TrainingRuleDecision(action=ACTION_NONE, expected=expected)

        ball = object_balls[0]
        if ball not in expected:
            return TrainingRuleDecision(
                action=self.wrong_ball_action,
                reason="wrong_ball",
                ball=ball,
                balls=(ball,),
                expected=expected,
            )
        return TrainingRuleDecision(
            action=ACTION_ACCEPT,
            ball=ball,
            balls=(ball,),
            expected=expected,
        )


GUIDED_RULES = TrainingRuleSet(
    rule_set_id="guided",
    cue_ball_pocketed_action=ACTION_RESPOT_CONTINUE,
    wrong_ball_action=ACTION_RESPOT_CONTINUE,
    multiple_object_balls_action=ACTION_RESPOT_CONTINUE,
)

STRICT_RULES = TrainingRuleSet(
    rule_set_id="strict",
    cue_ball_pocketed_action=ACTION_FAIL,
    wrong_ball_action=ACTION_FAIL,
    multiple_object_balls_action=ACTION_FAIL,
)

_RULE_SETS = {
    GUIDED_RULES.rule_set_id: GUIDED_RULES,
    STRICT_RULES.rule_set_id: STRICT_RULES,
}


def get_training_rule_set(rule_set_id: str | None) -> TrainingRuleSet:
    normalized = str(rule_set_id or STRICT_RULES.rule_set_id).strip().lower()
    try:
        return _RULE_SETS[normalized]
    except KeyError as exc:
        raise ValueError(f"未知训练规则集: {rule_set_id}") from exc


__all__ = [
    "ACTION_NONE",
    "ACTION_ACCEPT",
    "ACTION_RESPOT_CONTINUE",
    "ACTION_FAIL",
    "TrainingRuleDecision",
    "TrainingRuleSet",
    "GUIDED_RULES",
    "STRICT_RULES",
    "get_training_rule_set",
]
