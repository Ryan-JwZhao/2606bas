from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import TrainingStateFrame
from .rules import ACTION_RESPOT_CONTINUE, TrainingRuleDecision


@dataclass(frozen=True)
class TrainingProjectionPrompt:
    text: str
    color: tuple[int, int, int] = (255, 255, 255)


def running_message(
    expected_numbers: Iterable[int],
    *,
    cue_respot_required: bool = False,
    respot_required_numbers: Iterable[int] = (),
) -> str:
    if cue_respot_required:
        return "请重新放置白球后继续，已完成进度会保留"
    respot = tuple(sorted({int(number) for number in respot_required_numbers}))
    if respot:
        return "请重摆误进球后继续：" + "、".join(str(number) for number in respot)
    expected = tuple(int(number) for number in expected_numbers)
    if not expected:
        return "正在确认训练结果"
    if len(expected) == 1:
        return f"当前目标：{expected[0]} 号球"
    return "当前可进：" + "、".join(str(number) for number in expected)


def rule_decision_message(decision: TrainingRuleDecision) -> str:
    expected_text = " / ".join(str(number) for number in decision.expected) or "目标球"
    ball_text = "、".join(str(number) for number in decision.balls)
    if decision.reason == "cue_ball_pocketed":
        if decision.action == ACTION_RESPOT_CONTINUE:
            message = "白球落袋：请重新放置白球后继续，已完成进度会保留"
            if decision.balls:
                message += f"；同时入袋的目标球也请重摆：{ball_text}"
            return message
        return "白球落袋，本次训练失败"
    if decision.reason == "ambiguous_multiple_pots":
        if decision.action == ACTION_RESPOT_CONTINUE:
            return f"同时检测到多颗球入袋，请将这些球重摆后继续：{ball_text}"
        return "多颗球同时消失，无法确认进球顺序"
    if decision.reason == "wrong_ball":
        if decision.action == ACTION_RESPOT_CONTINUE:
            return f"误进 {decision.ball} 号球；请重摆该球后继续，当前目标仍为 {expected_text}"
        return f"误进 {decision.ball} 号球；当前应进 {expected_text}"
    return ""


def mode_hint(stages: Sequence[Sequence[int]]) -> str:
    return "自由选择模式" if any(len(stage) > 1 for stage in stages) else "固定目标模式"


def completion_message(stages: Sequence[Sequence[int]]) -> str:
    return "训练完成，全部目标球已清台" if mode_hint(stages) == "自由选择模式" else "训练完成，全部目标球顺序正确"


def projection_prompt_for_state(state: TrainingStateFrame) -> TrainingProjectionPrompt:
    progress = f"进度 {state.progress_current}/{state.progress_total}"
    if state.phase == "setup":
        return TrainingProjectionPrompt(f"请完成摆球\n{state.message}", (255, 255, 255))
    if state.phase == "ready":
        return TrainingProjectionPrompt(f"摆球正确，可以开始训练\n{state.scenario_title}", (100, 235, 100))
    if state.phase == "passed":
        detail = f"{progress} · 用时 {state.elapsed_s:.1f} 秒"
        if state.error_count:
            detail += f" · 提示 {state.error_count} 次"
        return TrainingProjectionPrompt(f"训练完成！\n{detail}", (100, 235, 100))
    if state.phase == "failed":
        return TrainingProjectionPrompt(f"本次训练结束\n{state.message}；请重新开始", (80, 100, 255))

    if state.respot_required_numbers:
        return TrainingProjectionPrompt(f"请先完成重摆\n{state.message}", (0, 220, 255))
    target = state.message or running_message(state.expected_numbers)
    detail = f"{progress} · 用时 {state.elapsed_s:.1f} 秒"
    if state.error_count:
        detail += f" · 提示 {state.error_count} 次"
    return TrainingProjectionPrompt(f"{target}\n{detail}", (255, 255, 255))


__all__ = [
    "TrainingProjectionPrompt",
    "completion_message",
    "mode_hint",
    "projection_prompt_for_state",
    "rule_decision_message",
    "running_message",
]
