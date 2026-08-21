from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from ..config import PlannerConfig
from ..schemas import ShotCandidate
from .target_shot import TargetShotDecision, TargetShotPlanner


@dataclass(frozen=True)
class HookShotResult:
    candidates: list[ShotCandidate]
    status: str
    evaluated_target_ids: tuple[int, ...]

    @property
    def best(self) -> ShotCandidate | None:
        return self.candidates[0] if self.candidates else None


class HookShotPlanner:
    """Ranks the best direct/one-bank/two-bank route across eligible balls."""

    version = "hook_shot_global_v1"

    def __init__(self, config: PlannerConfig, route_planner: TargetShotPlanner):
        self.config = config
        self.route_planner = route_planner
        self.last_status = "idle"

    def plan(
        self,
        *,
        cue_ball: object,
        targets: Sequence[object],
        balls: Sequence[object],
        selection_source: str,
    ) -> HookShotResult:
        candidates: list[ShotCandidate] = []
        eligible_targets = [target for target in targets if str(getattr(target, "group")) != "cue"]
        evaluated_ids = [int(getattr(target, "track_id")) for target in eligible_targets]
        for target in eligible_targets:
            target_id = int(getattr(target, "track_id"))
            target_group = str(getattr(target, "group"))
            decision = TargetShotDecision(
                active_target_id=target_id,
                active_group=target_group,
                status=f"hook_{selection_source}",
                active=True,
            )
            candidate = self.route_planner.plan(
                cue_ball=cue_ball,
                target=target,
                balls=balls,
                decision=decision,
            )
            if candidate is not None:
                explanation = dict(candidate.explanation)
                explanation.update(
                    {
                        "hook_shot": True,
                        "hook_shot_version": self.version,
                        "hook_selection_source": str(selection_source),
                        "hook_evaluated_target_ids": list(evaluated_ids),
                    }
                )
                annotated = replace(candidate, explanation=explanation)
                candidates.append(annotated)

        candidates.sort(key=lambda item: (float(item.score), -float(item.risk)), reverse=True)
        candidates = candidates[: max(1, int(self.config.top_k))]
        if candidates:
            self.last_status = f"ok:{selection_source}:{len(evaluated_ids)}_targets"
        elif evaluated_ids:
            self.last_status = f"no_theoretical_route:{selection_source}:{len(evaluated_ids)}_targets"
        else:
            self.last_status = f"no_eligible_target:{selection_source}"
        return HookShotResult(
            candidates=candidates,
            status=self.last_status,
            evaluated_target_ids=tuple(evaluated_ids),
        )
