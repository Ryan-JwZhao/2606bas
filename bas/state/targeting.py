from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from .models import GROUPS, ObjectGroup, TargetGroup, normalize_object_group


@dataclass(frozen=True)
class TargetGroupResolution:
    target_group: Optional[TargetGroup]
    effective_remaining: dict[str, int]
    review_required: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        return {
            "target_group": self.target_group,
            "effective_remaining": dict(self.effective_remaining),
            "review_required": bool(self.review_required),
            "reasons": list(self.reasons),
        }


def resolve_turn_target_group(
    raw_hint: Optional[str],
    *,
    actor_group: Optional[str],
    ledger_remaining: Mapping[str, int],
    observation_effective_remaining: Mapping[str, int] | None = None,
    visible_counts: Mapping[str, int] | None = None,
    stable_frames: Mapping[str, int] | None = None,
    stable_frames_required: int = 1,
    observation_review_required: bool = False,
) -> TargetGroupResolution:
    remaining = _remaining_view(ledger_remaining, observation_effective_remaining)
    reasons: list[str] = []
    review_required = bool(observation_review_required)
    active_object_group = _active_object_group(raw_hint, actor_group)

    if active_object_group is not None and _stable_zero_visible(
        active_object_group,
        visible_counts=visible_counts,
        stable_frames=stable_frames,
        stable_frames_required=stable_frames_required,
    ):
        if remaining.get(active_object_group, 0) > 0:
            reasons.append(f"stable_visible_{active_object_group}_cleared")
        remaining[active_object_group] = 0
        review_required = True

    target = _target_from_remaining(raw_hint, active_object_group, remaining)
    return TargetGroupResolution(
        target_group=target,
        effective_remaining=remaining,
        review_required=review_required,
        reasons=reasons,
    )


def _remaining_view(
    ledger_remaining: Mapping[str, int],
    observation_effective_remaining: Mapping[str, int] | None,
) -> dict[str, int]:
    remaining = {group: int(ledger_remaining.get(group, 0)) for group in GROUPS}
    if observation_effective_remaining is not None:
        for group in GROUPS:
            if group in observation_effective_remaining:
                remaining[group] = max(0, int(observation_effective_remaining.get(group, remaining[group])))
    return remaining


def _active_object_group(raw_hint: Optional[str], actor_group: Optional[str]) -> Optional[ObjectGroup]:
    return normalize_object_group(raw_hint) or normalize_object_group(actor_group)


def _stable_zero_visible(
    group: str,
    *,
    visible_counts: Mapping[str, int] | None,
    stable_frames: Mapping[str, int] | None,
    stable_frames_required: int,
) -> bool:
    if visible_counts is None or stable_frames is None:
        return False
    return int(visible_counts.get(group, 0)) == 0 and int(stable_frames.get(group, 0)) >= max(1, int(stable_frames_required))


def _target_from_remaining(
    raw_hint: Optional[str],
    active_object_group: Optional[ObjectGroup],
    remaining: Mapping[str, int],
) -> Optional[TargetGroup]:
    hint = str(raw_hint or "").strip().lower()
    if hint == "black":
        return "black" if int(remaining.get("black", 0)) > 0 else None

    if active_object_group is not None:
        if int(remaining.get(active_object_group, 0)) > 0:
            return active_object_group
        return "black" if int(remaining.get("black", 0)) > 0 else None

    if int(remaining.get("solid", 0)) == 0 and int(remaining.get("stripe", 0)) == 0:
        return "black" if int(remaining.get("black", 0)) > 0 else None
    return None
