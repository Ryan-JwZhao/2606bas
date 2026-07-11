from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .schemas import Event


def normalize_shot_mode(value: str | None) -> str:
    mode = str(value or "rule").strip().lower()
    return "free" if mode in {"free", "free_shot"} else "rule"


def normalize_target_group(value: str | None) -> Optional[str]:
    group = str(value or "").strip().lower()
    return group if group in {"solid", "stripe", "black"} else None


def toggled_object_group(value: str | None) -> Optional[str]:
    current = normalize_target_group(value)
    if current == "solid":
        return "stripe"
    if current == "stripe":
        return "solid"
    return None


@dataclass
class RuntimeControlState:
    free_shot_active: bool = False
    black_shot_active: bool = False

    def clear_turn_overrides(self) -> None:
        self.free_shot_active = False
        self.black_shot_active = False

    def arm_free_shot(self) -> None:
        self.free_shot_active = True
        self.black_shot_active = False

    def arm_black_shot(self) -> None:
        self.black_shot_active = True

    def advance_from_events(self, events: Iterable[Event]) -> None:
        if any(str(event.name).strip().upper() == "TURN_RESOLVE" for event in events):
            self.clear_turn_overrides()

    def effective_shot_mode(self, base_mode: str | None) -> str:
        if self.free_shot_active:
            return "free"
        return normalize_shot_mode(base_mode)

    def effective_turn_target_group(
        self,
        base_turn_target_group: str | None,
        effective_shot_mode: str | None,
    ) -> Optional[str]:
        if normalize_shot_mode(effective_shot_mode) != "rule":
            return normalize_target_group(base_turn_target_group)
        if self.black_shot_active:
            return "black"
        return normalize_target_group(base_turn_target_group)
