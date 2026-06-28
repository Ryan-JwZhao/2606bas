from __future__ import annotations

from ..config import StateConfig
from .machine import MatchStateMachine as LegacyMatchStateMachine
from .modern import ModernMatchStateMachine


def normalize_state_machine_engine(value: object) -> str:
    engine = str(value or "legacy").strip().lower()
    if engine in {"modern", "new", "v2", "state_machine_new"}:
        return "modern"
    return "legacy"


def create_match_state_machine(config: StateConfig):
    engine = normalize_state_machine_engine(getattr(config, "engine", "legacy"))
    if engine == "modern":
        return ModernMatchStateMachine(config)
    return LegacyMatchStateMachine(config)


MatchStateMachine = LegacyMatchStateMachine

__all__ = [
    "MatchStateMachine",
    "LegacyMatchStateMachine",
    "ModernMatchStateMachine",
    "create_match_state_machine",
    "normalize_state_machine_engine",
]
