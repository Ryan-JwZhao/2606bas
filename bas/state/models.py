from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Group = Literal["cue", "solid", "stripe", "black"]
ObjectGroup = Literal["solid", "stripe"]
TargetGroup = Literal["solid", "stripe", "black"]
PocketState = Literal[
    "on_table",
    "pocket_candidate",
    "pocket_tentative",
    "pocket_confirmed",
    "pocket_review_required",
    "pocket_rejected",
    "off_table_candidate",
    "lost",
]
TableState = Literal["open", "closed"]
BallInHandScope = Literal["none", "table_anywhere", "behind_head_string"]
GameStatus = Literal["in_progress", "ended_pending_review"]

GROUPS: tuple[Group, ...] = ("cue", "solid", "stripe", "black")
OBJECT_GROUPS: tuple[ObjectGroup, ...] = ("solid", "stripe")


def empty_group_counts() -> dict[Group, int]:
    return {group: 0 for group in GROUPS}


@dataclass
class BallEvidence:
    logical_id: str
    group: Group
    ts_ms: int
    center_mm: tuple[float, float]
    speed_mm_s: float
    det_conf: float
    pocket_state: PocketState
    pocket_index: Optional[int] = None
    missing_ms: int = 0
    toward_pocket_score: float = 0.0
    crossed_throat: bool = False
    reappeared: bool = False
    confidence: float = 0.0


@dataclass
class ShotContext:
    shot_id: int
    ts_start_ms: int
    ts_end_ms: Optional[int] = None
    break_shot: bool = False
    table_state_before: TableState = "open"
    actor_group: Optional[ObjectGroup] = None
    first_contact_group: Optional[Group] = None
    first_contact_confidence: float = 0.0
    potted_confirmed: dict[Group, int] = field(default_factory=empty_group_counts)
    off_table_confirmed: dict[Group, int] = field(default_factory=empty_group_counts)
    committed_pockets: list[dict[str, object]] = field(default_factory=list)
    tentative_pockets: list[dict[str, object]] = field(default_factory=list)
    review_pockets: list[dict[str, object]] = field(default_factory=list)
    rejected_pockets: list[dict[str, object]] = field(default_factory=list)
    rail_contact_seen: bool = False
    cue_scratch_candidate: bool = False
    wrong_first_contact_candidate: bool = False
    review_required: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass
class InventoryLedger:
    remaining: dict[Group, int] = field(default_factory=lambda: {"cue": 1, "solid": 7, "stripe": 7, "black": 1})
    removed_confirmed: dict[Group, int] = field(default_factory=empty_group_counts)

    def reset(self) -> None:
        self.remaining = {"cue": 1, "solid": 7, "stripe": 7, "black": 1}
        self.removed_confirmed = empty_group_counts()

    def apply(self, shot_ctx: ShotContext) -> None:
        for source in (shot_ctx.potted_confirmed, shot_ctx.off_table_confirmed):
            for group, count in source.items():
                if group == "cue" or count <= 0:
                    continue
                before = int(self.remaining.get(group, 0))
                taken = min(before, int(count))
                self.remaining[group] = max(0, before - taken)
                self.removed_confirmed[group] = int(self.removed_confirmed.get(group, 0)) + taken


@dataclass
class MatchRuleState:
    table_state: TableState = "open"
    actor_group: Optional[ObjectGroup] = None
    opponent_group: Optional[ObjectGroup] = None
    shot_number: int = 0
    game_status: GameStatus = "in_progress"

    @property
    def break_shot(self) -> bool:
        return self.shot_number == 0

    def reset(self) -> None:
        self.table_state = "open"
        self.actor_group = None
        self.opponent_group = None
        self.shot_number = 0
        self.game_status = "in_progress"


@dataclass
class RefereeIntent:
    next_group_hint: Optional[TargetGroup]
    next_actor_changed: bool
    table_state_after: TableState
    actor_group_after: Optional[ObjectGroup]
    opponent_group_after: Optional[ObjectGroup]
    ball_in_hand_scope: BallInHandScope
    foul_flags: dict[str, bool]
    review_required: bool
    group_choice_required: bool = False
    game_status: GameStatus = "in_progress"
    reasons: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        return {
            "next_group_hint": self.next_group_hint,
            "next_actor_changed": self.next_actor_changed,
            "table_state_after": self.table_state_after,
            "actor_group_after": self.actor_group_after,
            "opponent_group_after": self.opponent_group_after,
            "ball_in_hand_scope": self.ball_in_hand_scope,
            "foul_flags": dict(self.foul_flags),
            "review_required": self.review_required,
            "group_choice_required": self.group_choice_required,
            "game_status": self.game_status,
            "reasons": list(self.reasons),
        }


def normalize_group(value: object) -> Optional[Group]:
    group = str(value or "").strip().lower()
    if group in GROUPS:
        return group  # type: ignore[return-value]
    return None


def normalize_object_group(value: object) -> Optional[ObjectGroup]:
    group = str(value or "").strip().lower()
    if group in OBJECT_GROUPS:
        return group  # type: ignore[return-value]
    return None


def other_object_group(group: ObjectGroup) -> ObjectGroup:
    return "stripe" if group == "solid" else "solid"
