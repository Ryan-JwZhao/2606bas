from __future__ import annotations

from typing import Iterable, List, Optional

from ..schemas import Event
from .models import (
    GROUPS,
    InventoryLedger,
    MatchRuleState,
    ObjectGroup,
    RefereeIntent,
    ShotContext,
    TargetGroup,
    normalize_group,
    normalize_object_group,
    other_object_group,
)


class ShotContextAggregator:
    def __init__(self) -> None:
        self._active: Optional[ShotContext] = None
        self._next_shot_id = 1

    @property
    def active(self) -> Optional[ShotContext]:
        return self._active

    def reset(self) -> None:
        self._active = None
        self._next_shot_id = 1

    def begin_if_needed(self, *, ts_ms: int, rule_state: MatchRuleState) -> ShotContext:
        if self._active is None:
            self._active = ShotContext(
                shot_id=self._next_shot_id,
                ts_start_ms=ts_ms,
                break_shot=rule_state.break_shot,
                table_state_before=rule_state.table_state,
                actor_group=rule_state.actor_group,
            )
            self._next_shot_id += 1
        return self._active

    def ingest(self, events: Iterable[Event], *, ts_ms: int, rule_state: MatchRuleState) -> None:
        event_list = list(events)
        if any(event.name in {"SHOT_STARTED", "SHOT_START_VOTED", "POCKET_CANDIDATE", "POCKET_CONFIRMED"} for event in event_list):
            ctx = self.begin_if_needed(ts_ms=ts_ms, rule_state=rule_state)
        else:
            ctx = self._active
        if ctx is None:
            return
        for event in event_list:
            self._ingest_event(ctx, event)

    def finalize(self, *, ts_ms: int, rule_state: MatchRuleState) -> ShotContext:
        ctx = self.begin_if_needed(ts_ms=ts_ms, rule_state=rule_state)
        ctx.ts_end_ms = ts_ms
        self._active = None
        return ctx

    def _ingest_event(self, ctx: ShotContext, event: Event) -> None:
        payload = dict(event.payload or {})
        if event.name == "POCKET_CONFIRMED":
            group = normalize_group(payload.get("group"))
            if group is not None:
                ctx.potted_confirmed[group] = int(ctx.potted_confirmed.get(group, 0)) + 1
                if group == "cue":
                    ctx.cue_scratch_candidate = True
        elif event.name == "BALL_OFF_TABLE_CONFIRMED":
            group = normalize_group(payload.get("group"))
            if group is not None:
                ctx.off_table_confirmed[group] = int(ctx.off_table_confirmed.get(group, 0)) + 1
        elif event.name == "BALL_COLLISION_CANDIDATE":
            self._capture_first_contact(ctx, payload, confidence=float(event.confidence))
        elif event.name == "RAIL_COLLISION_CANDIDATE":
            ctx.rail_contact_seen = True
        elif event.name in {"POCKET_REAPPEARED", "BALL_LOST_UNCONFIRMED"}:
            ctx.reasons.append(event.name.lower())

    def _capture_first_contact(self, ctx: ShotContext, payload: dict[str, object], *, confidence: float) -> None:
        if ctx.first_contact_group is not None:
            return
        groups = [normalize_group(payload.get("group_a")), normalize_group(payload.get("group_b"))]
        if any(group == "cue" for group in groups):
            other = next((group for group in groups if group is not None and group != "cue"), None)
            if other is not None:
                ctx.first_contact_group = other
                ctx.first_contact_confidence = confidence


class RefereeAdapter:
    """Structured referee interface. It exposes flags but keeps final judging conservative."""

    def evaluate(self, shot_ctx: ShotContext, ledger: InventoryLedger, rule_state: MatchRuleState) -> RefereeIntent:
        actor = rule_state.actor_group
        opponent = rule_state.opponent_group
        legal_first = self._legal_first_contact_group(actor, ledger)
        foul_flags = {
            "cue_scratch": bool(shot_ctx.cue_scratch_candidate or shot_ctx.potted_confirmed.get("cue", 0) > 0),
            "wrong_first_contact": bool(
                legal_first is not None
                and shot_ctx.first_contact_group is not None
                and shot_ctx.first_contact_group != legal_first
            ),
            "no_rail_after_contact": False,
            "break_foul": False,
        }
        foul_flags["break_foul"] = bool(shot_ctx.break_shot and (foul_flags["cue_scratch"] or foul_flags["wrong_first_contact"]))
        foul = bool(foul_flags["cue_scratch"] or foul_flags["wrong_first_contact"] or foul_flags["break_foul"])
        ball_in_hand_scope = "behind_head_string" if shot_ctx.break_shot and foul else "table_anywhere" if foul else "none"
        reasons: List[str] = list(shot_ctx.reasons)
        if foul_flags["cue_scratch"]:
            reasons.append("cue_scratch")
        if foul_flags["wrong_first_contact"]:
            reasons.append("wrong_first_contact")

        black_potted = shot_ctx.potted_confirmed.get("black", 0) > 0
        game_status = "ended_pending_review" if black_potted else rule_state.game_status

        if rule_state.table_state == "open":
            return self._evaluate_open_table(shot_ctx, ledger, foul_flags, ball_in_hand_scope, game_status, reasons)

        if actor is None or opponent is None:
            return RefereeIntent(
                next_group_hint=None,
                next_actor_changed=True,
                table_state_after="open",
                actor_group_after=None,
                opponent_group_after=None,
                ball_in_hand_scope=ball_in_hand_scope,
                foul_flags=foul_flags,
                review_required=True,
                game_status=game_status,
                reasons=reasons + ["closed_table_missing_group_owner"],
            )

        keep_turn = (not foul) and shot_ctx.potted_confirmed.get(actor, 0) > 0
        next_actor = actor if keep_turn else opponent
        next_group = self._target_for_actor(next_actor, ledger)
        if game_status != "in_progress":
            next_group = None
        return RefereeIntent(
            next_group_hint=next_group,
            next_actor_changed=not keep_turn,
            table_state_after="closed",
            actor_group_after=next_actor,
            opponent_group_after=other_object_group(next_actor),
            ball_in_hand_scope=ball_in_hand_scope,
            foul_flags=foul_flags,
            review_required=bool(shot_ctx.review_required),
            game_status=game_status,
            reasons=reasons,
        )

    def _evaluate_open_table(
        self,
        shot_ctx: ShotContext,
        ledger: InventoryLedger,
        foul_flags: dict[str, bool],
        ball_in_hand_scope: str,
        game_status: str,
        reasons: list[str],
    ) -> RefereeIntent:
        object_potted = {
            group: int(shot_ctx.potted_confirmed.get(group, 0))
            for group in ("solid", "stripe")
            if shot_ctx.potted_confirmed.get(group, 0) > 0
        }
        foul = bool(foul_flags["cue_scratch"] or foul_flags["wrong_first_contact"] or foul_flags["break_foul"])
        if shot_ctx.break_shot or foul or not object_potted:
            return RefereeIntent(
                next_group_hint=None,
                next_actor_changed=not bool(object_potted and not foul),
                table_state_after="open",
                actor_group_after=None,
                opponent_group_after=None,
                ball_in_hand_scope=ball_in_hand_scope,  # type: ignore[arg-type]
                foul_flags=foul_flags,
                review_required=bool(shot_ctx.review_required),
                game_status=game_status,  # type: ignore[arg-type]
                reasons=reasons,
            )
        if len(object_potted) > 1:
            return RefereeIntent(
                next_group_hint=None,
                next_actor_changed=False,
                table_state_after="open",
                actor_group_after=None,
                opponent_group_after=None,
                ball_in_hand_scope=ball_in_hand_scope,  # type: ignore[arg-type]
                foul_flags=foul_flags,
                review_required=True,
                group_choice_required=True,
                game_status=game_status,  # type: ignore[arg-type]
                reasons=reasons + ["open_table_group_choice_required"],
            )
        actor = next(iter(object_potted.keys()))
        actor_group = normalize_object_group(actor)
        if actor_group is None:
            return RefereeIntent(
                next_group_hint=None,
                next_actor_changed=True,
                table_state_after="open",
                actor_group_after=None,
                opponent_group_after=None,
                ball_in_hand_scope=ball_in_hand_scope,  # type: ignore[arg-type]
                foul_flags=foul_flags,
                review_required=True,
                game_status=game_status,  # type: ignore[arg-type]
                reasons=reasons + ["open_table_invalid_group"],
            )
        next_group = self._target_for_actor(actor_group, ledger)
        return RefereeIntent(
            next_group_hint=next_group,
            next_actor_changed=False,
            table_state_after="closed",
            actor_group_after=actor_group,
            opponent_group_after=other_object_group(actor_group),
            ball_in_hand_scope=ball_in_hand_scope,  # type: ignore[arg-type]
            foul_flags=foul_flags,
            review_required=bool(shot_ctx.review_required),
            game_status=game_status,  # type: ignore[arg-type]
            reasons=reasons,
        )

    @staticmethod
    def _target_for_actor(actor: ObjectGroup, ledger: InventoryLedger) -> Optional[TargetGroup]:
        if int(ledger.remaining.get(actor, 0)) > 0:
            return actor
        if int(ledger.remaining.get("black", 0)) > 0:
            return "black"
        return None

    @staticmethod
    def _legal_first_contact_group(actor: Optional[ObjectGroup], ledger: InventoryLedger) -> Optional[str]:
        if actor is None:
            return None
        if int(ledger.remaining.get(actor, 0)) > 0:
            return actor
        return "black"


def shot_context_payload(ctx: ShotContext) -> dict[str, object]:
    return {
        "shot_id": ctx.shot_id,
        "ts_start_ms": ctx.ts_start_ms,
        "ts_end_ms": ctx.ts_end_ms,
        "break_shot": ctx.break_shot,
        "table_state_before": ctx.table_state_before,
        "actor_group": ctx.actor_group,
        "first_contact_group": ctx.first_contact_group,
        "first_contact_confidence": ctx.first_contact_confidence,
        "potted_confirmed": {group: int(ctx.potted_confirmed.get(group, 0)) for group in GROUPS},
        "off_table_confirmed": {group: int(ctx.off_table_confirmed.get(group, 0)) for group in GROUPS},
        "rail_contact_seen": ctx.rail_contact_seen,
        "cue_scratch_candidate": ctx.cue_scratch_candidate,
        "wrong_first_contact_candidate": ctx.wrong_first_contact_candidate,
        "review_required": ctx.review_required,
        "reasons": list(ctx.reasons),
    }
