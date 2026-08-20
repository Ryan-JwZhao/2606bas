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
    empty_group_counts,
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
        if any(
            event.name
            in {
                "SHOT_STARTED",
                "SHOT_START_VOTED",
                "POCKET_CANDIDATE",
                "POCKET_TENTATIVE",
                "POCKET_COMMIT_READY",
                "POCKET_CONFIRMED",
                "POCKET_REJECTED",
            }
            for event in event_list
        ):
            ctx = self.begin_if_needed(ts_ms=ts_ms, rule_state=rule_state)
        else:
            ctx = self._active
        if ctx is None:
            return
        for event in event_list:
            self._ingest_event(ctx, event)

    def finalize(self, *, ts_ms: int, rule_state: MatchRuleState) -> Optional[ShotContext]:
        ctx = self._active
        if ctx is None:
            return None
        ctx.ts_end_ms = ts_ms
        for pocket in ctx.tentative_pockets:
            rejected = dict(pocket)
            rejected["decision"] = "rejected"
            rejected["reason_codes"] = ["automatic_unresolved_tentative_reject"]
            ctx.rejected_pockets.append(rejected)
        if ctx.tentative_pockets:
            ctx.reasons.append("automatic_unresolved_tentative_reject")
            ctx.tentative_pockets = []
        self._active = None
        return ctx

    def _ingest_event(self, ctx: ShotContext, event: Event) -> None:
        payload = dict(event.payload or {})
        if event.name in {"SHOT_STARTED", "SHOT_START_VOTED"}:
            ctx.shot_started = True
        elif event.name == "POCKET_COMMIT_READY":
            self._clear_pending_pocket(ctx, payload)
            if not self._has_committed_pocket(ctx, payload):
                ctx.committed_pockets.append(_normalized_pocket_payload(payload))
                self._rebuild_potted_counts(ctx)
        elif event.name == "POCKET_CONFIRMED":
            self._clear_pending_pocket(ctx, payload)
            already_known = self._has_committed_pocket(ctx, payload)
            if not already_known:
                ctx.committed_pockets.append(_normalized_pocket_payload(payload))
                self._rebuild_potted_counts(ctx)
        elif event.name == "POCKET_TENTATIVE":
            ctx.tentative_pockets.append(_normalized_pocket_payload(payload))
        elif event.name == "POCKET_REJECTED":
            self._clear_pending_pocket(ctx, payload)
            self._remove_committed_pocket(ctx, payload)
            ctx.rejected_pockets.append(_normalized_pocket_payload(payload))
        elif event.name == "BALL_OFF_TABLE_CONFIRMED":
            group = normalize_group(payload.get("group"))
            if group is not None:
                ctx.off_table_confirmed[group] = int(ctx.off_table_confirmed.get(group, 0)) + 1
        elif event.name == "BALL_COLLISION_CANDIDATE":
            self._capture_first_contact(
                ctx,
                payload,
                confidence=float(event.confidence),
                ts_cam_ns=int(event.ts_cam_ns),
            )
        elif event.name == "RAIL_COLLISION_CANDIDATE":
            ctx.rail_contact_seen = True
            if ctx.first_contact_ts_ns is not None and int(event.ts_cam_ns) >= ctx.first_contact_ts_ns:
                ctx.rail_contact_after_first_seen = True
            if normalize_group(payload.get("group")) in {"solid", "stripe", "black"}:
                try:
                    ctx.rail_contact_track_ids.add(int(payload.get("track_id")))
                except (TypeError, ValueError):
                    pass
        elif event.name in {"POCKET_REAPPEARED", "BALL_LOST_UNCONFIRMED"}:
            ctx.reasons.append(event.name.lower())

    def _capture_first_contact(
        self,
        ctx: ShotContext,
        payload: dict[str, object],
        *,
        confidence: float,
        ts_cam_ns: int,
    ) -> None:
        if ctx.first_contact_group is not None:
            return
        groups = [normalize_group(payload.get("group_a")), normalize_group(payload.get("group_b"))]
        if any(group == "cue" for group in groups):
            other = next((group for group in groups if group is not None and group != "cue"), None)
            if other is not None:
                ctx.first_contact_group = other
                ctx.first_contact_confidence = confidence
                ctx.first_contact_ts_ns = int(ts_cam_ns)

    @staticmethod
    def _clear_pending_pocket(ctx: ShotContext, payload: dict[str, object]) -> None:
        decision_id = str(payload.get("decision_id") or "").strip()
        if not decision_id:
            return
        ctx.tentative_pockets = [item for item in ctx.tentative_pockets if str(item.get("decision_id") or "") != decision_id]

    @staticmethod
    def _has_committed_pocket(ctx: ShotContext, payload: dict[str, object]) -> bool:
        decision_id = str(payload.get("decision_id") or "").strip()
        return bool(
            decision_id
            and any(str(item.get("decision_id") or "").strip() == decision_id for item in ctx.committed_pockets)
        )

    @classmethod
    def _remove_committed_pocket(cls, ctx: ShotContext, payload: dict[str, object]) -> None:
        decision_id = str(payload.get("decision_id") or "").strip()
        if not decision_id:
            return
        retained = [
            item
            for item in ctx.committed_pockets
            if str(item.get("decision_id") or "").strip() != decision_id
        ]
        if len(retained) == len(ctx.committed_pockets):
            return
        ctx.committed_pockets = retained
        cls._rebuild_potted_counts(ctx)

    @staticmethod
    def _rebuild_potted_counts(ctx: ShotContext) -> None:
        counts = empty_group_counts()
        for pocket in ctx.committed_pockets:
            group = normalize_group(pocket.get("group"))
            if group is not None:
                counts[group] = int(counts.get(group, 0)) + 1
        ctx.potted_confirmed = counts
        ctx.cue_scratch_candidate = bool(counts.get("cue", 0))


class RefereeAdapter:
    """Structured referee interface. It exposes flags but keeps final judging conservative."""

    def evaluate(
        self,
        shot_ctx: ShotContext,
        ledger: InventoryLedger,
        rule_state: MatchRuleState,
        *,
        effective_remaining: dict[str, int] | None = None,
        observation_reasons: list[str] | None = None,
        ledger_before: InventoryLedger | None = None,
        legal_first_group: str | None = None,
    ) -> RefereeIntent:
        actor = rule_state.actor_group
        opponent = rule_state.opponent_group
        effective = effective_remaining or {}
        before = ledger_before or ledger
        explicit_legal_first = normalize_group(legal_first_group)
        legal_first = (
            explicit_legal_first
            if actor is not None and explicit_legal_first in {"solid", "stripe", "black"}
            else self._legal_first_contact_group(actor, before)
        )
        open_table_black_first = bool(
            actor is None
            and not shot_ctx.break_shot
            and shot_ctx.first_contact_group == "black"
        )
        object_potted = sum(int(shot_ctx.potted_confirmed.get(group, 0)) for group in ("solid", "stripe", "black"))
        no_rail_after_contact = bool(
            not shot_ctx.break_shot
            and shot_ctx.first_contact_group is not None
            and object_potted <= 0
            and not (
                shot_ctx.rail_contact_after_first_seen
                or (shot_ctx.first_contact_ts_ns is None and shot_ctx.rail_contact_seen)
            )
        )
        break_rail_count = len(shot_ctx.rail_contact_track_ids)
        legal_break_action = bool(object_potted > 0 or break_rail_count >= 4)
        foul_flags = {
            "cue_scratch": bool(shot_ctx.cue_scratch_candidate or shot_ctx.potted_confirmed.get("cue", 0) > 0),
            "wrong_first_contact": bool(
                open_table_black_first
                or (
                    legal_first is not None
                    and shot_ctx.first_contact_group is not None
                    and shot_ctx.first_contact_group != legal_first
                )
            ),
            "no_rail_after_contact": no_rail_after_contact,
            "break_foul": False,
        }
        foul_flags["break_foul"] = bool(
            shot_ctx.break_shot and (foul_flags["cue_scratch"] or not legal_break_action)
        )
        foul = bool(
            foul_flags["cue_scratch"]
            or foul_flags["wrong_first_contact"]
            or foul_flags["no_rail_after_contact"]
            or foul_flags["break_foul"]
        )
        ball_in_hand_scope = "behind_head_string" if shot_ctx.break_shot and foul else "table_anywhere" if foul else "none"
        reasons: List[str] = list(shot_ctx.reasons)
        reasons.extend(observation_reasons or [])
        if foul_flags["cue_scratch"]:
            reasons.append("cue_scratch")
        if foul_flags["wrong_first_contact"]:
            reasons.append("wrong_first_contact")
        if foul_flags["no_rail_after_contact"]:
            reasons.append("no_rail_after_contact")
        if shot_ctx.break_shot and not legal_break_action:
            reasons.append("illegal_break_insufficient_rail_or_pot")

        black_potted = shot_ctx.potted_confirmed.get("black", 0) > 0
        game_status = rule_state.game_status
        game_outcome = rule_state.game_outcome
        winner_group = rule_state.winner_group
        if black_potted:
            game_status = "ended"
            actor_ready_for_black = bool(actor is not None and legal_first == "black")
            if actor_ready_for_black and not foul:
                game_outcome = "legal_black_win"
                winner_group = actor
                reasons.append("legal_black_potted")
            else:
                game_outcome = "illegal_black_loss"
                winner_group = opponent
                reasons.append("illegal_black_potted")

        if rule_state.table_state == "open":
            return self._evaluate_open_table(
                shot_ctx,
                ledger,
                foul_flags,
                ball_in_hand_scope,
                game_status,
                game_outcome,
                winner_group,
                reasons,
                effective,
            )

        if actor is None or opponent is None:
            return RefereeIntent(
                next_group_hint=None,
                next_actor_changed=True,
                table_state_after="open",
                actor_group_after=None,
                opponent_group_after=None,
                ball_in_hand_scope=ball_in_hand_scope,
                foul_flags=foul_flags,
                game_status=game_status,
                game_outcome=game_outcome,
                winner_group=winner_group,
                reasons=reasons + ["closed_table_missing_group_owner_auto_open"],
            )

        keep_turn = (not foul) and shot_ctx.potted_confirmed.get(actor, 0) > 0
        next_actor = actor if keep_turn else opponent
        next_group = self._target_for_actor(next_actor, ledger, effective)
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
            game_status=game_status,
            game_outcome=game_outcome,
            winner_group=winner_group,
            reasons=reasons,
        )

    def _evaluate_open_table(
        self,
        shot_ctx: ShotContext,
        ledger: InventoryLedger,
        foul_flags: dict[str, bool],
        ball_in_hand_scope: str,
        game_status: str,
        game_outcome: str,
        winner_group: Optional[ObjectGroup],
        reasons: list[str],
        effective_remaining: dict[str, int] | None = None,
    ) -> RefereeIntent:
        object_potted = {
            group: int(shot_ctx.potted_confirmed.get(group, 0))
            for group in ("solid", "stripe")
            if shot_ctx.potted_confirmed.get(group, 0) > 0
        }
        foul = bool(
            foul_flags["cue_scratch"]
            or foul_flags["wrong_first_contact"]
            or foul_flags["no_rail_after_contact"]
            or foul_flags["break_foul"]
        )
        if game_status != "in_progress" or shot_ctx.break_shot or foul or not object_potted:
            return RefereeIntent(
                next_group_hint=None,
                next_actor_changed=not bool(object_potted and not foul),
                table_state_after="open",
                actor_group_after=None,
                opponent_group_after=None,
                ball_in_hand_scope=ball_in_hand_scope,  # type: ignore[arg-type]
                foul_flags=foul_flags,
                game_status=game_status,  # type: ignore[arg-type]
                game_outcome=game_outcome,  # type: ignore[arg-type]
                winner_group=winner_group,
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
                game_status=game_status,  # type: ignore[arg-type]
                game_outcome=game_outcome,  # type: ignore[arg-type]
                winner_group=winner_group,
                reasons=reasons + ["open_table_multiple_groups_keep_open"],
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
                game_status=game_status,  # type: ignore[arg-type]
                game_outcome=game_outcome,  # type: ignore[arg-type]
                winner_group=winner_group,
                reasons=reasons + ["open_table_invalid_group_keep_open"],
            )
        next_group = self._target_for_actor(actor_group, ledger, effective_remaining)
        return RefereeIntent(
            next_group_hint=next_group,
            next_actor_changed=False,
            table_state_after="closed",
            actor_group_after=actor_group,
            opponent_group_after=other_object_group(actor_group),
            ball_in_hand_scope=ball_in_hand_scope,  # type: ignore[arg-type]
            foul_flags=foul_flags,
            game_status=game_status,  # type: ignore[arg-type]
            game_outcome=game_outcome,  # type: ignore[arg-type]
            winner_group=winner_group,
            reasons=reasons,
        )

    @staticmethod
    def _target_for_actor(
        actor: ObjectGroup,
        ledger: InventoryLedger,
        effective_remaining: dict[str, int] | None = None,
    ) -> Optional[TargetGroup]:
        if RefereeAdapter._remaining(ledger, effective_remaining, actor) > 0:
            return actor
        if RefereeAdapter._remaining(ledger, effective_remaining, "black") > 0:
            return "black"
        return None

    @staticmethod
    def _legal_first_contact_group(
        actor: Optional[ObjectGroup],
        ledger: InventoryLedger,
        effective_remaining: dict[str, int] | None = None,
    ) -> Optional[str]:
        if actor is None:
            return None
        if RefereeAdapter._remaining(ledger, effective_remaining, actor) > 0:
            return actor
        return "black"

    @staticmethod
    def _remaining(ledger: InventoryLedger, effective_remaining: dict[str, int] | None, group: str) -> int:
        if effective_remaining is not None and group in effective_remaining:
            return int(effective_remaining.get(group, 0))
        return int(ledger.remaining.get(group, 0))


def shot_context_payload(ctx: ShotContext) -> dict[str, object]:
    return {
        "shot_id": ctx.shot_id,
        "ts_start_ms": ctx.ts_start_ms,
        "ts_end_ms": ctx.ts_end_ms,
        "break_shot": ctx.break_shot,
        "table_state_before": ctx.table_state_before,
        "actor_group": ctx.actor_group,
        "legal_first_group": ctx.legal_first_group,
        "shot_started": ctx.shot_started,
        "first_contact_group": ctx.first_contact_group,
        "first_contact_confidence": ctx.first_contact_confidence,
        "first_contact_ts_ns": ctx.first_contact_ts_ns,
        "potted_confirmed": {group: int(ctx.potted_confirmed.get(group, 0)) for group in GROUPS},
        "off_table_confirmed": {group: int(ctx.off_table_confirmed.get(group, 0)) for group in GROUPS},
        "committed_pockets": list(ctx.committed_pockets),
        "tentative_pockets": list(ctx.tentative_pockets),
        "rejected_pockets": list(ctx.rejected_pockets),
        "rail_contact_seen": ctx.rail_contact_seen,
        "rail_contact_after_first_seen": ctx.rail_contact_after_first_seen,
        "rail_contact_track_ids": sorted(ctx.rail_contact_track_ids),
        "cue_scratch_candidate": ctx.cue_scratch_candidate,
        "wrong_first_contact_candidate": ctx.wrong_first_contact_candidate,
        "reasons": list(ctx.reasons),
    }


def _normalized_pocket_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        "track_id": payload.get("track_id"),
        "logical_id": payload.get("logical_id"),
        "group": payload.get("group"),
        "pocket_index": payload.get("pocket_index"),
        "shot_id": payload.get("shot_id"),
        "decision_id": payload.get("decision_id"),
        "decision": payload.get("decision"),
        "reason_codes": list(payload.get("reason_codes") or []),
        "evidence": dict(payload.get("evidence") or {}),
    }
