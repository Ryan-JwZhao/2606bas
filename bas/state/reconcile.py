from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..config import StateConfig
from ..schemas import Event, TrackObservation, TracksFrame
from .models import GROUPS, Group, InventoryLedger, empty_group_counts, normalize_group


@dataclass
class ObservationReconcileResult:
    visible_counts: dict[Group, int] = field(default_factory=empty_group_counts)
    stable_frames: dict[Group, int] = field(default_factory=empty_group_counts)
    effective_remaining: dict[Group, int] = field(default_factory=dict)
    mismatches: list[dict[str, object]] = field(default_factory=list)


class ObservationReconciler:
    """Compares the event ledger with long-running YOLO observations.

    Stable observations provide a planning view and can restore balls that a
    previous false-positive pocket decision removed.  Missing observations do
    not decrement the ledger without a confirmed pocket event.
    """

    def __init__(self, config: StateConfig):
        self.config = config
        self._counts: dict[Group, int] = empty_group_counts()
        self._stable_frames: dict[Group, int] = empty_group_counts()
        self._last_result = ObservationReconcileResult()

    def reset(self) -> None:
        self._counts = empty_group_counts()
        self._stable_frames = empty_group_counts()
        self._last_result = ObservationReconcileResult()

    @property
    def last_result(self) -> ObservationReconcileResult:
        return self._last_result

    def current_observation_result(self, ledger: InventoryLedger | None = None) -> ObservationReconcileResult:
        effective = (
            {group: int(ledger.remaining.get(group, 0)) for group in GROUPS}
            if ledger is not None
            else {group: int(self._last_result.effective_remaining.get(group, 0)) for group in GROUPS}
        )
        if ledger is not None:
            corrected_groups = {str(item.get("group") or "") for item in self._last_result.mismatches}
            for group in corrected_groups:
                if group in effective and group in self._last_result.effective_remaining:
                    effective[group] = int(self._last_result.effective_remaining[group])
        return ObservationReconcileResult(
            visible_counts={group: int(self._counts.get(group, 0)) for group in GROUPS},
            stable_frames={group: int(self._stable_frames.get(group, 0)) for group in GROUPS},
            effective_remaining=effective,
            mismatches=list(self._last_result.mismatches),
        )

    def update_observation(self, frame: TracksFrame) -> None:
        counts = empty_group_counts()
        min_quality = float(getattr(self.config, "observation_reconcile_min_quality", 0.45))
        min_confidence = float(getattr(self.config, "observation_reconcile_min_confidence", 0.35))
        for track in frame.tracks:
            if not self._track_eligible(track, min_quality=min_quality, min_confidence=min_confidence):
                continue
            group = normalize_group(track.group)
            if group is not None:
                counts[group] += 1
        for group in GROUPS:
            if counts[group] == self._counts[group]:
                self._stable_frames[group] += 1
            else:
                self._stable_frames[group] = 1
        self._counts = counts

    def reconcile(
        self,
        ledger: InventoryLedger,
        *,
        supporting_events: Iterable[Event] = (),
    ) -> ObservationReconcileResult:
        effective = {group: int(ledger.remaining.get(group, 0)) for group in GROUPS}
        result = ObservationReconcileResult(
            visible_counts={group: int(self._counts.get(group, 0)) for group in GROUPS},
            stable_frames={group: int(self._stable_frames.get(group, 0)) for group in GROUPS},
            effective_remaining=dict(effective),
        )
        if not bool(getattr(self.config, "observation_reconcile_enabled", True)):
            self._last_result = result
            return result

        event_groups = self._supporting_event_groups(supporting_events)
        stable_needed = max(1, int(getattr(self.config, "observation_reconcile_stable_frames", 12)))
        infer_missing = bool(getattr(self.config, "observation_reconcile_infer_missing_with_event", True))
        for group in ("solid", "stripe", "black"):
            observed = int(self._counts[group])
            ledger_count = int(ledger.remaining.get(group, 0))
            if self._stable_frames[group] < stable_needed:
                continue
            if observed > ledger_count:
                effective[group] = observed
                result.mismatches.append(
                    {
                        "group": group,
                        "mode": "visible_exceeds_ledger",
                        "ledger_remaining": ledger_count,
                        "visible_count": observed,
                        "stable_frames": int(self._stable_frames[group]),
                    }
                )
            elif infer_missing and observed < ledger_count and group in event_groups:
                effective[group] = observed
                result.mismatches.append(
                    {
                        "group": group,
                        "mode": "event_supported_visible_below_ledger",
                        "ledger_remaining": ledger_count,
                        "visible_count": observed,
                        "stable_frames": int(self._stable_frames[group]),
                    }
                )
        result.effective_remaining = effective
        self._last_result = result
        return result

    @staticmethod
    def apply_automatic_restorations(
        ledger: InventoryLedger,
        result: ObservationReconcileResult,
    ) -> list[dict[str, object]]:
        """Restore only provable false removals; never infer a pot from absence."""

        corrections: list[dict[str, object]] = []
        maximum = {"solid": 7, "stripe": 7, "black": 1}
        for mismatch in result.mismatches:
            if str(mismatch.get("mode") or "") != "visible_exceeds_ledger":
                continue
            group = normalize_group(mismatch.get("group"))
            if group not in {"solid", "stripe", "black"}:
                continue
            before = int(ledger.remaining.get(group, 0))
            observed = min(maximum[group], max(0, int(mismatch.get("visible_count", before))))
            if observed <= before:
                continue
            restored = observed - before
            ledger.remaining[group] = observed
            ledger.removed_confirmed[group] = max(0, int(ledger.removed_confirmed.get(group, 0)) - restored)
            corrections.append(
                {
                    "group": group,
                    "before": before,
                    "after": observed,
                    "restored": restored,
                    "reason": "stable_visible_exceeds_ledger",
                }
            )
        return corrections

    def event_payload(self, result: ObservationReconcileResult) -> dict[str, object]:
        return {
            "visible_counts": {group: int(result.visible_counts.get(group, 0)) for group in GROUPS},
            "stable_frames": {group: int(result.stable_frames.get(group, 0)) for group in GROUPS},
            "effective_remaining": {group: int(result.effective_remaining.get(group, 0)) for group in GROUPS},
            "mismatches": list(result.mismatches),
        }

    @staticmethod
    def _track_eligible(track: TrackObservation, *, min_quality: float, min_confidence: float) -> bool:
        return (
            track.visibility == "visible"
            and float(track.quality) >= min_quality
            and float(track.confidence) >= min_confidence
            and normalize_group(track.group) is not None
        )

    @staticmethod
    def _supporting_event_groups(events: Iterable[Event]) -> set[Group]:
        groups: set[Group] = set()
        supporting_names = {
            "POCKET_CANDIDATE",
            "POCKET_TENTATIVE",
            "POCKET_COMMIT_READY",
            "POCKET_DETECTED",
            "POCKET_CONFIRMED",
            "POCKET_REJECTED",
            "BALL_LOST_UNCONFIRMED",
            "BALL_OFF_TABLE_CONFIRMED",
            "POT_PROBABLE",
        }
        for event in events:
            if event.name not in supporting_names:
                continue
            group = normalize_group((event.payload or {}).get("group"))
            if group is not None:
                groups.add(group)
        return groups
