from __future__ import annotations

from typing import Any, Iterable


def resolve_potted_target_track_ids(plan: Any, labels: dict[str, Any] | None = None) -> set[int]:
    labels = labels or {}
    candidates = _candidate_rows(plan)
    if not candidates:
        return set()

    candidate_target_ids = {row["target_track_id"] for row in candidates}
    resolved = _int_set(labels.get("aligned_potted_track_ids") or [])
    if resolved:
        return resolved & candidate_target_ids

    raw_ids = _int_set(labels.get("potted_track_ids") or [])
    resolved |= raw_ids & candidate_target_ids

    for pot in labels.get("potted") or []:
        if not isinstance(pot, dict):
            continue
        raw_tid = _safe_int(pot.get("track_id"))
        if raw_tid is not None and raw_tid in candidate_target_ids:
            resolved.add(raw_tid)
            continue
        group = str(pot.get("group", "")).strip().lower()
        pocket_index = _safe_int(pot.get("pocket_index"))
        if not group or pocket_index is None:
            continue
        matched = {
            row["target_track_id"]
            for row in candidates
            if row["target_group"] == group and row["pocket_index"] == pocket_index
        }
        if len(matched) == 1:
            resolved |= matched
    return resolved


def build_alignment_labels(plan: Any, labels: dict[str, Any] | None = None) -> dict[str, Any]:
    labels = labels or {}
    resolved = resolve_potted_target_track_ids(plan, labels)
    raw_ids = _int_set(labels.get("potted_track_ids") or [])
    return {
        "aligned_potted_track_ids": sorted(resolved),
        "aligned_from_raw_match_count": sum(1 for tid in resolved if tid in raw_ids),
        "aligned_from_group_pocket_count": sum(1 for tid in resolved if tid not in raw_ids),
    }


def _candidate_rows(plan: Any) -> list[dict[str, int | str]]:
    source = plan.get("candidates") if isinstance(plan, dict) else getattr(plan, "candidates", [])
    rows: list[dict[str, int | str]] = []
    for candidate in source or []:
        target_track_id = _safe_int(_get(candidate, "target_track_id"))
        pocket_index = _safe_int(_get(candidate, "pocket_index"))
        target_group = str(_get(candidate, "target_group", "")).strip().lower()
        if target_track_id is None or pocket_index is None or not target_group:
            continue
        rows.append(
            {
                "target_track_id": target_track_id,
                "target_group": target_group,
                "pocket_index": pocket_index,
            }
        )
    return rows


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_set(values: Iterable[Any]) -> set[int]:
    out: set[int] = set()
    for value in values:
        parsed = _safe_int(value)
        if parsed is not None:
            out.add(parsed)
    return out


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
