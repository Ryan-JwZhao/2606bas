from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from bas.planning.learning import FEATURE_NAMES, LearningFeatureBuilder


@dataclass
class TrainingData:
    features: np.ndarray
    pot: np.ndarray
    scratch: np.ndarray
    foul: np.ndarray
    leave: np.ndarray
    rank: np.ndarray
    sample_ids: list[str]
    candidate_ids: list[str]
    feature_names: list[str]

    @property
    def target_matrix(self) -> np.ndarray:
        return np.stack([self.pot, self.scratch, self.foul, self.leave, self.rank], axis=1).astype(np.float32)


def iter_sample_files(paths: str | Path | Sequence[str | Path]) -> Iterator[Path]:
    roots: Iterable[str | Path]
    if isinstance(paths, (str, Path)):
        roots = [paths]
    else:
        roots = paths
    seen: set[Path] = set()
    for item in roots:
        path = Path(item)
        candidates = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen and candidate.exists():
                seen.add(resolved)
                yield candidate


def iter_samples(paths: str | Path | Sequence[str | Path]) -> Iterator[dict[str, Any]]:
    for path in iter_sample_files(paths):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if isinstance(sample, dict) and sample.get("format") == "bas_shot_sample_v1":
                    yield sample


def build_training_data(
    sample_paths: str | Path | Sequence[str | Path],
    *,
    table_width_mm: float = 2540.0,
    table_height_mm: float = 1270.0,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> TrainingData:
    builder = LearningFeatureBuilder(table_width_mm=table_width_mm, table_height_mm=table_height_mm)
    features: list[np.ndarray] = []
    pot_targets: list[float] = []
    scratch_targets: list[float] = []
    foul_targets: list[float] = []
    leave_targets: list[float] = []
    rank_targets: list[float] = []
    sample_ids: list[str] = []
    candidate_ids: list[str] = []
    for sample in iter_samples(sample_paths):
        labels = sample.get("labels") or {}
        plan = sample.get("plan") or {}
        candidates = plan.get("candidates") or []
        if not isinstance(candidates, list) or not candidates:
            continue
        potted_ids = _int_set(labels.get("potted_track_ids") or [])
        scratch = 1.0 if bool(labels.get("scratch")) else 0.0
        foul = 1.0 if bool(labels.get("foul")) else 0.0
        leave = _estimate_leave_score(sample)
        for candidate in candidates:
            target_id = _safe_int(candidate.get("target_track_id"))
            pot = 1.0 if target_id is not None and target_id in potted_ids else 0.0
            valid_success = 1.0 if pot > 0.5 and scratch < 0.5 and foul < 0.5 else 0.0
            features.append(builder.vector(candidate, sample.get("pre_state"), feature_names))
            pot_targets.append(pot)
            scratch_targets.append(scratch)
            foul_targets.append(foul)
            leave_targets.append(leave)
            rank_targets.append(valid_success)
            sample_ids.append(str(sample.get("sample_id") or "unknown"))
            candidate_ids.append(str(candidate.get("candidate_id") or "unknown"))
    if not features:
        empty = np.zeros((0, len(feature_names)), dtype=np.float32)
        zeros = np.zeros((0,), dtype=np.float32)
        return TrainingData(empty, zeros, zeros, zeros, zeros, zeros, [], [], list(feature_names))
    return TrainingData(
        features=np.vstack(features).astype(np.float32),
        pot=np.asarray(pot_targets, dtype=np.float32),
        scratch=np.asarray(scratch_targets, dtype=np.float32),
        foul=np.asarray(foul_targets, dtype=np.float32),
        leave=np.asarray(leave_targets, dtype=np.float32),
        rank=np.asarray(rank_targets, dtype=np.float32),
        sample_ids=sample_ids,
        candidate_ids=candidate_ids,
        feature_names=list(feature_names),
    )


def summarize_samples(sample_paths: str | Path | Sequence[str | Path]) -> dict[str, Any]:
    sample_count = 0
    candidate_count = 0
    positive_pot = 0
    scratch_count = 0
    foul_count = 0
    for sample in iter_samples(sample_paths):
        sample_count += 1
        labels = sample.get("labels") or {}
        plan = sample.get("plan") or {}
        candidates = plan.get("candidates") or []
        candidate_count += len(candidates) if isinstance(candidates, list) else 0
        potted_ids = _int_set(labels.get("potted_track_ids") or [])
        scratch_count += 1 if labels.get("scratch") else 0
        foul_count += 1 if labels.get("foul") else 0
        if isinstance(candidates, list):
            for candidate in candidates:
                target_id = _safe_int(candidate.get("target_track_id"))
                positive_pot += 1 if target_id is not None and target_id in potted_ids else 0
    return {
        "samples": sample_count,
        "candidates": candidate_count,
        "positive_pot_candidates": positive_pot,
        "scratch_samples": scratch_count,
        "foul_samples": foul_count,
    }


def _estimate_leave_score(sample: dict[str, Any]) -> float:
    labels = sample.get("labels") or {}
    if labels.get("foul") or labels.get("scratch"):
        return -1.0
    before = _target_count(sample.get("pre_state") or {})
    after = _target_count(sample.get("end_state") or {})
    if before <= 0:
        return 0.0
    potted_bonus = max(0.0, float(before - after)) / max(1.0, float(before))
    return float(np.clip(potted_bonus, -1.0, 1.0))


def _target_count(state: dict[str, Any]) -> int:
    layout = state.get("layout") or []
    if not isinstance(layout, list):
        return 0
    return sum(1 for item in layout if str(item.get("group", "")) in {"solid", "stripe", "black"})


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
