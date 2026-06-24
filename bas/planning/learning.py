from __future__ import annotations

import json
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np

from ..schemas import ShotCandidate

LOGGER = logging.getLogger(__name__)


FEATURE_NAMES: list[str] = [
    "base_score",
    "base_risk",
    "cut_angle_deg",
    "cue_distance_mm",
    "object_distance_mm",
    "cue_clearance_mm",
    "object_clearance_mm",
    "cut_penalty",
    "distance_penalty",
    "pocket_angle_penalty",
    "cue_x_norm",
    "cue_y_norm",
    "object_x_norm",
    "object_y_norm",
    "ghost_x_norm",
    "ghost_y_norm",
    "pocket_x_norm",
    "pocket_y_norm",
    "pocket_index_norm",
    "ball_count",
    "target_count",
    "blocker_count",
    "cue_visible",
    "layout_quality_mean",
    "target_group_solid",
    "target_group_stripe",
    "target_group_black",
]


class LearningFeatureBuilder:
    def __init__(self, table_width_mm: float = 2540.0, table_height_mm: float = 1270.0):
        self.table_width_mm = max(1.0, float(table_width_mm))
        self.table_height_mm = max(1.0, float(table_height_mm))

    def feature_map(self, candidate: Any, state: Any | None = None) -> dict[str, float]:
        explanation = _as_dict(_get(candidate, "explanation", {}))
        cue = _point(_get(candidate, "cue_ball", (0.0, 0.0)))
        obj = _point(_get(candidate, "object_ball", (0.0, 0.0)))
        ghost = _point(_get(candidate, "ghost_ball", (0.0, 0.0)))
        pocket = _point(_get(candidate, "pocket_point", (0.0, 0.0)))
        layout = list(_layout_tracks(state))
        target_group = str(_get(candidate, "target_group", ""))
        ball_tracks = [t for t in layout if str(_get(t, "group", "")) in {"cue", "solid", "stripe", "black"}]
        target_tracks = [t for t in layout if str(_get(t, "group", "")) in {"solid", "stripe", "black"}]
        quality_values = [_safe_float(_get(t, "quality", 0.0), 0.0) for t in ball_tracks]
        quality_mean = float(sum(quality_values) / len(quality_values)) if quality_values else 0.0
        ball_count = float(len(ball_tracks))
        target_count = float(len(target_tracks))
        return {
            "base_score": _safe_float(_get(candidate, "score", 0.0), 0.0),
            "base_risk": _safe_float(_get(candidate, "risk", 0.0), 0.0),
            "cut_angle_deg": _safe_float(_get(candidate, "cut_angle_deg", 0.0), 0.0),
            "cue_distance_mm": _safe_float(_get(candidate, "cue_distance_mm", 0.0), 0.0),
            "object_distance_mm": _safe_float(_get(candidate, "object_distance_mm", 0.0), 0.0),
            "cue_clearance_mm": _safe_float(explanation.get("cue_clearance_mm", 0.0), 0.0),
            "object_clearance_mm": _safe_float(explanation.get("object_clearance_mm", 0.0), 0.0),
            "cut_penalty": _safe_float(explanation.get("cut_penalty", 0.0), 0.0),
            "distance_penalty": _safe_float(explanation.get("distance_penalty", 0.0), 0.0),
            "pocket_angle_penalty": _safe_float(explanation.get("pocket_angle_penalty", 0.0), 0.0),
            "cue_x_norm": cue[0] / self.table_width_mm,
            "cue_y_norm": cue[1] / self.table_height_mm,
            "object_x_norm": obj[0] / self.table_width_mm,
            "object_y_norm": obj[1] / self.table_height_mm,
            "ghost_x_norm": ghost[0] / self.table_width_mm,
            "ghost_y_norm": ghost[1] / self.table_height_mm,
            "pocket_x_norm": pocket[0] / self.table_width_mm,
            "pocket_y_norm": pocket[1] / self.table_height_mm,
            "pocket_index_norm": _safe_float(_get(candidate, "pocket_index", 0), 0.0) / 5.0,
            "ball_count": ball_count,
            "target_count": target_count,
            "blocker_count": max(0.0, ball_count - 2.0),
            "cue_visible": 1.0 if any(str(_get(t, "group", "")) == "cue" for t in ball_tracks) else 0.0,
            "layout_quality_mean": quality_mean,
            "target_group_solid": 1.0 if target_group == "solid" else 0.0,
            "target_group_stripe": 1.0 if target_group == "stripe" else 0.0,
            "target_group_black": 1.0 if target_group == "black" else 0.0,
        }

    def vector(self, candidate: Any, state: Any | None = None, feature_names: Sequence[str] | None = None) -> np.ndarray:
        values = self.feature_map(candidate, state)
        names = list(feature_names or FEATURE_NAMES)
        return np.asarray([_safe_float(values.get(name, 0.0), 0.0) for name in names], dtype=np.float32)


class DisabledLearningRanker:
    """Default ranker used when no frozen learning model is configured."""

    version = "learning_disabled"

    def rerank(self, candidates: List[ShotCandidate], state: Any | None = None) -> List[ShotCandidate]:
        return sorted(candidates, key=lambda c: c.score, reverse=True)


class JsonLearningRanker:
    def __init__(
        self,
        model_path: str | Path,
        *,
        table_width_mm: float,
        table_height_mm: float,
        score_blend: float = 0.65,
    ):
        self.model_path = Path(model_path)
        with self.model_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Learning model must be a JSON object.")
        self.format = str(payload.get("format", ""))
        if self.format not in {"bas_mlp_ranker_v1", "bas_linear_ranker_v1"}:
            raise ValueError(f"Unsupported learning model format: {self.format}")
        self.model_version = str(payload.get("model_version") or self.model_path.stem)
        self.feature_names = [str(name) for name in payload.get("feature_names", FEATURE_NAMES)]
        self.output_names = [str(name) for name in payload.get("output_names", ["rank_score"])]
        self.score_weights = dict(payload.get("score_weights") or {})
        self.score_blend = _clamp(_safe_float(score_blend, 0.65), 0.0, 1.0)
        self.builder = LearningFeatureBuilder(table_width_mm=table_width_mm, table_height_mm=table_height_mm)
        self._mean, self._std = self._normalization(payload.get("normalization"))
        self._layers = list(payload.get("layers") or [])
        self._linear_weights = payload.get("weights")
        self._linear_bias = payload.get("bias", 0.0)
        if self.format == "bas_mlp_ranker_v1" and not self._layers:
            raise ValueError("MLP learning model has no layers.")
        if self.format == "bas_linear_ranker_v1" and self._linear_weights is None:
            raise ValueError("Linear learning model has no weights.")
        self.version = f"learning_json:{self.model_version}"

    def rerank(self, candidates: List[ShotCandidate], state: Any | None = None) -> List[ShotCandidate]:
        ranked: list[ShotCandidate] = []
        for candidate in candidates:
            try:
                ranked.append(self._apply(candidate, state))
            except Exception as exc:
                LOGGER.debug("Learning ranker skipped candidate %s: %s", candidate.candidate_id, exc)
                ranked.append(candidate)
        return sorted(ranked, key=lambda c: c.score, reverse=True)

    def _apply(self, candidate: ShotCandidate, state: Any | None) -> ShotCandidate:
        feature_map = self.builder.feature_map(candidate, state)
        x = np.asarray([feature_map.get(name, 0.0) for name in self.feature_names], dtype=np.float32)
        y = self._predict((x - self._mean) / self._std)
        outputs = {name: _safe_float(value, 0.0) for name, value in zip(self.output_names, y.tolist())}
        learned_score, detail = self._learned_score(outputs, feature_map)
        base_score = _safe_float(candidate.score, 0.0)
        final_score = (1.0 - self.score_blend) * base_score + self.score_blend * learned_score
        scratch = detail.get("scratch_probability")
        foul = detail.get("foul_probability")
        learned_risk = max(candidate.risk, scratch if scratch is not None else 0.0, foul if foul is not None else 0.0)
        explanation = dict(candidate.explanation)
        explanation.update(
            {
                "learning_ranker": self.version,
                "learning_model_path": str(self.model_path),
                "learning_base_score": base_score,
                "learning_score": float(learned_score),
                "learning_score_blend": float(self.score_blend),
                **{k: float(v) for k, v in detail.items() if v is not None},
            }
        )
        return replace(
            candidate,
            score=float(final_score),
            risk=float(_clamp(learned_risk, 0.0, 1.0)),
            explanation=explanation,
        )

    def _predict(self, x: np.ndarray) -> np.ndarray:
        if self.format == "bas_linear_ranker_v1":
            weights = np.asarray(self._linear_weights, dtype=np.float32)
            bias = np.asarray(self._linear_bias, dtype=np.float32)
            return np.atleast_1d(np.matmul(weights, x) + bias).astype(np.float32)
        a = x.astype(np.float32)
        for index, layer in enumerate(self._layers):
            weight = np.asarray(layer["weight"], dtype=np.float32)
            bias = np.asarray(layer.get("bias", np.zeros(weight.shape[0], dtype=np.float32)), dtype=np.float32)
            a = np.matmul(a, weight.T) + bias
            if index < len(self._layers) - 1:
                a = np.maximum(a, 0.0)
        return np.atleast_1d(a).astype(np.float32)

    def _learned_score(self, outputs: dict[str, float], feature_map: dict[str, float]) -> tuple[float, dict[str, float | None]]:
        pot = _prob(outputs, "pot")
        scratch = _prob(outputs, "scratch")
        foul = _prob(outputs, "foul")
        rank = _prob(outputs, "rank")
        leave = outputs.get("leave_score", outputs.get("leave"))
        residual_x = outputs.get("residual_x", 0.0)
        residual_y = outputs.get("residual_y", 0.0)
        residual = float(math.hypot(_safe_float(residual_x, 0.0), _safe_float(residual_y, 0.0)))
        score = 0.0
        score += _weight(self.score_weights, "pot", 1.2) * (pot if pot is not None else 0.0)
        score -= _weight(self.score_weights, "scratch", 1.1) * (scratch if scratch is not None else 0.0)
        score -= _weight(self.score_weights, "foul", 0.9) * (foul if foul is not None else 0.0)
        score += _weight(self.score_weights, "leave", 0.4) * (_safe_float(leave, 0.0) if leave is not None else 0.0)
        score += _weight(self.score_weights, "rank", 0.8) * (rank if rank is not None else 0.0)
        score -= _weight(self.score_weights, "risk", 0.5) * _safe_float(feature_map.get("base_risk", 0.0), 0.0)
        score -= _weight(self.score_weights, "residual", 0.1) * residual
        return float(score), {
            "pot_probability": pot,
            "scratch_probability": scratch,
            "foul_probability": foul,
            "rank_probability": rank,
            "leave_score": _safe_float(leave, 0.0) if leave is not None else None,
            "residual_norm": residual,
        }

    def _normalization(self, value: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(value, dict):
            return np.zeros(len(self.feature_names), dtype=np.float32), np.ones(len(self.feature_names), dtype=np.float32)
        mean = np.asarray(value.get("mean", []), dtype=np.float32)
        std = np.asarray(value.get("std", []), dtype=np.float32)
        if mean.shape[0] != len(self.feature_names):
            mean = np.zeros(len(self.feature_names), dtype=np.float32)
        if std.shape[0] != len(self.feature_names):
            std = np.ones(len(self.feature_names), dtype=np.float32)
        std = np.where(np.abs(std) < 1e-6, 1.0, std)
        return mean, std


def create_learning_ranker(
    config: Any | None,
    *,
    table_width_mm: float,
    table_height_mm: float,
) -> DisabledLearningRanker | JsonLearningRanker:
    if config is None or not bool(getattr(config, "ranker_enabled", True)):
        return DisabledLearningRanker()
    model_path = getattr(config, "ranker_model_path", None)
    if not model_path:
        return DisabledLearningRanker()
    path = Path(str(model_path))
    if not path.exists():
        LOGGER.warning("Learning ranker model does not exist: %s", path)
        return DisabledLearningRanker()
    try:
        return JsonLearningRanker(
            path,
            table_width_mm=table_width_mm,
            table_height_mm=table_height_mm,
            score_blend=getattr(config, "score_blend", 0.65),
        )
    except Exception as exc:
        LOGGER.warning("Failed to load learning ranker model %s: %s", path, exc)
        return DisabledLearningRanker()


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _point(value: Any) -> tuple[float, float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (_safe_float(value[0], 0.0), _safe_float(value[1], 0.0))
    return (0.0, 0.0)


def _layout_tracks(state: Any | None) -> Iterable[Any]:
    if state is None:
        return []
    layout = _get(state, "layout", [])
    return layout if isinstance(layout, list) else []


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _sigmoid(value: float) -> float:
    value = _clamp(value, -60.0, 60.0)
    return float(1.0 / (1.0 + math.exp(-value)))


def _prob(outputs: dict[str, float], prefix: str) -> float | None:
    probability_key = f"{prefix}_probability"
    if probability_key in outputs:
        return _clamp(outputs[probability_key], 0.0, 1.0)
    logit_key = f"{prefix}_logit"
    if logit_key in outputs:
        return _sigmoid(outputs[logit_key])
    score_key = f"{prefix}_score"
    if score_key in outputs:
        return _sigmoid(outputs[score_key])
    if prefix in outputs:
        return _sigmoid(outputs[prefix])
    return None


def _weight(weights: dict[str, Any], name: str, default: float) -> float:
    return _safe_float(weights.get(name, default), default)
