from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..utils import unit


@dataclass(frozen=True)
class ResolvedCueDirectionPx:
    tip_px: np.ndarray
    tail_px: np.ndarray
    direction_px: np.ndarray
    score: float
    margin: float
    rear_support_px: float
    forward_overlap_px: float
    center_overlap_px: float
    status: str


@dataclass(frozen=True)
class _OrientationScore:
    tip_px: np.ndarray
    tail_px: np.ndarray
    direction_px: np.ndarray
    score: float
    rear_support_px: float
    forward_overlap_px: float
    center_overlap_px: float


class CueDirectionResolver:
    """Orients an undirected cue-stick axis using the cue-ball position.

    The valid strike direction is the half-ray that points away from the cue
    stick body. Along the correct orientation, the detected cue-stick segment
    should lie predominantly behind the cue ball, not in front of it.
    """

    def __init__(
        self,
        *,
        exclusion_radius_factor: float = 0.22,
        ambiguity_margin_px: float = 6.0,
    ) -> None:
        self.exclusion_radius_factor = float(exclusion_radius_factor)
        self.ambiguity_margin_px = float(ambiguity_margin_px)

    def resolve(
        self,
        *,
        cue_center_px: np.ndarray,
        cue_radius_px: float,
        p1_px: np.ndarray,
        p2_px: np.ndarray,
    ) -> Optional[ResolvedCueDirectionPx]:
        cue_center = np.asarray(cue_center_px, dtype=np.float32).reshape((2,))
        p1 = np.asarray(p1_px, dtype=np.float32).reshape((2,))
        p2 = np.asarray(p2_px, dtype=np.float32).reshape((2,))
        axis = unit(p2 - p1)
        if float(np.linalg.norm(p2 - p1)) < 1e-6:
            return None

        positive = self._score_orientation(
            cue_center=cue_center,
            cue_radius_px=float(cue_radius_px),
            p1_px=p1,
            p2_px=p2,
            forward_px=axis,
        )
        negative = self._score_orientation(
            cue_center=cue_center,
            cue_radius_px=float(cue_radius_px),
            p1_px=p1,
            p2_px=p2,
            forward_px=-axis,
        )
        if positive.score >= negative.score:
            best, other = positive, negative
        else:
            best, other = negative, positive
        margin = float(best.score - other.score)
        status = "body_side_strong" if margin >= self.ambiguity_margin_px else "body_side_ambiguous"
        return ResolvedCueDirectionPx(
            tip_px=best.tip_px,
            tail_px=best.tail_px,
            direction_px=best.direction_px,
            score=float(best.score),
            margin=margin,
            rear_support_px=float(best.rear_support_px),
            forward_overlap_px=float(best.forward_overlap_px),
            center_overlap_px=float(best.center_overlap_px),
            status=status,
        )

    def _score_orientation(
        self,
        *,
        cue_center: np.ndarray,
        cue_radius_px: float,
        p1_px: np.ndarray,
        p2_px: np.ndarray,
        forward_px: np.ndarray,
    ) -> _OrientationScore:
        forward = unit(forward_px)
        line_anchor = self._line_anchor(cue_center=cue_center, p1_px=p1_px, p2_px=p2_px)
        t1 = float(np.dot(p1_px - cue_center, forward))
        t2 = float(np.dot(p2_px - cue_center, forward))
        lo = min(t1, t2)
        hi = max(t1, t2)

        exclusion = max(2.0, float(cue_radius_px) * self.exclusion_radius_factor)
        rear_support = _overlap_length(lo, hi, -1.0e6, -exclusion)
        forward_overlap = _overlap_length(lo, hi, exclusion, 1.0e6)
        center_overlap = _overlap_length(lo, hi, -exclusion, exclusion)
        front_clearance = max(0.0, -hi)
        score = float(
            2.4 * rear_support
            - 4.8 * forward_overlap
            - 1.6 * center_overlap
            + 0.3 * front_clearance
        )

        tip_t = hi if hi <= -exclusion else (-exclusion if lo <= -exclusion else lo)
        tail_t = lo
        tip_px = (line_anchor + forward * tip_t).astype(np.float32)
        tail_px = (line_anchor + forward * tail_t).astype(np.float32)
        return _OrientationScore(
            tip_px=tip_px,
            tail_px=tail_px,
            direction_px=forward.astype(np.float32),
            score=score,
            rear_support_px=float(rear_support),
            forward_overlap_px=float(forward_overlap),
            center_overlap_px=float(center_overlap),
        )

    @staticmethod
    def _line_anchor(*, cue_center: np.ndarray, p1_px: np.ndarray, p2_px: np.ndarray) -> np.ndarray:
        seg = p2_px - p1_px
        denom = float(np.dot(seg, seg))
        if denom <= 1e-9:
            return p1_px.astype(np.float32)
        t = float(np.dot(cue_center - p1_px, seg) / denom)
        return (p1_px + seg * t).astype(np.float32)


def _overlap_length(lo: float, hi: float, window_lo: float, window_hi: float) -> float:
    start = max(float(lo), float(window_lo))
    end = min(float(hi), float(window_hi))
    return max(0.0, float(end - start))
