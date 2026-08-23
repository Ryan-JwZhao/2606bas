from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import PlannerConfig
from ..utils import angle_deg, unit


@dataclass(frozen=True)
class CueDirectionDecision:
    direction_px: tuple[float, float]
    direction_mm: tuple[float, float]
    status: str


class CueDirectionStabilizer:
    """Keeps cue aiming direction stable during backstroke/forward stroke motion."""

    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self._held_px: Optional[np.ndarray] = None
        self._held_mm: Optional[np.ndarray] = None
        self._pending_px: Optional[np.ndarray] = None
        self._pending_mm: Optional[np.ndarray] = None
        self._pending_frames = 0

    def reset(self) -> None:
        self._held_px = None
        self._held_mm = None
        self._pending_px = None
        self._pending_mm = None
        self._pending_frames = 0

    def stabilize(self, direction_px, direction_mm) -> CueDirectionDecision:
        current_px = unit(np.asarray(direction_px, dtype=np.float32))
        current_mm = unit(np.asarray(direction_mm, dtype=np.float32))
        if self._held_px is None or self._held_mm is None:
            self._held_px = current_px
            self._held_mm = current_mm
            self._clear_pending()
            return CueDirectionDecision(
                direction_px=(float(current_px[0]), float(current_px[1])),
                direction_mm=(float(current_mm[0]), float(current_mm[1])),
                status="seed",
            )

        aligned_px, aligned_mm, flipped = self._align_to_held(current_px, current_mm)
        axis_delta = angle_deg(aligned_px, self._held_px)
        same_axis_limit = max(1.0, float(getattr(self.config, "cue_sector_angle_deg", 15.0)))
        edge_margin = max(0.0, float(getattr(self.config, "cue_sector_edge_margin_deg", 1.0)))
        if axis_delta <= same_axis_limit + edge_margin:
            self._held_px = aligned_px
            self._held_mm = aligned_mm
            self._clear_pending()
            return CueDirectionDecision(
                direction_px=(float(aligned_px[0]), float(aligned_px[1])),
                direction_mm=(float(aligned_mm[0]), float(aligned_mm[1])),
                status="axis_flip_hold" if flipped else "forward_track",
            )

        if self._pending_px is not None and self._pending_mm is not None:
            pending_delta = angle_deg(aligned_px, self._pending_px)
            if pending_delta <= same_axis_limit:
                self._pending_frames += 1
            else:
                self._pending_px = aligned_px
                self._pending_mm = aligned_mm
                self._pending_frames = 1
        else:
            self._pending_px = aligned_px
            self._pending_mm = aligned_mm
            self._pending_frames = 1

        confirm_frames = max(1, int(getattr(self.config, "cue_sector_switch_confirm_frames", 2)))
        if self._pending_frames < confirm_frames:
            return CueDirectionDecision(
                direction_px=(float(self._held_px[0]), float(self._held_px[1])),
                direction_mm=(float(self._held_mm[0]), float(self._held_mm[1])),
                status=f"switch_pending:{self._pending_frames}/{confirm_frames}",
            )

        self._held_px = aligned_px
        self._held_mm = aligned_mm
        self._clear_pending()
        return CueDirectionDecision(
            direction_px=(float(aligned_px[0]), float(aligned_px[1])),
            direction_mm=(float(aligned_mm[0]), float(aligned_mm[1])),
            status="switch_commit",
        )

    def _align_to_held(
        self,
        direction_px: np.ndarray,
        direction_mm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        assert self._held_px is not None
        if float(np.dot(direction_px, self._held_px)) >= 0.0:
            return direction_px, direction_mm, False
        return (-direction_px).astype(np.float32), (-direction_mm).astype(np.float32), True

    def _clear_pending(self) -> None:
        self._pending_px = None
        self._pending_mm = None
        self._pending_frames = 0
