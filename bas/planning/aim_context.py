from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..schemas import TrackObservation
from .cue_aim import CueStickAimDetector, CueStickAimPx


@dataclass
class PlannerAimFrameContext:
    frame_bgr: Optional[np.ndarray]
    tracks: Sequence[TrackObservation]
    cue_center_px: np.ndarray
    cue_radius_px: float
    inner_polygon_px: Optional[np.ndarray]
    aim_detector: CueStickAimDetector
    min_stick_quality: float
    detect_call_count: int = 0
    _shared_aim_ready: bool = field(default=False, init=False, repr=False)
    _shared_aim: Optional[CueStickAimPx] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cue_center_px = np.asarray(self.cue_center_px, dtype=np.float32).reshape((2,))
        self.cue_radius_px = max(2.0, float(self.cue_radius_px))
        self.min_stick_quality = max(0.0, float(self.min_stick_quality))
        if self.inner_polygon_px is not None:
            polygon = np.asarray(self.inner_polygon_px, dtype=np.float32).reshape((-1, 2))
            self.inner_polygon_px = polygon if polygon.shape[0] >= 3 else None

    def shared_aim(self) -> Optional[CueStickAimPx]:
        if not self._shared_aim_ready:
            self.detect_call_count += 1
            self._shared_aim = self.aim_detector.detect(
                frame_bgr=self.frame_bgr,
                tracks=self.tracks,
                cue_center_px=self.cue_center_px,
                cue_radius_px=self.cue_radius_px,
                inner_polygon_px=self.inner_polygon_px,
                min_stick_quality=self.min_stick_quality,
                prefer_tracks=False,
                allow_edge_detection=True,
            )
            self._shared_aim_ready = True
        return self._shared_aim
