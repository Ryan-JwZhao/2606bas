from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from ..schemas import TrackObservation
from .cue_aim import CueStickAimDetector, CueStickAimPx
from .cue_direction_stability import CueDirectionStabilizer


@dataclass
class PlannerAimFrameContext:
    frame_bgr: Optional[np.ndarray]
    tracks: Sequence[TrackObservation]
    cue_center_px: np.ndarray
    cue_radius_px: float
    inner_polygon_px: Optional[np.ndarray]
    aim_detector: CueStickAimDetector
    min_stick_quality: float
    direction_stabilizer: Optional[CueDirectionStabilizer] = None
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
            raw_aim = self.aim_detector.detect(
                frame_bgr=self.frame_bgr,
                tracks=self.tracks,
                cue_center_px=self.cue_center_px,
                cue_radius_px=self.cue_radius_px,
                inner_polygon_px=self.inner_polygon_px,
                min_stick_quality=self.min_stick_quality,
                prefer_tracks=False,
                allow_edge_detection=True,
            )
            if raw_aim is None:
                if self.direction_stabilizer is not None:
                    self.direction_stabilizer.reset()
                self._shared_aim = None
            elif self.direction_stabilizer is None:
                self._shared_aim = raw_aim
            else:
                decision = self.direction_stabilizer.stabilize(
                    raw_aim.direction_px,
                    raw_aim.direction_px,
                )
                self._shared_aim = replace(
                    raw_aim,
                    direction_px=np.asarray(decision.direction_px, dtype=np.float32),
                    stability_status=str(decision.status),
                )
            self._shared_aim_ready = True
        return self._shared_aim
