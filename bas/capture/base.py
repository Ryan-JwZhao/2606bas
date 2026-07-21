from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np


@dataclass
class CaptureInfo:
    backend: str
    camera_id: str
    width: int
    height: int
    fps: float
    metadata: Dict[str, object]


@dataclass(frozen=True)
class VideoTimelineState:
    """Current position and bounds for a seekable video-file source."""

    current_frame: int
    total_frames: int
    fps: float

    @property
    def duration_seconds(self) -> float:
        return float(self.total_frames) / self.fps if self.fps > 0.0 else 0.0

    @property
    def current_seconds(self) -> float:
        return float(self.current_frame) / self.fps if self.fps > 0.0 else 0.0


class CaptureSource(Protocol):
    def is_opened(self) -> bool:
        ...

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, object]]:
        ...

    def release(self) -> None:
        ...

    def info(self) -> CaptureInfo:
        ...
