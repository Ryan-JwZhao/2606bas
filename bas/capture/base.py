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


class CaptureSource(Protocol):
    def is_opened(self) -> bool:
        ...

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, object]]:
        ...

    def release(self) -> None:
        ...

    def info(self) -> CaptureInfo:
        ...

