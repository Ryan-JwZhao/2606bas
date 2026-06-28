from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional


class RecordingFpsEstimator:
    def __init__(
        self,
        *,
        max_samples: int = 90,
        min_samples: int = 6,
        min_span_ns: int = 250_000_000,
    ) -> None:
        self._max_samples = max(4, int(max_samples))
        self._min_samples = max(2, int(min_samples))
        self._min_span_ns = max(1, int(min_span_ns))
        self._timestamps_ns: Deque[int] = deque(maxlen=self._max_samples)

    def reset(self) -> None:
        self._timestamps_ns.clear()

    def observe(self, ts_ns: int) -> None:
        value = int(ts_ns)
        if value <= 0:
            return
        if self._timestamps_ns and value < self._timestamps_ns[-1]:
            self.reset()
        if self._timestamps_ns and value == self._timestamps_ns[-1]:
            return
        self._timestamps_ns.append(value)

    def estimate(self, *, require_ready: bool = True) -> Optional[float]:
        if len(self._timestamps_ns) < 2:
            return None
        first = self._timestamps_ns[0]
        last = self._timestamps_ns[-1]
        span_ns = int(last - first)
        if span_ns <= 0:
            return None
        if require_ready and (len(self._timestamps_ns) < self._min_samples or span_ns < self._min_span_ns):
            return None
        fps = (len(self._timestamps_ns) - 1) * 1_000_000_000.0 / float(span_ns)
        if not math.isfinite(fps) or fps <= 0.0:
            return None
        return float(fps)
