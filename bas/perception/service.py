from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..schemas import DetectionsFrame, FramePacket
from .detector import Detector
from .regions import DetectionRegionPolicy, filter_detections_by_region


class DetectService:
    def __init__(self, detector: Detector, detect_interval_frames: int = 1, detect_fps_limit_hz: float = 0.0):
        self.detector = detector
        self.detect_interval_frames = max(1, int(detect_interval_frames))
        self.detect_fps_limit_hz = max(0.0, float(detect_fps_limit_hz))
        self._frame_index = 0
        self._last_detect_ts = 0.0
        self._last_result: Optional[DetectionsFrame] = None

    def reset_cache(self) -> None:
        self._last_detect_ts = 0.0
        self._last_result = None

    def process(
        self,
        frame: FramePacket,
        mask_polygon: Optional[np.ndarray] = None,
        detection_regions: DetectionRegionPolicy | None = None,
    ) -> DetectionsFrame:
        start = time.perf_counter()
        frame_index = self._frame_index
        self._frame_index += 1
        if detection_regions is not None and not detection_regions.detection_enabled:
            self.reset_cache()
            return DetectionsFrame(
                frame_id=frame.frame_id,
                ts_cam_ns=frame.ts_cam_ns,
                detections=[],
                detector_version="region_disabled",
                latency_ms=0.0,
            )
        if self._last_result is not None and not self._should_detect(start, frame_index):
            cached_detections = filter_detections_by_region(
                self._last_result.detections,
                detection_regions,
            )
            return DetectionsFrame(
                frame_id=frame.frame_id,
                ts_cam_ns=frame.ts_cam_ns,
                detections=cached_detections,
                detector_version=f"{getattr(self.detector, 'version', 'unknown')}:cached",
                latency_ms=0.0,
            )
        detections = self.detector.detect(frame.image, mask_polygon=mask_polygon) if frame.image is not None else []
        detections = filter_detections_by_region(detections, detection_regions)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._last_detect_ts = time.perf_counter()
        result = DetectionsFrame(
            frame_id=frame.frame_id,
            ts_cam_ns=frame.ts_cam_ns,
            detections=detections,
            detector_version=getattr(self.detector, "version", "unknown"),
            latency_ms=float(latency_ms),
        )
        self._last_result = result
        return result

    def _should_detect(self, now: float, frame_index: int) -> bool:
        frame_due = (frame_index % self.detect_interval_frames) == 0
        if not frame_due:
            return False
        if self.detect_fps_limit_hz <= 0.0:
            return True
        return (now - self._last_detect_ts) >= (1.0 / self.detect_fps_limit_hz)
