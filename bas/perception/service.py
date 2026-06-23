from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..schemas import DetectionsFrame, FramePacket
from .detector import Detector


class DetectService:
    def __init__(self, detector: Detector):
        self.detector = detector

    def process(self, frame: FramePacket, mask_polygon: Optional[np.ndarray] = None) -> DetectionsFrame:
        start = time.perf_counter()
        detections = self.detector.detect(frame.image, mask_polygon=mask_polygon) if frame.image is not None else []
        latency_ms = (time.perf_counter() - start) * 1000.0
        return DetectionsFrame(
            frame_id=frame.frame_id,
            ts_cam_ns=frame.ts_cam_ns,
            detections=detections,
            detector_version=getattr(self.detector, "version", "unknown"),
            latency_ms=float(latency_ms),
        )

