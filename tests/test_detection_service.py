from __future__ import annotations

import numpy as np

from bas.perception.detector import Detector
from bas.perception.service import DetectService
from bas.schemas import Detection, FramePacket


class CountingDetector(Detector):
    version = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_bgr, mask_polygon=None):
        self.calls += 1
        return [
            Detection(
                bbox=(1.0, 2.0, 3.0, 4.0),
                conf=0.9,
                cls_id=0,
                cls_name=f"call_{self.calls}",
            )
        ]


def _frame(frame_id: int) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        ts_cam_ns=frame_id * 1_000_000,
        camera_id="test",
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def test_detect_service_reuses_detections_between_intervals() -> None:
    detector = CountingDetector()
    service = DetectService(detector, detect_interval_frames=3, detect_fps_limit_hz=0.0)

    outputs = [service.process(_frame(i)) for i in range(5)]

    assert detector.calls == 2
    assert outputs[0].detections[0].cls_name == "call_1"
    assert outputs[1].detections[0].cls_name == "call_1"
    assert outputs[2].detections[0].cls_name == "call_1"
    assert outputs[3].detections[0].cls_name == "call_2"
    assert outputs[1].detector_version == "counting:cached"


def test_detect_service_resets_cached_result() -> None:
    detector = CountingDetector()
    service = DetectService(detector, detect_interval_frames=5, detect_fps_limit_hz=0.0)

    first = service.process(_frame(0))
    service.reset_cache()
    second = service.process(_frame(1))

    assert detector.calls == 2
    assert first.detections[0].cls_name == "call_1"
    assert second.detections[0].cls_name == "call_2"
