from __future__ import annotations

import numpy as np

from bas.perception.detector import Detector, _cue_axis_from_polygon
from bas.perception.regions import DetectionRegionPolicy
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


def test_cached_detections_are_refiltered_against_latest_region_policy() -> None:
    class BallDetector(Detector):
        version = "ball"

        def detect(self, frame_bgr, mask_polygon=None):
            return [Detection(bbox=(4.0, 4.0, 6.0, 6.0), conf=0.9, cls_id=0, cls_name="solid")]

    service = DetectService(BallDetector(), detect_interval_frames=5, detect_fps_limit_hz=0.0)
    permissive = DetectionRegionPolicy(
        ball_polygon=np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
    )
    restrictive = DetectionRegionPolicy(
        ball_polygon=np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    )

    first = service.process(_frame(0), detection_regions=permissive)
    cached = service.process(_frame(1), detection_regions=restrictive)

    assert len(first.detections) == 1
    assert cached.detections == []
    assert cached.detector_version == "ball:cached"


def test_disabled_region_policy_skips_detector_and_drops_cached_results() -> None:
    detector = CountingDetector()
    service = DetectService(detector, detect_interval_frames=5, detect_fps_limit_hz=0.0)

    first = service.process(_frame(0))
    disabled = service.process(
        _frame(1),
        detection_regions=DetectionRegionPolicy(detection_enabled=False),
    )
    resumed = service.process(_frame(2))

    assert first.detections
    assert disabled.detections == []
    assert disabled.detector_version == "region_disabled"
    assert detector.calls == 2
    assert resumed.detections[0].cls_name == "call_2"


def test_cue_axis_is_fitted_from_segmentation_polygon() -> None:
    axis = np.asarray([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    center = np.asarray([100.0, 80.0], dtype=np.float32)
    polygon = np.asarray(
        [
            center - axis * 70.0 - normal * 4.0,
            center + axis * 70.0 - normal * 4.0,
            center + axis * 70.0 + normal * 4.0,
            center - axis * 70.0 + normal * 4.0,
            center - axis * 60.0 - normal * 4.0,
        ],
        dtype=np.float32,
    )

    endpoints, quality = _cue_axis_from_polygon(polygon)

    assert endpoints is not None
    detected_axis = np.asarray(endpoints[1], dtype=np.float32) - np.asarray(endpoints[0], dtype=np.float32)
    detected_axis = detected_axis / np.linalg.norm(detected_axis)
    assert abs(float(np.dot(detected_axis, axis))) > 0.99
    assert quality > 0.5
