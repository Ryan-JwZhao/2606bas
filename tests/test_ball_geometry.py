from __future__ import annotations

import cv2
import numpy as np

from bas.perception.ball_geometry import BallCenterRefiner
from bas.perception.detector import UltralyticsDetector
from bas.schemas import Detection


def _ellipse_polygon(center: tuple[float, float], axes: tuple[float, float], count: int = 96) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack(
        [
            center[0] + axes[0] * np.cos(angles),
            center[1] + axes[1] * np.sin(angles),
        ]
    ).astype(np.float32)


def test_segmentation_contour_refines_center_beyond_box_midpoint() -> None:
    frame = np.full((180, 220, 3), (40, 120, 40), dtype=np.uint8)
    polygon = _ellipse_polygon((112.4, 86.7), (21.0, 19.0))
    bbox = (86.0, 62.0, 134.0, 110.0)

    estimate = BallCenterRefiner().refine(frame, bbox, mask_polygon=polygon)

    np.testing.assert_allclose(estimate.center_px, [112.4, 86.7], atol=0.12)
    assert estimate.method == "segmentation_ellipse"
    assert estimate.quality > 0.8
    assert abs(estimate.radius_px - 20.0) < 0.2


def test_appearance_refinement_recovers_ball_when_model_has_no_mask() -> None:
    frame = np.full((160, 220, 3), (45, 125, 55), dtype=np.uint8)
    cv2.circle(frame, (121, 79), 22, (235, 235, 235), -1, cv2.LINE_AA)
    bbox = (94.0, 53.0, 142.0, 105.0)

    estimate = BallCenterRefiner().refine(frame, bbox)

    np.testing.assert_allclose(estimate.center_px, [121.0, 79.0], atol=0.8)
    assert estimate.method == "appearance_ellipse"
    assert estimate.quality > 0.6
    assert abs(estimate.radius_px - 22.0) < 1.5


def test_appearance_refinement_does_not_merge_touching_balls() -> None:
    frame = np.full((150, 220, 3), (45, 125, 55), dtype=np.uint8)
    cv2.circle(frame, (90, 75), 22, (235, 235, 235), -1, cv2.LINE_AA)
    cv2.circle(frame, (132, 75), 22, (235, 235, 235), -1, cv2.LINE_AA)
    bbox = (67.0, 52.0, 113.0, 98.0)

    estimate = BallCenterRefiner().refine(frame, bbox)

    np.testing.assert_allclose(estimate.center_px, [90.0, 75.0], atol=0.1)
    assert estimate.method == "bbox"
    assert estimate.radius_px == 23.0


def test_detection_interface_prefers_refined_geometry() -> None:
    detection = Detection(
        bbox=(10.0, 20.0, 50.0, 60.0),
        conf=0.9,
        cls_id=0,
        cls_name="cue",
        refined_center_px=(31.25, 42.5),
        refined_radius_px=18.75,
        geometry_quality=0.9,
        geometry_method="segmentation_ellipse",
    )

    assert detection.center == (31.25, 42.5)
    assert detection.radius_px == 18.75


class _ArrayValue:
    def __init__(self, value: np.ndarray):
        self._value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _Boxes:
    def __init__(self):
        self.xyxy = _ArrayValue(np.asarray([[28.0, 22.0, 76.0, 68.0]], dtype=np.float32))
        self.conf = _ArrayValue(np.asarray([0.93], dtype=np.float32))
        self.cls = _ArrayValue(np.asarray([0], dtype=np.float32))

    def __len__(self) -> int:
        return 1


class _Masks:
    def __init__(self):
        self.xy = [_ellipse_polygon((54.2, 43.7), (21.0, 20.0))]


class _Result:
    boxes = _Boxes()
    masks = _Masks()


class _Model:
    def predict(self, **_kwargs):
        return [_Result()]


def test_ultralytics_adapter_preserves_segmentation_geometry() -> None:
    detector = object.__new__(UltralyticsDetector)
    detector.model = _Model()
    detector.class_names = ["cue"]
    detector.conf = 0.35
    detector.iou = 0.45
    detector.device = "cpu"
    detector.tile_size = 320
    detector.tile_overlap = 0.0
    detector.max_det_per_tile = 10
    detector.batch_size = 1
    frame = np.full((100, 120, 3), (40, 120, 40), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 1
    np.testing.assert_allclose(detections[0].center, [54.2, 43.7], atol=0.15)
    assert detections[0].geometry_method == "segmentation_ellipse"
    assert detections[0].geometry_quality > 0.8
