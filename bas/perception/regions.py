from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from ..geometry import TableGeometry
from ..schemas import Detection
from ..utils import ensure_numpy_points, group_from_class

BALL_GROUPS = {"cue", "solid", "stripe", "black"}


@dataclass(frozen=True)
class DetectionRegionPolicy:
    global_polygon: Optional[np.ndarray] = None
    ball_polygon: Optional[np.ndarray] = None
    cue_stick_polygon: Optional[np.ndarray] = None


def build_detection_region_policy(
    frame_shape: Sequence[int],
    geometry: TableGeometry | None,
    fallback_polygon: Optional[np.ndarray] = None,
) -> DetectionRegionPolicy:
    height, width = _frame_size(frame_shape)
    fallback = _normalize_polygon(fallback_polygon)
    outer = None
    inner = None
    if geometry is not None and not bool(getattr(geometry, "is_empty", True)):
        try:
            outer_scaled, inner_scaled, _ = geometry.scaled(width, height)
            outer = _normalize_polygon(outer_scaled)
            inner = _normalize_polygon(inner_scaled)
        except Exception:
            outer = None
            inner = None
    global_polygon = outer if outer is not None else fallback
    cue_stick_polygon = outer if outer is not None else global_polygon
    ball_polygon = inner if inner is not None else global_polygon
    return DetectionRegionPolicy(
        global_polygon=global_polygon,
        ball_polygon=ball_polygon,
        cue_stick_polygon=cue_stick_polygon,
    )


def filter_detections_by_region(
    detections: Sequence[Detection],
    policy: DetectionRegionPolicy | None,
) -> list[Detection]:
    if policy is None:
        return list(detections)
    out: list[Detection] = []
    for detection in detections:
        polygon = _polygon_for_detection(detection, policy)
        if polygon is not None and not _contains_point(polygon, detection.center):
            continue
        out.append(detection)
    return out


def _frame_size(frame_shape: Sequence[int]) -> tuple[int, int]:
    dims = list(frame_shape)
    if len(dims) < 2:
        return 0, 0
    return int(dims[0]), int(dims[1])


def _normalize_polygon(polygon: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if polygon is None:
        return None
    arr = ensure_numpy_points(polygon)
    if arr.shape[0] < 3:
        return None
    return arr.astype(np.float32)


def _polygon_for_detection(detection: Detection, policy: DetectionRegionPolicy) -> Optional[np.ndarray]:
    group = group_from_class(detection.cls_name)
    if group in BALL_GROUPS:
        return policy.ball_polygon if policy.ball_polygon is not None else policy.global_polygon
    if group == "cue_stick":
        return policy.cue_stick_polygon if policy.cue_stick_polygon is not None else policy.global_polygon
    return policy.global_polygon


def _contains_point(polygon: np.ndarray, point: tuple[float, float]) -> bool:
    inside = cv2.pointPolygonTest(
        polygon.reshape((-1, 1, 2)).astype(np.float32),
        (float(point[0]), float(point[1])),
        False,
    )
    return inside >= 0.0
