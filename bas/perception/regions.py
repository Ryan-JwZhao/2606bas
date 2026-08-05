from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from ..geometry import TableGeometry, canonical_pocket_indices, pocket_arc_center
from ..schemas import Detection
from ..utils import ensure_numpy_points, group_from_class

BALL_GROUPS = {"cue", "solid", "stripe", "black"}


@dataclass(frozen=True)
class PocketGuardRegion:
    pocket_index: int
    polygon: np.ndarray
    center_px: tuple[float, float]
    ball_diameter_px: float


@dataclass(frozen=True)
class DetectionRegionPolicy:
    global_polygon: Optional[np.ndarray] = None
    ball_polygon: Optional[np.ndarray] = None
    cue_stick_polygon: Optional[np.ndarray] = None
    ball_guard_regions: tuple[PocketGuardRegion, ...] = ()
    detection_enabled: bool = True


def build_detection_region_policy(
    frame_shape: Sequence[int],
    geometry: TableGeometry | None,
    fallback_polygon: Optional[np.ndarray] = None,
    ball_diameter_px_by_pocket: Optional[Sequence[float]] = None,
) -> DetectionRegionPolicy:
    height, width = _frame_size(frame_shape)
    fallback = _normalize_polygon(fallback_polygon)
    outer = None
    inner = None
    pockets_scaled: list[np.ndarray] = []
    if geometry is not None and not bool(getattr(geometry, "is_empty", True)):
        try:
            outer_scaled, inner_scaled, pockets = geometry.scaled(width, height)
            outer = _normalize_polygon(outer_scaled)
            inner = _normalize_polygon(inner_scaled)
            pockets_scaled = [np.asarray(pocket, dtype=np.float32).reshape((-1, 2)) for pocket in pockets]
        except Exception:
            outer = None
            inner = None
            pockets_scaled = []
    global_polygon = outer if outer is not None else fallback
    cue_stick_polygon = outer if outer is not None else global_polygon
    ball_polygon = inner if inner is not None else global_polygon
    guards = _build_pocket_guards(pockets_scaled, ball_diameter_px_by_pocket)
    return DetectionRegionPolicy(
        global_polygon=global_polygon,
        ball_polygon=ball_polygon,
        cue_stick_polygon=cue_stick_polygon,
        ball_guard_regions=guards,
    )


def filter_detections_by_region(
    detections: Sequence[Detection],
    policy: DetectionRegionPolicy | None,
) -> list[Detection]:
    if policy is None:
        return list(detections)
    if not policy.detection_enabled:
        return []
    out: list[Detection] = []
    for detection in detections:
        polygon = _polygon_for_detection(detection, policy)
        allowed = polygon is None or _contains_point(polygon, detection.center)
        if not allowed:
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


def _build_pocket_guards(
    pockets: Sequence[np.ndarray],
    ball_diameters_px: Optional[Sequence[float]],
) -> tuple[PocketGuardRegion, ...]:
    if ball_diameters_px is None:
        return ()
    diameters = [float(value) for value in ball_diameters_px]
    guards: list[PocketGuardRegion] = []
    ordered_indices = canonical_pocket_indices(
        [np.asarray(curve, dtype=np.float32).reshape((-1, 2)) for curve in pockets]
    )
    for pocket_index, source_index in enumerate(ordered_indices):
        curve = np.asarray(pockets[source_index], dtype=np.float32).reshape((-1, 2))
        points = np.asarray(curve, dtype=np.float32).reshape((-1, 2))
        if points.shape[0] < 2 or source_index >= len(diameters):
            continue
        diameter = max(1.0, diameters[source_index])
        # A ball centre can remain detectable for slightly more than one diameter
        # around the annotated jaw curve.  The value scales with calibration and
        # never depends on camera resolution or a hard-coded pixel distance.
        radius = max(1, int(round(diameter * 1.15)))
        samples: list[np.ndarray] = []
        for x, y in points:
            circle = cv2.ellipse2Poly(
                (int(round(float(x))), int(round(float(y)))),
                (radius, radius),
                0,
                0,
                360,
                20,
            )
            samples.append(circle.astype(np.float32))
        hull = cv2.convexHull(np.vstack([points, *samples]).astype(np.float32)).reshape((-1, 2))
        center = pocket_arc_center(points, diameter)
        guards.append(
            PocketGuardRegion(
                pocket_index=int(pocket_index),
                polygon=hull.astype(np.float32),
                center_px=(float(center[0]), float(center[1])),
                ball_diameter_px=float(diameter),
            )
        )
    return tuple(guards)


__all__ = [
    "DetectionRegionPolicy",
    "PocketGuardRegion",
    "build_detection_region_policy",
    "filter_detections_by_region",
]
