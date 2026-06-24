from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .utils import unit


def cue_alignment_start(
    cue: np.ndarray,
    ghost: np.ndarray,
    table_polygon_mm: np.ndarray,
    *,
    table_width_mm: float,
    table_height_mm: float,
    step_mm: float = 12.0,
) -> np.ndarray:
    aim = np.asarray(ghost, dtype=np.float32) - np.asarray(cue, dtype=np.float32)
    cue_point = np.asarray(cue, dtype=np.float32)
    if float(np.linalg.norm(aim)) < 1e-6:
        return cue_point.copy()
    table_diag = math.hypot(float(table_width_mm), float(table_height_mm))
    return trace_ray_inside_polygon(cue_point, -unit(aim), table_polygon_mm, max_length=0.55 * table_diag, step_mm=step_mm)


def trace_ray_inside_polygon(
    origin: np.ndarray,
    direction: np.ndarray,
    polygon: np.ndarray,
    *,
    max_length: float,
    step_mm: float,
) -> np.ndarray:
    poly = np.asarray(polygon, dtype=np.float32).reshape((-1, 2))
    start = np.asarray(origin, dtype=np.float32)
    if poly.shape[0] < 3:
        return (start + unit(direction) * float(max_length)).astype(np.float32)
    best = start.copy()
    d = unit(direction)
    for dist in np.arange(float(step_mm), float(max_length) + float(step_mm), float(step_mm)):
        point = start + d * float(dist)
        if cv2.pointPolygonTest(poly.reshape((-1, 1, 2)), (float(point[0]), float(point[1])), False) < 0.0:
            break
        best = point.astype(np.float32)
    return best


def rule_cue_separation_end(
    cue: np.ndarray,
    ghost: np.ndarray,
    target: np.ndarray,
    table_polygon_mm: np.ndarray,
    *,
    fallback_length_mm: float,
) -> Optional[np.ndarray]:
    incoming = np.asarray(ghost, dtype=np.float32) - np.asarray(cue, dtype=np.float32)
    object_dir = np.asarray(target, dtype=np.float32) - np.asarray(ghost, dtype=np.float32)
    incoming_norm = float(np.linalg.norm(incoming))
    object_norm = float(np.linalg.norm(object_dir))
    if incoming_norm < 1e-6 or object_norm < 1e-6:
        return None
    incoming_u = incoming / incoming_norm
    object_u = object_dir / object_norm
    cue_after = incoming_u - object_u * float(np.dot(incoming_u, object_u))
    if float(np.linalg.norm(cue_after)) < 1e-3:
        return None
    return estimate_route_end(ghost, unit(cue_after), table_polygon_mm, fallback_length_mm=fallback_length_mm)


def estimate_route_end(
    origin: np.ndarray,
    direction: np.ndarray,
    table_polygon_mm: np.ndarray,
    *,
    fallback_length_mm: float,
    min_t: float = 1.0,
) -> np.ndarray:
    start = np.asarray(origin, dtype=np.float32)
    hit = ray_polygon_first_hit(start, direction, table_polygon_mm, min_t=min_t)
    if hit is not None:
        return hit.astype(np.float32)
    return (start + unit(direction) * max(120.0, float(fallback_length_mm))).astype(np.float32)


def ray_polygon_first_hit(origin: np.ndarray, direction: np.ndarray, polygon: np.ndarray, *, min_t: float) -> Optional[np.ndarray]:
    poly = np.asarray(polygon, dtype=np.float32).reshape((-1, 2))
    if poly.shape[0] < 3:
        return None
    start = np.asarray(origin, dtype=np.float32)
    d = unit(direction)
    best_t: Optional[float] = None
    best_hit: Optional[np.ndarray] = None
    for idx in range(poly.shape[0]):
        a = poly[idx].astype(np.float32)
        b = poly[(idx + 1) % poly.shape[0]].astype(np.float32)
        seg = b - a
        denom = float(d[0] * seg[1] - d[1] * seg[0])
        if abs(denom) < 1e-7:
            continue
        ao = a - start
        t = float((ao[0] * seg[1] - ao[1] * seg[0]) / denom)
        u = float((ao[0] * d[1] - ao[1] * d[0]) / denom)
        if t < min_t or u < -1e-4 or u > 1.0001:
            continue
        if best_t is None or t < best_t:
            best_t = t
            best_hit = start + d * t
    return best_hit
