from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from ..schemas import TrackObservation
from ..utils import clamp, point_segment_distance, unit


@dataclass(frozen=True)
class CueStickAimPx:
    tip_px: np.ndarray
    tail_px: np.ndarray
    direction_px: np.ndarray
    source: str
    score: float


class CueStickAimDetector:
    """Finds the cue-stick aim line in camera pixels.

    The detector prefers frame edges because an axis-aligned object box cannot
    reliably encode diagonal stick direction. Detection tracks remain the
    fallback when a frame is unavailable or edges are too weak.
    """

    def __init__(
        self,
        *,
        min_line_length_px: float = 72.0,
        cue_tip_near_factor: float = 5.2,
        cue_line_near_factor: float = 3.2,
    ) -> None:
        self.min_line_length_px = float(min_line_length_px)
        self.cue_tip_near_factor = float(cue_tip_near_factor)
        self.cue_line_near_factor = float(cue_line_near_factor)

    def detect(
        self,
        *,
        frame_bgr: Optional[np.ndarray],
        tracks: Sequence[TrackObservation],
        cue_center_px: np.ndarray,
        cue_radius_px: float,
        inner_polygon_px: Optional[np.ndarray] = None,
        min_stick_quality: float = 0.25,
    ) -> Optional[CueStickAimPx]:
        cue_center = np.asarray(cue_center_px, dtype=np.float32).reshape((2,))
        cue_radius = max(2.0, float(cue_radius_px))
        if frame_bgr is not None:
            edge_aim = self._detect_from_edges(frame_bgr, cue_center, cue_radius, inner_polygon_px)
            if edge_aim is not None:
                return edge_aim
        return self._detect_from_tracks(tracks, cue_center, cue_radius, min_stick_quality=min_stick_quality)

    def _detect_from_tracks(
        self,
        tracks: Sequence[TrackObservation],
        cue_center: np.ndarray,
        cue_radius: float,
        *,
        min_stick_quality: float,
    ) -> Optional[CueStickAimPx]:
        best: Optional[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = None
        line_limit = max(18.0, 4.5 * cue_radius)
        near_limit = max(line_limit, 8.0 * cue_radius)
        for track in tracks:
            if str(track.group).strip().lower() != "cue_stick":
                continue
            if str(getattr(track, "visibility", "visible")).strip().lower() != "visible":
                continue
            if float(getattr(track, "quality", 0.0)) < float(min_stick_quality):
                continue
            endpoints = self._track_axis_endpoints(track)
            if endpoints is None:
                continue
            p1, p2 = endpoints
            seg_len = float(np.linalg.norm(p2 - p1))
            if seg_len < max(24.0, 1.6 * cue_radius):
                continue
            line_dist = point_segment_distance(cue_center, p1, p2)
            near, far = (p1, p2) if float(np.linalg.norm(p1 - cue_center)) <= float(np.linalg.norm(p2 - cue_center)) else (p2, p1)
            near_dist = float(np.linalg.norm(near - cue_center))
            if line_dist > line_limit and near_dist > near_limit:
                continue
            direction = unit(cue_center - near)
            align_axis = abs(float(np.dot(unit(p2 - p1), direction)))
            if align_axis < 0.42:
                continue
            score = float(seg_len + 70.0 * float(track.confidence) - 1.4 * line_dist - 0.28 * near_dist)
            if best is None or score > best[0]:
                best = (score, near.astype(np.float32), far.astype(np.float32), direction.astype(np.float32))
        if best is None:
            return None
        score, tip, tail, direction = best
        return CueStickAimPx(tip_px=tip, tail_px=tail, direction_px=direction, source="track_bbox", score=float(score))

    def _detect_from_edges(
        self,
        frame_bgr: np.ndarray,
        cue_center: np.ndarray,
        cue_radius: float,
        inner_polygon_px: Optional[np.ndarray],
    ) -> Optional[CueStickAimPx]:
        h, w = frame_bgr.shape[:2]
        diag = float(math.hypot(w, h))
        polygon = np.asarray(inner_polygon_px, dtype=np.float32).reshape((-1, 2)) if inner_polygon_px is not None else np.empty((0, 2))
        if polygon.size >= 6:
            x_min = float(np.min(polygon[:, 0]))
            x_max = float(np.max(polygon[:, 0]))
            y_min = float(np.min(polygon[:, 1]))
            y_max = float(np.max(polygon[:, 1]))
            diag = max(120.0, float(math.hypot(x_max - x_min, y_max - y_min)))

        roi_half = int(clamp(0.28 * diag, 220.0, 560.0))
        cx = int(round(float(cue_center[0])))
        cy = int(round(float(cue_center[1])))
        x1 = max(0, cx - roi_half)
        y1 = max(0, cy - roi_half)
        x2 = min(w, cx + roi_half)
        y2 = min(h, cy + roi_half)
        if (x2 - x1) < 24 or (y2 - y1) < 24:
            return None

        roi = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0.0)
        edges = cv2.Canny(gray, 55, 150)
        if int(np.count_nonzero(edges)) < 16:
            return None

        line_blocks: list[np.ndarray] = []
        for threshold, min_len_factor, max_gap in ((36, 0.60, 14), (26, 0.45, 20), (18, 0.35, 28)):
            lines = cv2.HoughLinesP(
                edges,
                rho=1.0,
                theta=float(np.pi / 180.0),
                threshold=int(threshold),
                minLineLength=int(max(24.0, self.min_line_length_px * float(min_len_factor))),
                maxLineGap=int(max_gap),
            )
            if lines is not None and lines.size > 0:
                line_blocks.append(lines.reshape((-1, 4)))
        if not line_blocks:
            return None

        tip_limit = max(26.0, self.cue_tip_near_factor * cue_radius + 10.0)
        line_limit = max(14.0, self.cue_line_near_factor * cue_radius + 8.0)
        best: Optional[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = None
        for line_block in line_blocks:
            for line in line_block:
                p1 = np.asarray([float(line[0] + x1), float(line[1] + y1)], dtype=np.float32)
                p2 = np.asarray([float(line[2] + x1), float(line[3] + y1)], dtype=np.float32)
                seg = p2 - p1
                seg_len = float(np.linalg.norm(seg))
                if seg_len < self.min_line_length_px:
                    continue
                line_dist = point_segment_distance(cue_center, p1, p2)
                if line_dist > line_limit:
                    continue
                d1 = float(np.linalg.norm(p1 - cue_center))
                d2 = float(np.linalg.norm(p2 - cue_center))
                near = p1 if d1 <= d2 else p2
                far = p2 if d1 <= d2 else p1

                denom = float(np.dot(seg, seg))
                if denom <= 1e-9:
                    continue
                t = clamp(float(np.dot(cue_center - p1, seg) / denom), 0.0, 1.0)
                projected = p1 + seg * t
                tip = projected if float(np.linalg.norm(projected - cue_center)) >= max(3.0, 0.22 * cue_radius) else near
                tip_dist = float(np.linalg.norm(tip - cue_center))
                if tip_dist > tip_limit:
                    tip = near
                    tip_dist = float(np.linalg.norm(tip - cue_center))
                    if tip_dist > tip_limit:
                        continue

                axis = unit(tip - far)
                cue_to_tip = unit(cue_center - tip)
                align = abs(float(np.dot(axis, cue_to_tip)))
                if align < 0.42:
                    continue
                direction = axis if float(np.dot(axis, cue_to_tip)) >= 0.0 else -axis
                score = float(seg_len + 70.0 * align - 1.2 * line_dist - 0.9 * tip_dist + 0.1 * max(d1, d2))
                if best is None or score > best[0]:
                    best = (score, tip.astype(np.float32), far.astype(np.float32), direction.astype(np.float32))
        if best is None:
            return None
        score, tip, tail, direction = best
        return CueStickAimPx(tip_px=tip, tail_px=tail, direction_px=direction, source="frame_edges", score=float(score))

    @staticmethod
    def _track_axis_endpoints(track: TrackObservation) -> Optional[tuple[np.ndarray, np.ndarray]]:
        x1, y1, x2, y2 = [float(value) for value in track.bbox]
        w = x2 - x1
        h = y2 - y1
        if w <= 1.0 or h <= 1.0:
            return None
        if abs(w) >= abs(h):
            p1 = np.asarray([x1, (y1 + y2) * 0.5], dtype=np.float32)
            p2 = np.asarray([x2, (y1 + y2) * 0.5], dtype=np.float32)
        else:
            p1 = np.asarray([(x1 + x2) * 0.5, y1], dtype=np.float32)
            p2 = np.asarray([(x1 + x2) * 0.5, y2], dtype=np.float32)
        return p1, p2
