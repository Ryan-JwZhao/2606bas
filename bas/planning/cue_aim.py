from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from ..schemas import TrackObservation
from ..utils import clamp, point_segment_distance, unit
from .cue_direction_resolver import CueDirectionResolver, ResolvedCueDirectionPx


@dataclass(frozen=True)
class CueStickAimPx:
    tip_px: np.ndarray
    tail_px: np.ndarray
    direction_px: np.ndarray
    source: str
    score: float
    direction_confidence: float = 1.0
    direction_status: str = "body_side_strong"
    stability_status: str = "raw"
    track_id: Optional[int] = None
    track_quality: Optional[float] = None
    track_confidence: Optional[float] = None


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
        cue_line_near_factor: float = 1.25,
    ) -> None:
        self.min_line_length_px = float(min_line_length_px)
        self.cue_tip_near_factor = float(cue_tip_near_factor)
        self.cue_line_near_factor = float(cue_line_near_factor)
        self.direction_resolver = CueDirectionResolver()

    def detect(
        self,
        *,
        frame_bgr: Optional[np.ndarray],
        tracks: Sequence[TrackObservation],
        cue_center_px: np.ndarray,
        cue_radius_px: float,
        inner_polygon_px: Optional[np.ndarray] = None,
        min_stick_quality: float = 0.25,
        prefer_tracks: bool = False,
        allow_edge_detection: bool = True,
    ) -> Optional[CueStickAimPx]:
        cue_center = np.asarray(cue_center_px, dtype=np.float32).reshape((2,))
        cue_radius = max(2.0, float(cue_radius_px))
        track_aim: Optional[CueStickAimPx] = None
        if prefer_tracks:
            track_aim = self._detect_from_tracks(
                tracks,
                cue_center,
                cue_radius,
                min_stick_quality=min_stick_quality,
            )
            if track_aim is not None:
                return track_aim
        if allow_edge_detection and frame_bgr is not None:
            edge_aim = self._detect_from_edges(
                frame_bgr,
                cue_center,
                cue_radius,
                inner_polygon_px,
                tracks=tracks,
                min_stick_quality=min_stick_quality,
            )
            if edge_aim is not None:
                return edge_aim
        if track_aim is None:
            track_aim = self._detect_from_tracks(
                tracks,
                cue_center,
                cue_radius,
                min_stick_quality=min_stick_quality,
            )
        return track_aim

    def _detect_from_tracks(
        self,
        tracks: Sequence[TrackObservation],
        cue_center: np.ndarray,
        cue_radius: float,
        *,
        min_stick_quality: float,
    ) -> Optional[CueStickAimPx]:
        best: Optional[tuple[float, TrackObservation, ResolvedCueDirectionPx]] = None
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
            near = p1 if float(np.linalg.norm(p1 - cue_center)) <= float(np.linalg.norm(p2 - cue_center)) else p2
            near_dist = float(np.linalg.norm(near - cue_center))
            if line_dist > line_limit and near_dist > near_limit:
                continue
            resolved = self._resolve_direction(cue_center, cue_radius, p1, p2)
            if resolved is None:
                continue
            if resolved.status != "body_side_strong":
                continue
            align_axis = self._cue_alignment(cue_center, resolved.tip_px, resolved.direction_px)
            if align_axis < 0.42:
                continue
            tip_dist = float(np.linalg.norm(resolved.tip_px - cue_center))
            score = float(
                seg_len
                + 70.0 * float(track.confidence)
                - 1.4 * line_dist
                - 0.28 * tip_dist
                + self._orientation_bonus(resolved)
            )
            if best is None or score > best[0]:
                best = (score, track, resolved)
        if best is None:
            return None
        score, track, resolved = best
        return CueStickAimPx(
            tip_px=resolved.tip_px.astype(np.float32),
            tail_px=resolved.tail_px.astype(np.float32),
            direction_px=resolved.direction_px.astype(np.float32),
            source="track_bbox",
            score=float(score),
            direction_confidence=float(resolved.confidence),
            direction_status=str(resolved.status),
            track_id=int(track.track_id),
            track_quality=float(track.quality),
            track_confidence=float(track.confidence),
        )

    def _detect_from_edges(
        self,
        frame_bgr: np.ndarray,
        cue_center: np.ndarray,
        cue_radius: float,
        inner_polygon_px: Optional[np.ndarray],
        *,
        tracks: Sequence[TrackObservation],
        min_stick_quality: float,
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
        line_limit = max(8.0, self.cue_line_near_factor * cue_radius + 4.0)
        stick_tracks = self._eligible_stick_tracks(tracks, min_stick_quality=min_stick_quality)
        best: Optional[tuple[float, ResolvedCueDirectionPx, Optional[TrackObservation]]] = None
        for line_block in line_blocks:
            for line in line_block:
                p1 = np.asarray([float(line[0] + x1), float(line[1] + y1)], dtype=np.float32)
                p2 = np.asarray([float(line[2] + x1), float(line[3] + y1)], dtype=np.float32)
                seg = p2 - p1
                seg_len = float(np.linalg.norm(seg))
                if seg_len < self.min_line_length_px:
                    continue
                line_dist = self._point_line_distance(cue_center, p1, p2)
                if line_dist > line_limit:
                    continue
                association = self._edge_track_association(
                    p1,
                    p2,
                    stick_tracks,
                    cue_radius=cue_radius,
                )
                if stick_tracks and association is None:
                    continue
                associated_track = association[0] if association is not None else None
                association_score = association[1] if association is not None else 0.0
                d1 = float(np.linalg.norm(p1 - cue_center))
                d2 = float(np.linalg.norm(p2 - cue_center))
                resolved = self._resolve_direction(cue_center, cue_radius, p1, p2)
                if resolved is None:
                    continue
                if resolved.status != "body_side_strong":
                    continue
                tip_dist = float(np.linalg.norm(resolved.tip_px - cue_center))
                if tip_dist > tip_limit:
                    continue
                align = self._cue_alignment(cue_center, resolved.tip_px, resolved.direction_px)
                if align < 0.42:
                    continue
                score = float(
                    seg_len
                    + 70.0 * align
                    - 1.2 * line_dist
                    - 0.9 * tip_dist
                    + 0.1 * max(d1, d2)
                    + self._orientation_bonus(resolved)
                    + 0.35 * association_score
                )
                if best is None or score > best[0]:
                    best = (score, resolved, associated_track)
        if best is None:
            return None
        score, resolved, associated_track = best
        return CueStickAimPx(
            tip_px=resolved.tip_px.astype(np.float32),
            tail_px=resolved.tail_px.astype(np.float32),
            direction_px=resolved.direction_px.astype(np.float32),
            source="frame_edges",
            score=float(score),
            direction_confidence=float(resolved.confidence),
            direction_status=str(resolved.status),
            track_id=int(associated_track.track_id) if associated_track is not None else None,
            track_quality=float(associated_track.quality) if associated_track is not None else None,
            track_confidence=float(associated_track.confidence) if associated_track is not None else None,
        )

    def _resolve_direction(
        self,
        cue_center: np.ndarray,
        cue_radius: float,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> Optional[ResolvedCueDirectionPx]:
        resolved = self.direction_resolver.resolve(
            cue_center_px=cue_center,
            cue_radius_px=cue_radius,
            p1_px=p1,
            p2_px=p2,
        )
        if resolved is None:
            return None
        if float(np.linalg.norm(resolved.direction_px)) < 1e-6:
            return None
        return resolved

    @staticmethod
    def _cue_alignment(cue_center: np.ndarray, tip_px: np.ndarray, direction_px: np.ndarray) -> float:
        cue_to_tip = cue_center - tip_px
        if float(np.linalg.norm(cue_to_tip)) < 1e-6:
            return 1.0
        return abs(float(np.dot(unit(direction_px), unit(cue_to_tip))))

    @staticmethod
    def _orientation_bonus(resolved: ResolvedCueDirectionPx) -> float:
        return float(0.12 * resolved.score)

    @staticmethod
    def _point_line_distance(point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        segment = np.asarray(p2, dtype=np.float32) - np.asarray(p1, dtype=np.float32)
        length = float(np.linalg.norm(segment))
        if length < 1.0e-6:
            return float("inf")
        offset = np.asarray(point, dtype=np.float32) - np.asarray(p1, dtype=np.float32)
        cross = float(segment[0] * offset[1] - segment[1] * offset[0])
        return abs(cross) / length

    @staticmethod
    def _eligible_stick_tracks(
        tracks: Sequence[TrackObservation],
        *,
        min_stick_quality: float,
    ) -> list[TrackObservation]:
        return [
            track
            for track in tracks
            if str(track.group).strip().lower() == "cue_stick"
            and str(getattr(track, "visibility", "visible")).strip().lower() == "visible"
            and float(getattr(track, "quality", 0.0)) >= float(min_stick_quality)
        ]

    @staticmethod
    def _edge_track_association(
        p1: np.ndarray,
        p2: np.ndarray,
        tracks: Sequence[TrackObservation],
        *,
        cue_radius: float,
    ) -> Optional[tuple[TrackObservation, float]]:
        if not tracks:
            return None
        segment_length = float(np.linalg.norm(p2 - p1))
        margin = max(6.0, 0.75 * float(cue_radius))
        best: Optional[tuple[TrackObservation, float]] = None
        for track in tracks:
            x1, y1, x2, y2 = [float(value) for value in track.bbox]
            left = int(np.floor(x1 - margin))
            top = int(np.floor(y1 - margin))
            width = max(1, int(np.ceil((x2 - x1) + 2.0 * margin)))
            height = max(1, int(np.ceil((y2 - y1) + 2.0 * margin)))
            intersects, clipped_p1, clipped_p2 = cv2.clipLine(
                (left, top, width, height),
                tuple(int(round(value)) for value in p1),
                tuple(int(round(value)) for value in p2),
            )
            if not intersects:
                continue
            overlap = float(
                np.linalg.norm(
                    np.asarray(clipped_p2, dtype=np.float32)
                    - np.asarray(clipped_p1, dtype=np.float32)
                )
            )
            major = max(abs(x2 - x1), abs(y2 - y1))
            minimum_overlap = max(18.0, 0.35 * min(max(major, 1.0), segment_length))
            if overlap < minimum_overlap:
                continue
            score = overlap + 20.0 * float(getattr(track, "quality", 0.0))
            if best is None or score > best[1]:
                best = (track, score)
        return best

    @staticmethod
    def _track_axis_endpoints(track: TrackObservation) -> Optional[tuple[np.ndarray, np.ndarray]]:
        oriented = getattr(track, "axis_endpoints_px", None)
        axis_quality = float(getattr(track, "axis_quality", 0.0))
        if oriented is not None and axis_quality >= 0.20:
            points = np.asarray(oriented, dtype=np.float32).reshape((-1, 2))
            if points.shape[0] == 2 and np.all(np.isfinite(points)):
                if float(np.linalg.norm(points[1] - points[0])) > 1.0:
                    return points[0].copy(), points[1].copy()
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
