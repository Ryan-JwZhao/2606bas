from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..config import TrackerConfig
from ..schemas import Detection, DetectionsFrame, TrackObservation, TracksFrame
from ..utils import clamp, group_from_class, iou_xyxy
from .geometry_continuity import BallGeometryContinuity, blend_tracked_center


@dataclass
class _Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    center: np.ndarray
    radius_px: float
    confidence: float
    geometry_quality: float
    geometry_method: str
    geometry_continuity: BallGeometryContinuity
    votes: Deque[str]
    last_ts_ns: int
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((2,), dtype=np.float32))
    age: int = 1
    lost_frames: int = 0
    consecutive_hits: int = 1
    confirmed: bool = False

    @property
    def stable_class(self) -> str:
        if not self.votes:
            return "unknown"
        return Counter(self.votes).most_common(1)[0][0]


class TemporalTracker:
    version = "temporal_centroid_v2"

    def __init__(self, config: TrackerConfig):
        self.config = config
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections_frame: DetectionsFrame) -> TracksFrame:
        start = time.perf_counter()
        detections = [d for d in detections_frame.detections if d.conf >= self.config.low_conf]
        matches, unmatched_tracks, unmatched_dets = self._match(detections, detections_frame.ts_cam_ns)

        for tid, did in matches:
            self._update_track(self._tracks[tid], detections[did], detections_frame.ts_cam_ns)

        for tid in list(unmatched_tracks):
            tr = self._tracks.get(tid)
            if tr is None:
                continue
            tr.lost_frames += 1
            tr.consecutive_hits = 0
            dt = max(1e-6, (detections_frame.ts_cam_ns - tr.last_ts_ns) / 1e9)
            tr.center = tr.center + tr.velocity * min(dt, 0.1)
            x1, y1, x2, y2 = tr.bbox
            w = x2 - x1
            h = y2 - y1
            tr.bbox = (
                float(tr.center[0] - w * 0.5),
                float(tr.center[1] - h * 0.5),
                float(tr.center[0] + w * 0.5),
                float(tr.center[1] + h * 0.5),
            )
            if tr.lost_frames > self.config.max_lost_frames:
                del self._tracks[tid]

        for did in unmatched_dets:
            det = detections[did]
            if det.conf < self.config.high_conf:
                continue
            geometry_continuity = BallGeometryContinuity()
            geometry = geometry_continuity.measure(det)
            center = geometry.center_px
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(
                track_id=tid,
                bbox=tuple(float(v) for v in det.bbox),
                center=center,
                radius_px=float(geometry.radius_px),
                confidence=float(det.conf),
                geometry_quality=float(np.clip(det.geometry_quality, 0.0, 1.0)),
                geometry_method=str(det.geometry_method or "unknown"),
                geometry_continuity=geometry_continuity,
                votes=deque([det.cls_name], maxlen=int(self.config.vote_window)),
                last_ts_ns=detections_frame.ts_cam_ns,
                confirmed=max(1, int(self.config.min_confirmed_hits)) <= 1,
            )

        self._prune_duplicate_tracks()

        observations = [self._to_observation(tr) for tr in sorted(self._tracks.values(), key=lambda t: t.track_id)]
        latency_ms = (time.perf_counter() - start) * 1000.0
        return TracksFrame(
            frame_id=detections_frame.frame_id,
            ts_cam_ns=detections_frame.ts_cam_ns,
            tracks=observations,
            tracker_version=self.version,
            latency_ms=float(latency_ms),
        )

    def _match(self, detections: List[Detection], ts_ns: int) -> Tuple[List[Tuple[int, int]], set[int], set[int]]:
        unmatched_tracks = set(self._tracks.keys())
        unmatched_dets = set(range(len(detections)))
        if not self._tracks or not detections:
            return [], unmatched_tracks, unmatched_dets
        candidates: List[Tuple[float, int, int]] = []
        for tid, track in self._tracks.items():
            dt = max(0.0, (ts_ns - track.last_ts_ns) / 1e9)
            predicted = track.center + track.velocity * min(dt, 0.2)
            for did, det in enumerate(detections):
                center = np.asarray(det.center, dtype=np.float32)
                dist = float(np.linalg.norm(center - predicted))
                iou_score = iou_xyxy(track.bbox, det.bbox)
                radius = max(1.0, float(max(track.radius_px, det.radius_px)))
                cls_bonus = self._compatibility_bonus(track.stable_class, det.cls_name, dist=dist, iou_score=iou_score, radius_px=radius)
                if cls_bonus is None:
                    continue
                max_distance = self._max_match_distance(track, det, dt)
                dist_score = 1.0 - clamp(dist / max(1.0, max_distance), 0.0, 1.0)
                size_score = 1.0 - clamp(abs(float(det.radius_px) - float(track.radius_px)) / max(4.0, radius), 0.0, 1.0)
                score = 0.50 * dist_score + 0.24 * iou_score + 0.08 * size_score + 0.08 * float(det.conf) + cls_bonus
                if dist <= max_distance or iou_score >= self.config.match_iou:
                    candidates.append((score, tid, did))
        candidates.sort(reverse=True, key=lambda x: x[0])
        matches: List[Tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for score, tid, did in candidates:
            if tid in used_tracks or did in used_dets:
                continue
            if score < 0.10:
                continue
            used_tracks.add(tid)
            used_dets.add(did)
            matches.append((tid, did))
        unmatched_tracks -= used_tracks
        unmatched_dets -= used_dets
        return matches, unmatched_tracks, unmatched_dets

    def _update_track(self, track: _Track, det: Detection, ts_ns: int) -> None:
        geometry = track.geometry_continuity.measure(det)
        new_center = blend_tracked_center(
            track.center,
            geometry,
            previous_radius_px=track.radius_px,
            geometry_quality=det.geometry_quality,
        )
        dt = max(1e-6, (ts_ns - track.last_ts_ns) / 1e9)
        instant_v = (new_center - track.center) / dt
        alpha = float(clamp(self.config.velocity_smoothing, 0.0, 1.0))
        track.velocity = (1.0 - alpha) * track.velocity + alpha * instant_v
        track.center = new_center
        track.bbox = tuple(float(v) for v in det.bbox)
        track.radius_px = float(geometry.radius_px)
        track.confidence = float(det.conf)
        track.geometry_quality = float(np.clip(det.geometry_quality, 0.0, 1.0))
        track.geometry_method = str(det.geometry_method or "unknown")
        track.votes.append(det.cls_name)
        track.last_ts_ns = int(ts_ns)
        track.age += 1
        track.consecutive_hits += 1
        if track.consecutive_hits >= max(1, int(self.config.min_confirmed_hits)):
            track.confirmed = True
        track.lost_frames = 0

    def _to_observation(self, track: _Track) -> TrackObservation:
        stable = track.stable_class
        geometry_factor = 0.65 + 0.35 * float(np.clip(track.geometry_quality, 0.0, 1.0))
        quality = float(track.confidence) * geometry_factor * (0.72 ** max(0, track.lost_frames))
        visibility = "visible" if track.lost_frames == 0 else "occluded"
        return TrackObservation(
            track_id=track.track_id,
            bbox=track.bbox,
            center_px=(float(track.center[0]), float(track.center[1])),
            radius_px=float(track.radius_px),
            cls_name=stable,
            group=group_from_class(stable),
            confidence=float(track.confidence),
            velocity_px_s=(float(track.velocity[0]), float(track.velocity[1])),
            quality=quality,
            age=int(track.age),
            confirmed=bool(track.confirmed),
            lost_frames=int(track.lost_frames),
            visibility=visibility,
            geometry_quality=float(track.geometry_quality),
            geometry_method=str(track.geometry_method),
        )

    def _max_match_distance(self, track: _Track, det: Detection, dt: float) -> float:
        base = max(1.0, float(self.config.match_distance_px))
        motion = float(np.linalg.norm(track.velocity)) * min(max(0.0, dt), 0.2)
        radius = float(max(track.radius_px, det.radius_px, 1.0))
        return max(base, radius * 2.5 + motion * 3.0)

    def _compatibility_bonus(
        self,
        track_class: str,
        det_class: str,
        *,
        dist: float,
        iou_score: float,
        radius_px: float,
    ) -> float | None:
        track_group = group_from_class(track_class)
        det_group = group_from_class(det_class)
        if track_group == det_group:
            return 0.12
        if "cue_stick" in {track_group, det_group}:
            return None
        if "cue" in {track_group, det_group}:
            # Allow near-identical cue-ball classification blips, but avoid
            # snapping a cue track onto unrelated object-ball detections.
            if dist <= max(10.0, radius_px * 0.85) or iou_score >= 0.55:
                return -0.04
            return None
        if track_group == "other" or det_group == "other":
            return -0.06
        if "black" in {track_group, det_group}:
            return -0.04
        return -0.06

    def _prune_duplicate_tracks(self) -> None:
        ordered = sorted(self._tracks.values(), key=self._track_priority, reverse=True)
        remove: set[int] = set()
        for index, track in enumerate(ordered):
            if track.track_id in remove:
                continue
            for other in ordered[index + 1 :]:
                if other.track_id in remove:
                    continue
                if self._tracks_are_duplicates(track, other):
                    remove.add(other.track_id)
        for tid in remove:
            self._tracks.pop(tid, None)

    def _tracks_are_duplicates(self, first: _Track, second: _Track) -> bool:
        first_group = group_from_class(first.stable_class)
        second_group = group_from_class(second.stable_class)
        if "cue_stick" in {first_group, second_group}:
            return False
        if first_group not in {"cue", "solid", "stripe", "black"}:
            return False
        if second_group not in {"cue", "solid", "stripe", "black"}:
            return False
        dist = float(np.linalg.norm(first.center - second.center))
        radius = float(max(first.radius_px, second.radius_px, 1.0))
        if dist <= max(6.0, radius * 1.05):
            return True
        return iou_xyxy(first.bbox, second.bbox) >= 0.55

    def _track_priority(self, track: _Track) -> float:
        return float(track.confidence) + min(0.2, track.age * 0.01) - min(0.3, track.lost_frames * 0.08)
